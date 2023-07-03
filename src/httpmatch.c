#define _GNU_SOURCE
#include "dbg.h"
#include "httpmatch.h"
#include "scopestdlib.h"
#include "utils.h"

typedef struct {
    uint64_t id;
    http_map *map;
    uint64_t circBufCount;
} tree_node_t;

static int
compare(const void *itema, const void *itemb)
{
    tree_node_t *ia = (tree_node_t *)itema;
    tree_node_t *ib = (tree_node_t *)itemb;
    if (ia->id < ib->id) return -1;
    if (ia->id > ib->id) return 1;
    return 0;
}

// Yuck.  If musl supplied twalk_r, we could avoid having these globals.
// They're used to get info from httpReqExpire to markAndDeleteOld
httpmatch_t *g_match = NULL;
uint64_t g_circBufCountNow = 0;
bool g_circBufWasEmptied = FALSE;

// forward declaration
static void deleteAllReqsFromTree(httpmatch_t *match);



struct _httpmatch_t {
    void *treeRoot;
    net_info *netInfo;    // NET_ENTRIES array of pointers to net_info
    list_t *extraNetInfo; // list of pointers to net_info
    freeReq_fn freeReq;
    size_t cbufSize;
};

httpmatch_t *
httpMatchCreate(net_info const * const netInfo,
                list_t const * const extraNetInfo,
                freeReq_fn freeReq)
{
    if (!netInfo || !extraNetInfo || !freeReq) return NULL;

    httpmatch_t *match = scope_calloc(1, sizeof(httpmatch_t));
    if (!match) {
        DBG(NULL);
    }

    match->netInfo = (net_info *) netInfo;
    match->extraNetInfo = (list_t *)extraNetInfo;
    match->freeReq = freeReq;

    // Not great, but duplicated from ctlCreate()
    match->cbufSize = DEFAULT_CBUF_SIZE;
    char *qlen_str;
    if ((qlen_str = fullGetEnv("SCOPE_QUEUE_LENGTH")) != NULL) {
        unsigned long qlen;
        scope_errno = 0;
        qlen = scope_strtoul(qlen_str, NULL, 10);
        if (!scope_errno && qlen) {
            match->cbufSize = qlen;
        }
    }

    return match;
}

void
httpMatchDestroy(httpmatch_t **matchptr)
{
    if (!matchptr || !*matchptr) return;
    httpmatch_t *match = *matchptr;

    deleteAllReqsFromTree(match);

    scope_free(match);
    *matchptr = NULL;
}



static tree_node_t *
createTreeItem(http_map *req)
{
    tree_node_t *item = scope_calloc(1, sizeof(*item));
    if (!item) return NULL;
    item->id = req->id.uid;
    item->map = req;
    item->circBufCount = 0;

    return item;
}

bool
httpReqSave(httpmatch_t *match, http_map *req)
{
    if (!match || !req ) return FALSE;

    tree_node_t *item = createTreeItem(req);
    if (!item) goto err;

    tree_node_t **ptr = scope_tsearch(item, &match->treeRoot, compare);
    if (!ptr) {
        DBG("failed to add id %" PRIu64 ". (insufficient mem)", item->id);
        goto err;
    }
    if (item != *ptr) {
        DBG("failed to add id %" PRIu64 ". (duplicate exists)", item->id);
        goto err;
    }
    return  TRUE;

err:
    match->freeReq(req);
    if (item) scope_free(item);
    return FALSE;
}

http_map *
httpReqGet(httpmatch_t *match, uint64_t id)
{
    if (!match) return NULL;

    tree_node_t key = {.id = id, .map = NULL, .circBufCount = 0};
    tree_node_t **ptr = scope_tfind(&key, &match->treeRoot, compare);
    if (!ptr || !*ptr) return NULL;
    return (*ptr)->map;
}



static void
deleteTreeNode(httpmatch_t *match, tree_node_t ** itemptr)
{
    if (!itemptr || !*itemptr) return;

    tree_node_t *item = *itemptr;
    http_map *req = item->map;
    tree_node_t **ptr = scope_tdelete(item, &match->treeRoot, compare);
    if (!ptr) {
        DBG("failed to delete tree node");
        return;
    }
    match->freeReq(req);
    scope_free(item);
}

bool
httpReqDelete(httpmatch_t *match, uint64_t id)
{
    if (!match) return FALSE;

    tree_node_t key = {.id = id, .map = NULL, .circBufCount = 0};
    tree_node_t **ptr = scope_tfind(&key, &match->treeRoot, compare);
    if (!ptr) {
        // nothing to delete
        return TRUE;
    }

    deleteTreeNode(match,ptr);
    return TRUE;
}

static void
deleteAllReqsFromTree(httpmatch_t *match)
{
    while (match->treeRoot) {
        deleteTreeNode(match, match->treeRoot);
    }
}


// Behavior is undefined if we delete nodes while walking the tree.
// Sooo... we store a list of things to delete while walking the tree
// and delete them when we're done.
typedef struct _nodelist_t {
    uint64_t uid;
    struct _nodelist_t *next;
} nodelist_t;
nodelist_t *g_rememberedTreeNodes = NULL;

static void
rememberTreeNodeToDelete(uint64_t uid)
{
    nodelist_t *newNode = scope_calloc (1, sizeof (*newNode));
    newNode->uid = uid;
    newNode->next = NULL;

    // add to end
    if (!g_rememberedTreeNodes) {
        g_rememberedTreeNodes = newNode;
        return;
    }

    nodelist_t *end = g_rememberedTreeNodes;
    while (end->next) {
        end = end->next;
    }
    end->next = newNode;

}

static void
deleteRememberedTreeNodes(void)
{
    while (g_rememberedTreeNodes) {
        nodelist_t *next = g_rememberedTreeNodes->next;
        httpReqDelete(g_match, g_rememberedTreeNodes->uid);
        scope_free(g_rememberedTreeNodes);
        g_rememberedTreeNodes = next;
    }
}

static void
markAndDeleteOld(const void *nodep, VISIT which, int depth)
{
    // look at each node in the tree once
    if (which != leaf && which != endorder) return;

    // item is one tree_node as we walk all nodes of the tree.
    tree_node_t *item = *(tree_node_t **)nodep;

    // if this has been marked for deletion (circBufCount != 0)
    // and the circbufWasEmptied or this is more than cbufSize events old,
    // we can safely delete it.
    bool circBufHasWrapped =
        (item->circBufCount + g_match->cbufSize) < g_circBufCountNow;
    if (item->circBufCount &&
          (g_circBufWasEmptied || circBufHasWrapped)) {
        rememberTreeNodeToDelete(item->id);
    }

    // netinfo and extraNetInfo are references to what sockets
    // are currently active on the datapath side of things.
    // If the socket descriptor is not in the range of the netInfo
    // then we'll have to look in extraNetInfo
    int sd = item->map->id.sockfd;
    net_info *net = NULL;
    if (sd < 0 || sd >= NET_ENTRIES) {
        net = lstFind(g_match->extraNetInfo, item->id);
    } else {
        net = &g_match->netInfo[sd];
    }

    // If the UID is not currently in use by the datapath, mark it for
    // deletion by saving the circBufCount at this time.
    if ((!net || net->uid != item->id) && !item->circBufCount) {
        item->circBufCount = g_circBufCountNow;
    }
}

bool
httpReqExpire(httpmatch_t *match, uint64_t circBufCount, bool circBufWasEmptied)
{
    if (!match) return FALSE;

    g_match = match;
    g_circBufCountNow = circBufCount;
    g_circBufWasEmptied = circBufWasEmptied;

    g_rememberedTreeNodes = NULL;
    twalk(match->treeRoot, markAndDeleteOld);
    deleteRememberedTreeNodes();
    return TRUE;
}

