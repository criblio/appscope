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

// forward declaration
static void deleteAllReqsFromTree(httpmatch_t *match);



struct _httpmatch_t {
    void *treeRoot;
    net_info const *netInfo; // NET_ENTRIES array of pointers to net_info
    list_t const *extraNetInfo; // list of pointers to net_info
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

    match->netInfo = netInfo;
    match->extraNetInfo = extraNetInfo;
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
    item->id = req->id;
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
        DBG("failed to add id " PRIu64 ". (insufficient mem)", item->id);
        goto err;
    }
    if (item != *ptr) {
        DBG("failed to add id " PRIu64 ". (duplicate exists)", item->id);
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
    scope_tdelete(item, &match->treeRoot, compare);
    match->freeReq(req);
    scope_free(item);
}

void
httpReqDelete(httpmatch_t *match, uint64_t id)
{
    if (!match) return;

    tree_node_t key = {.id = id, .map = NULL, .circBufCount = 0};
    tree_node_t **ptr = scope_tfind(&key, &match->treeRoot, compare);
    if (!ptr) {
        DBG("failed to delete id " PRIu64 ". (not found)");
        return;
    }

    deleteTreeNode(match,ptr);
}

static void
deleteAllReqsFromTree(httpmatch_t *match)
{
    while (match->treeRoot) {
        deleteTreeNode(match, match->treeRoot);
    }
}


static void
markAndDeleteOld(const void *nodep, VISIT which, int depth)
{
    tree_node_t *item = *(tree_node_t **)nodep;

    // if item is more than CircBufDepth old we can delete it.
    if (item->circBufCount + g_match->cbufSize > g_circBufCountNow) {
        deleteTreeNode(g_match, (tree_node_t **)nodep);
    }

/*
    // Look for new candidates to mark for future deletion
    if request.fd is 0..1023, look at g_netlist[request.fd].uid
        if request.uid != g_netlist[request.uid].uid, add curcBufCount
    else // we don't have an fd.  Look in g_extra_net_info_list
        if uid is not in g_extra_net_info_list add curcBufCount
*/

}

bool
httpReqExpire(httpmatch_t *match, uint64_t circBufCount)
{
    if (!match) return FALSE;

    g_match = match;
    g_circBufCountNow = circBufCount;
    twalk(match->treeRoot, markAndDeleteOld);
    return TRUE;
}


