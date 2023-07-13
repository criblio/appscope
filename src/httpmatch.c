#define _GNU_SOURCE
#include "dbg.h"
#include "httpmatch.h"
#include "scopestdlib.h"
#include "utils.h"

// Set to prime number of what seems like a reasonable size
#define HASH_TABLE_SIZE 257
//#define HASH_TABLE_SIZE 1


typedef struct _hashTable_t {
    uint64_t sockid;              // Unique for each instance of a socket.
    int sockfd;                   // socket descriptor.  -1 == "unknown socket"
    void *data;
    struct _hashTable_t *next;
    uint64_t circBufCount;        // non-zero means we're going to expire this
} hashTable_t;

typedef void (*freeData_fn)(void *);

typedef struct _store_t {
    hashTable_t *hashTable[HASH_TABLE_SIZE];
    net_info *netInfo;    // NET_ENTRIES array of pointers to net_info
    list_t *extraNetInfo; // list of pointers to net_info
    freeData_fn freeData;
    size_t cbufSize;
    struct {
        uint64_t totalSaves;
        uint64_t totalDeletes;
        uint64_t totalExpires;
        uint64_t maxListLen;
    } stats;
} store_t;

static store_t *
storeCreate(net_info const * const netInfo,
                list_t const * const extraNetInfo,
                freeData_fn freeData)
{
    if (!netInfo || !extraNetInfo || !freeData) return NULL;

    store_t *match = scope_calloc(1, sizeof(store_t));
    if (!match) {
        DBG(NULL);
        return NULL;
    }

    match->netInfo = (net_info *) netInfo;
    match->extraNetInfo = (list_t *)extraNetInfo;
    match->freeData = freeData;

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

static int
hashOfKey(uint64_t key) {
    return key % HASH_TABLE_SIZE;
}

static hashTable_t **
hashListForKey(httpmatch_t *match, uint64_t key)
{
    if (!match) return NULL;
    return &match->hashTable[hashOfKey(key)];
}


static void
deleteHashTableItem(httpmatch_t *match, hashTable_t **itemptr)
{
    if (!match || !itemptr || !*itemptr) return;
    hashTable_t *item = *itemptr;
    if (match->freeData && item->data) match->freeData(item->data);
    scope_free(item);
    *itemptr = NULL;
}

static void
deleteList(httpmatch_t *match, hashTable_t **head)
{
    if (!match || !head || !*head) return;
    hashTable_t *current = *head;
    while (current) {
        hashTable_t *next = current->next;
        deleteHashTableItem(match, &current);
        current = next;
    }
    *head = NULL;
}

static void
deleteAllLists(httpmatch_t *match)
{
    int i;
    for (i = 0; i < HASH_TABLE_SIZE; i++) {
        deleteList(match, &match->hashTable[i]);
    }
}

static void
storeDestroy(store_t **storeptr)
{
    if (!storeptr || !*storeptr) return;
    store_t *store = *storeptr;

    deleteAllLists(store);

    scope_free(store);
    *storeptr = NULL;
}

static hashTable_t *
createHashTableItem(void *data, uint64_t sockid, int sockfd)
{
    if (!data) return NULL;
    hashTable_t *item = scope_calloc(1, sizeof(*item));
    if (!item) return NULL;
    item->sockid = sockid;
    item->sockfd = sockfd;
    item->data = data;
    item->next = NULL;
    item->circBufCount = 0;
    return item;
}

static bool
addHashTableItemToList(hashTable_t **listptr, hashTable_t *item)
{
    if (!listptr || !item) return FALSE;
    hashTable_t *previous = NULL;
    hashTable_t *current = *listptr;
    while (current) {
        if (current->sockid == item->sockid) return FALSE;
        previous = current;
        current = current->next;
    }

    item->next = current;
    if (!previous) {
        *listptr = item;
    } else {
        previous->next = item;
    }
    return TRUE;
}

static bool
storeSave(store_t *store, void *data, uint64_t sockid, int sockfd)
{
    if (!store || !data ) return FALSE;

    hashTable_t **listptr = hashListForKey(store, sockid);

    hashTable_t *item = createHashTableItem(data, sockid, sockfd);
    if (!item) {
        DBG("failed to add id %" PRIu64 ". (insufficient mem)", sockid);
        return FALSE;
    }

    if (!addHashTableItemToList(listptr, item)) {
        DBG("Found duplicate data.  Deleting new data.");
        scope_free(item);
        return FALSE;
    }

    store->stats.totalSaves++;
    return TRUE;
}

static hashTable_t *
findHashTableItemFromList(hashTable_t *head, uint64_t sockid)
{
    if (!head) return NULL;
    hashTable_t *current = head;
    while (current) {
        if (current->sockid == sockid) return current;
        current = current->next;
    }
    return NULL;
}

static http_map *
storeGet(httpmatch_t *match, uint64_t sockid)
{
    if (!match) return NULL;

    hashTable_t *list = *hashListForKey(match, sockid);
    hashTable_t *hashTableItem = findHashTableItemFromList(list, sockid);
    if (!hashTableItem) return NULL;
    return hashTableItem->data;
}

static bool
storeDelete(httpmatch_t *match, uint64_t sockid)
{
    if (!match) return FALSE;

    hashTable_t **listptr = hashListForKey(match, sockid);
    hashTable_t *previous = NULL;
    hashTable_t *current = *listptr;
    while (current) {
        hashTable_t *next = current->next;
        if (current->sockid == sockid) {
            if (!previous) {
                *listptr = next;
            } else {
                previous->next = next;
            }
            match->stats.totalDeletes++;
            deleteHashTableItem(match, &current);
            return TRUE;
        }
        previous = current;
        current = next;
    }

//    DBG("Request to delete was never found.");
    return TRUE;
}

static bool
storeExpire(httpmatch_t *match, uint64_t circBufCount, bool circBufWasEmptied)
{
    if (!match) return FALSE;

    // Loop through each list
    int i;
    for (i = 0; i < HASH_TABLE_SIZE; i++) {

        hashTable_t **listptr = &match->hashTable[i];
        hashTable_t *previous = NULL;
        hashTable_t *current = *listptr;
        int listLen = 0;

        // Loop through each item in the last
        while (current) {
            hashTable_t *next = current->next;

            // if this has been marked for deletion (circBufCount != 0)
            // and the circbufWasEmptied or this is more than cbufSize events old,
            // we can safely delete it.
            bool circBufHasWrapped =
                (current->circBufCount + match->cbufSize) < circBufCount;
            if (current->circBufCount &&
                  (circBufWasEmptied || circBufHasWrapped)) {

                if (!previous) {
                    *listptr = next;
                } else {
                    previous->next = next;
                }
                match->stats.totalExpires++;
                deleteHashTableItem(match, &current);

            } else {

                // netinfo and extraNetInfo are references to what sockets
                // are currently active on the datapath side of things.
                // If the socket descriptor is not in the range of the netInfo
                // then we'll have to look in extraNetInfo
                int sockfd = current->sockfd;
                uint64_t sockid = current->sockid;
                net_info *net = NULL;
                if (sockfd < 0 || sockfd >= NET_ENTRIES) {
                    net = lstFind(match->extraNetInfo, sockid);
                } else {
                    net = &match->netInfo[sockfd];
                }

                // If the UID is not currently in use by the datapath, mark it for
                // deletion by saving the circBufCount at this time.
                if ((!net || net->uid != sockid || !net->active) && !current->circBufCount) {
                    current->circBufCount = circBufCount;
                }
            }

            previous = current;
            current = next;
            listLen++;
        }

        if (listLen > match->stats.maxListLen) {
            match->stats.maxListLen = listLen;
        }

    }

    return TRUE;
}

//////////////////////

httpmatch_t *
httpMatchCreate(net_info const * const netInfo, list_t const * const extraNetInfo, freeReq_fn freeReq)
{
    return (httpmatch_t *)storeCreate(netInfo, extraNetInfo, (freeData_fn)freeReq);
}

void
httpMatchDestroy(httpmatch_t **matchptr)
{
    storeDestroy((store_t **)matchptr);
}

bool
httpReqSave(httpmatch_t *match, http_map *req)
{
    if (!req) return FALSE;
    uint64_t sockid = req->id.uid;
    int sockfd = req->id.sockfd;
    return storeSave((store_t *)match, req, sockid, sockfd);
}

http_map *
httpReqGet(httpmatch_t *match, uint64_t sockid)
{
    return (http_map *)storeGet((store_t *)match, sockid);
}

bool
httpReqDelete(httpmatch_t *match, uint64_t sockid)
{
    return storeDelete((store_t *)match, sockid);
}

bool
httpReqExpire(httpmatch_t *match, uint64_t circBufCount, bool circBufWasEmptied)
{
    return storeExpire((store_t *)match, circBufCount, circBufWasEmptied);
}

//////////////////////

channelstore_t *
channelStoreCreate(net_info const * const netInfo, list_t const * const extraNetInfo, freeChannel_fn freeChannel)
{
    return (channelstore_t *)storeCreate(netInfo, extraNetInfo, (freeData_fn)freeChannel);
}

void
channelStoreDestroy(channelstore_t **chanStore)
{
    storeDestroy((store_t **)chanStore);
}

bool
channelSave(channelstore_t *chanStore, http2Channel_t *channel, uint64_t sockid, int sockfd)
{
    return storeSave((store_t*)chanStore, channel, sockid, sockfd);
}

http2Channel_t *
channelGet(channelstore_t *chanStore, uint64_t sockid)
{
    return (http2Channel_t *)storeGet((store_t *)chanStore, sockid);
}

bool
channelDelete(channelstore_t *chanStore, uint64_t sockid)
{
    return storeDelete((store_t *)chanStore, sockid);
}

bool
channelExpire(channelstore_t *chanStore, uint64_t circBufCount, bool circBufWasEmptied)
{
    return storeExpire((store_t *)chanStore, circBufCount, circBufWasEmptied);
}


