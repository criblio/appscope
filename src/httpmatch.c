#define _GNU_SOURCE
#include "dbg.h"
#include "httpmatch.h"
#include "scopestdlib.h"
#include "utils.h"

// Set to prime number of what seems like a reasonable size
#define HASH_TABLE_SIZE 257
//#define HASH_TABLE_SIZE 1

typedef struct _hashTable_t {
    http_map *req;
    struct _hashTable_t *next;
    uint64_t circBufCount;
} hashTable_t;

struct _httpmatch_t {
    hashTable_t *hashTable[HASH_TABLE_SIZE];
    net_info *netInfo;    // NET_ENTRIES array of pointers to net_info
    list_t *extraNetInfo; // list of pointers to net_info
    freeReq_fn freeReq;
    size_t cbufSize;
    struct {
        uint64_t totalSaves;
        uint64_t totalDeletes;
        uint64_t totalExpires;
        uint64_t maxListLen;
    } stats;
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
        return NULL;
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

static uint64_t
keyOfReq(http_map *req) {
    return req->id.uid;
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
    if (!itemptr || !*itemptr) return;
    hashTable_t *item = *itemptr;
    if (match->freeReq && item->req) match->freeReq(item->req);
    scope_free(item);
    *itemptr = NULL;
}

static void
deleteList(httpmatch_t *match, hashTable_t **head)
{
    if (!head || !*head) return;
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

void
httpMatchDestroy(httpmatch_t **matchptr)
{
    if (!matchptr || !*matchptr) return;
    httpmatch_t *match = *matchptr;

    deleteAllLists(match);

    scope_free(match);
    *matchptr = NULL;
}

static hashTable_t *
createHashTableItem(http_map *req)
{
    if (!req) return NULL;
    hashTable_t *item = scope_calloc(1, sizeof(*item));
    if (!item) return NULL;
    item->req = req;
    item->next = NULL;
    item->circBufCount = 0;
    return item;
}

static bool
addHashTableItemToList(hashTable_t **listptr, hashTable_t *item)
{
    if (!listptr || !item) return FALSE;
    uint64_t itemKey = keyOfReq(item->req);
    hashTable_t *previous = NULL;
    hashTable_t *current = *listptr;
    while (current) {
        uint64_t currentKey = keyOfReq(current->req);
        if (currentKey == itemKey) return FALSE;
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

bool
httpReqSave(httpmatch_t *match, http_map *req)
{
    if (!match || !req ) return FALSE;

    uint64_t key = keyOfReq(req);
    hashTable_t **listptr = hashListForKey(match, key);

    hashTable_t *item = createHashTableItem(req);
    if (!item) {
        DBG("failed to add id %" PRIu64 ". (insufficient mem)", key);
        match->freeReq(req);
        return FALSE;
    }

    if (!addHashTableItemToList(listptr, item)) {
        DBG("Found duplicate req.  Deleting new req.");
        deleteHashTableItem(match, &item);
        return FALSE;
    }

    match->stats.totalSaves++;
    return TRUE;
}

static hashTable_t *
findHashTableItemFromList(hashTable_t *head, uint64_t key)
{
    if (!head) return NULL;
    hashTable_t *current = head;
    while (current) {
        uint64_t currentKey = keyOfReq(current->req);
        if (currentKey == key) return current;
        current = current->next;
    }
    return NULL;
}

http_map *
httpReqGet(httpmatch_t *match, uint64_t key)
{
    if (!match) return NULL;

    hashTable_t *list = *hashListForKey(match, key);
    hashTable_t *hashTableItem = findHashTableItemFromList(list, key);
    if (!hashTableItem) return NULL;
    return hashTableItem->req;
}

bool
httpReqDelete(httpmatch_t *match, uint64_t key)
{
    if (!match) return FALSE;

    hashTable_t **listptr = hashListForKey(match, key);
    hashTable_t *previous = NULL;
    hashTable_t *current = *listptr;
    while (current) {
        hashTable_t *next = current->next;
        uint64_t currentKey = keyOfReq(current->req);
        if (currentKey == key) {
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

bool
httpReqExpire(httpmatch_t *match, uint64_t circBufCount, bool circBufWasEmptied)
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
                int sd = current->req->id.sockfd;
                uint64_t currentKey = keyOfReq(current->req);
                net_info *net = NULL;
                if (sd < 0 || sd >= NET_ENTRIES) {
                    net = lstFind(match->extraNetInfo, currentKey);
                } else {
                    net = &match->netInfo[sd];
                }

                // If the UID is not currently in use by the datapath, mark it for
                // deletion by saving the circBufCount at this time.
                if ((!net || net->uid != currentKey) && !current->circBufCount) {
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

