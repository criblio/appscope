#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dbg.h"
#include "httpmatch.h"
#include "scopestdlib.h"
#include "test.h"

#define NET_ENTRIES 1024
net_info *g_netinfo = NULL;
list_t *g_extra_net_info_list = NULL;

static int
setup(void **state)
{
    g_netinfo = scope_calloc(NET_ENTRIES, sizeof(net_info));
    g_extra_net_info_list = lstCreate(scope_free);

    return groupSetup(state);
}

static int
teardown(void **state)
{
    scope_free(g_netinfo);
    lstDestroy(&g_extra_net_info_list);

    return groupTeardown(state);
}

static http_map *
newReq(uint64_t id, int fd)
{
    http_map *map = scope_calloc(1, sizeof(http_map));
    assert_non_null(map);
    map->id.uid = id;
    map->id.sockfd = fd;
    return map;
}

static void
freeReq(http_map *req)
{
    assert_non_null(req);
    scope_free(req);
}


static void
httpMatchCreateReturnsNullWithNullParams(void **state)
{
   assert_null(httpMatchCreate(NULL, g_extra_net_info_list, freeReq));
   assert_null(httpMatchCreate(g_netinfo, NULL, freeReq));
   assert_null(httpMatchCreate(g_netinfo, g_extra_net_info_list, NULL));
}

static void
httpMatchCreateReturnsNonNull(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    assert_non_null(match);
    httpMatchDestroy(&match);
    assert_null(match);
}

static void
httpMatchDestroyForNullDoesNotCrash(void **state)
{
    httpMatchDestroy(NULL);
    httpmatch_t *match = NULL;
    httpMatchDestroy(&match);
}

static void
httpReqSaveReturnsFalseWithNullParams(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd = newReq(531, 3);

    assert_false(httpReqSave(NULL, reqToAdd));
    assert_false(httpReqSave(match, NULL));

    freeReq(reqToAdd);
    httpMatchDestroy(&match);
}

static void
httpReqSaveAddOfDuplicateFails(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd1 = newReq(531, 3);
    http_map *reqToAdd2 = newReq(531, 3);
    assert_true(httpReqSave(match, reqToAdd1));

    assert_false(httpReqSave(match, reqToAdd2));
    assert_int_equal(dbgCountMatchingLines("src/httpmatch.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    http_map *reqRetrieved = httpReqGet(match, 531);
    assert_ptr_equal(reqRetrieved, reqToAdd1); // First one stays

    // reqToAdd1 is still there, but should be cleaned up here
    httpMatchDestroy(&match);
    freeReq(reqToAdd2);
}

static void
httpReqSaveDoesNotCrash(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    assert_non_null(match);

    http_map *reqToAdd = newReq(531, 3);
    httpReqSave(match, reqToAdd);

    http_map *reqRetrieved = httpReqGet(match, 531);
    assert_ptr_equal(reqToAdd, reqRetrieved);
    httpReqDelete(match, 531);

    httpMatchDestroy(&match);
}

// I know that the code currently uses this many buckets, so
// I'm using this to ensure that we have to manage hash conflicts
// during the testing here.
#define HASH_PRIME 257

static void
httpReqSaveAndGetWorks(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd1 = newReq(1234, 4);
    http_map *reqToAdd2 = newReq(1235, 4);
    http_map *reqToAdd3 = newReq(1235+HASH_PRIME, 4);
    assert_true(httpReqSave(match, reqToAdd1));
    assert_true(httpReqSave(match, reqToAdd2));
    assert_true(httpReqSave(match, reqToAdd3));

    http_map *req1fetched = httpReqGet(match, 1234);
    http_map *req2fetched = httpReqGet(match, 1235);
    http_map *req3fetched = httpReqGet(match, 1235+HASH_PRIME);

    // deleting something that isn't there "succeeds"
    assert_true(httpReqDelete(match, 3333));

    // geting something that isn't there returns NULL
    assert_null(httpReqGet(match, 1234+HASH_PRIME));
    assert_null(httpReqGet(match, 1111));

    assert_ptr_equal(reqToAdd1, req1fetched);
    assert_ptr_equal(reqToAdd2, req2fetched);
    assert_ptr_equal(reqToAdd3, req3fetched);

    httpReqDelete(match, 1234);
    httpReqDelete(match, 1235+HASH_PRIME);
    httpReqDelete(match, 1235);

    httpMatchDestroy(&match);
}

static void
httpReqExpireWithoutAnyRequests(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    httpReqExpire(match, 123456789, FALSE);
    httpMatchDestroy(&match);
}

static void
httpReqExpireRequestsFromCircBufCount(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd1 = newReq(1234, 4);
    http_map *reqToAdd2 = newReq(1235, 4);
    http_map *reqToAdd3 = newReq(1235+HASH_PRIME, 4);
    assert_true(httpReqSave(match, reqToAdd1));
    assert_true(httpReqSave(match, reqToAdd2));
    assert_true(httpReqSave(match, reqToAdd3));

    http_map *req1fetched = httpReqGet(match, 1234);
    http_map *req2fetched = httpReqGet(match, 1235);
    http_map *req3fetched = httpReqGet(match, 1235+HASH_PRIME);

    assert_ptr_equal(reqToAdd1, req1fetched);
    assert_ptr_equal(reqToAdd2, req2fetched);
    assert_ptr_equal(reqToAdd3, req3fetched);

    httpReqExpire(match, 4545454, FALSE);
    httpReqExpire(match, 4545454 + 1000000, FALSE);

    assert_null(httpReqGet(match, 1234));
    assert_null(httpReqGet(match, 1235));
    assert_null(httpReqGet(match, 1235+HASH_PRIME));

    httpMatchDestroy(&match);
}

static void
httpReqExpireRequestsAtDifferentTimes(void **state)
{
    setenv("SCOPE_QUEUE_LENGTH", "5", 1);

    int red = 1235;
    int blue = red + HASH_PRIME;

    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);

    // add red
    http_map *req1 = newReq(red, 4);
    assert_true(httpReqSave(match, req1));

    // this sets expiration for red to 7
    httpReqExpire(match, 1, FALSE);
    assert_non_null(httpReqGet(match, red));

    // add blue
    http_map *req2 = newReq(blue, 4);
    assert_true(httpReqSave(match, req2));

    // red is almost expiring, but not yet
    // sets expiration for blue to 12
    httpReqExpire(match, 6, FALSE);
    assert_non_null(httpReqGet(match, red));

    // now red should expire, but blue shouldn't yet
    httpReqExpire(match, 7, FALSE);
    // now since red is expired it should no longer be there
    assert_null(httpReqGet(match, red));
    // but blue should still be around until 12
    assert_non_null(httpReqGet(match, blue));

    httpReqExpire(match, 12, FALSE);
    assert_null(httpReqGet(match, blue));

    unsetenv("SCOPE_QUEUE_LENGTH");
    httpMatchDestroy(&match);
}


static void
httpReqExpireRequestsFromEmptyFlag(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd1 = newReq(1234, 4);
    http_map *reqToAdd2 = newReq(1235, 4);
    http_map *reqToAdd3 = newReq(1235+HASH_PRIME, 4);
    assert_true(httpReqSave(match, reqToAdd1));
    assert_true(httpReqSave(match, reqToAdd2));
    assert_true(httpReqSave(match, reqToAdd3));

    http_map *req1fetched = httpReqGet(match, 1234);
    http_map *req2fetched = httpReqGet(match, 1235);
    http_map *req3fetched = httpReqGet(match, 1235+HASH_PRIME);

    assert_ptr_equal(reqToAdd1, req1fetched);
    assert_ptr_equal(reqToAdd2, req2fetched);
    assert_ptr_equal(reqToAdd3, req3fetched);

    httpReqExpire(match, 4545454, TRUE);
    httpReqExpire(match, 4545454, TRUE);

    assert_null(httpReqGet(match, 1234));
    assert_null(httpReqGet(match, 1235));
    assert_null(httpReqGet(match, 1235+HASH_PRIME));

    httpMatchDestroy(&match);
}
int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(httpMatchCreateReturnsNullWithNullParams),
        cmocka_unit_test(httpMatchCreateReturnsNonNull),
        cmocka_unit_test(httpMatchDestroyForNullDoesNotCrash),
        cmocka_unit_test(httpReqSaveReturnsFalseWithNullParams),
        cmocka_unit_test(httpReqSaveAddOfDuplicateFails),
        cmocka_unit_test(httpReqSaveDoesNotCrash),
        cmocka_unit_test(httpReqSaveAndGetWorks),
        cmocka_unit_test(httpReqExpireWithoutAnyRequests),
        cmocka_unit_test(httpReqExpireRequestsFromCircBufCount),
        cmocka_unit_test(httpReqExpireRequestsAtDifferentTimes),
        cmocka_unit_test(httpReqExpireRequestsFromEmptyFlag),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, setup, teardown);
}


