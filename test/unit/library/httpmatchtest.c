#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dbg.h"
#include "httpmatch.h"
#include "scopestdlib.h"
#include "test.h"


//   make FSAN=1 libtest
//   LD_LIBRARY_PATH=contrib/build/cmocka/src/ test/linux/httpmatchtest


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
newReq(uint64_t id)
{
    http_map *map = scope_calloc(1, sizeof(http_map));
    assert_non_null(map);
    map->id = id;
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
    http_map *reqToAdd = newReq(531);

    assert_false(httpReqSave(NULL, reqToAdd));
    assert_false(httpReqSave(match, NULL));

    freeReq(reqToAdd);
    httpMatchDestroy(&match);
}

static void
httpReqSaveAddOfDuplicateFails(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    http_map *reqToAdd1 = newReq(531);
    http_map *reqToAdd2 = newReq(531);
    assert_true(httpReqSave(match, reqToAdd1));

    assert_false(httpReqSave(match, reqToAdd2));
    assert_int_equal(dbgCountMatchingLines("src/httpmatch.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    http_map *reqRetrieved = httpReqGet(match, 531);
    assert_ptr_equal(reqRetrieved, reqToAdd1); // First one stays

    // reqToAdd1 is still there, but should be cleaned up here
    httpMatchDestroy(&match);
}

static void
httpReqSaveDoesNotCrash(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    assert_non_null(match);

    http_map *reqToAdd = newReq(531);
    httpReqSave(match, reqToAdd);

    http_map *reqRetrieved = httpReqGet(match, 531);
    assert_ptr_equal(reqToAdd, reqRetrieved);
    httpReqDelete(match, 531);

    httpMatchDestroy(&match);
}

static void
httpReqExpireWithoutAnyRequests(void **state)
{
    httpmatch_t *match = httpMatchCreate(g_netinfo, g_extra_net_info_list, freeReq);
    httpReqExpire(match, 123456789);
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
        cmocka_unit_test(httpReqExpireWithoutAnyRequests),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, setup, teardown);
}


