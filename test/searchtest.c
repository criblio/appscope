#include <stdio.h>
#include <string.h>
#include "search.h"
#include "test.h"

static void
searchCompReturnsNullForNullArg(void** state)
{
    search_t *handle = searchComp(NULL);
    assert_null(handle);
}

static void
searchCompReturnsNullForEmptyString(void** state)
{
    search_t *handle = searchComp("");
    assert_null(handle);
}

static void
searchCompReturnsValidPtrInHappyPath(void** state)
{
    search_t *handle = searchComp("hey");
    assert_non_null (handle);

    searchFree(&handle);
}

static void
searchFreeHandlesNullArg(void** state)
{
    search_t *handle = searchComp("hey");
    assert_non_null (handle);

    // test searchFree too
    searchFree(&handle);
    assert_null(handle);
    searchFree(&handle);
    searchFree(NULL);
}

static void
searchLenReturnsZeroForNullArg(void** state)
{
    assert_int_equal(searchLen(NULL), 0);
}

static void
searchLenReturnsLengthOfOriginalStr(void** state)
{
    search_t *handle = searchComp("hey");
    assert_int_equal(searchLen(handle), strlen("hey"));
    searchFree(&handle);
}

static void
searchExecReturnsMinusOneForBadArgs(void** state)
{
    char *buf = "hey";
    search_t *handle = searchComp("hey");

    assert_int_equal(searchExec(NULL, buf, sizeof(buf)), -1);

    assert_int_equal(searchExec(handle, NULL, sizeof(buf)), -1);

    assert_int_equal(searchExec(handle, buf, -1), -1);

    searchFree(&handle);
}

static void
searchExecReturnsExpectedResultsInHappyPath(void** state)
{
    char *buf = "hey";
    search_t *handle = searchComp("hey");

    // exact match
    assert_int_equal(searchExec(handle, buf, strlen(buf)), 0);

    // almost match; handle longer than buf size
    assert_int_equal(searchExec(handle, buf, strlen(buf)-1), -1);

    // match at end of buf
    assert_int_equal(searchExec(handle, " hey", strlen(" hey")), 1);
    assert_int_equal(searchExec(handle, "  hey", strlen("  hey")), 2);

    // first possible match
    assert_int_equal(searchExec(handle, " heyhey", strlen(" heyhey")), 1);
    searchFree(&handle);

    // Minimum possible handles
    handle = searchComp("h");
    assert_int_equal(searchExec(handle, buf, strlen(buf)), 0);
    searchFree(&handle);

    handle = searchComp("e");
    assert_int_equal(searchExec(handle, buf, strlen(buf)), 1);
    searchFree(&handle);

    handle = searchComp("y");
    assert_int_equal(searchExec(handle, buf, strlen(buf)), 2);
    searchFree(&handle);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(searchCompReturnsNullForNullArg),
        cmocka_unit_test(searchCompReturnsNullForEmptyString),
        cmocka_unit_test(searchCompReturnsValidPtrInHappyPath),
        cmocka_unit_test(searchFreeHandlesNullArg),
        cmocka_unit_test(searchLenReturnsZeroForNullArg),
        cmocka_unit_test(searchLenReturnsLengthOfOriginalStr),
        cmocka_unit_test(searchExecReturnsMinusOneForBadArgs),
        cmocka_unit_test(searchExecReturnsExpectedResultsInHappyPath),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}

