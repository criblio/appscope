#include <stdio.h>
#include <string.h>
#include "search.h"
#include "test.h"

static void
needleCreateReturnsNullForNullArg(void** state)
{
    needle_t *needle = needleCreate(NULL);
    assert_null(needle);
}

static void
needleCreateReturnsNullForEmptyString(void** state)
{
    needle_t *needle = needleCreate("");
    assert_null(needle);
}

static void
needleCreateReturnsValidPtrInHappyPath(void** state)
{
    needle_t *needle = needleCreate("hey");
    assert_non_null (needle);

    needleDestroy(&needle);
}

static void
needleDestroyHandlesNullArg(void** state)
{
    needle_t *needle = needleCreate("hey");
    assert_non_null (needle);

    // test needleDestroy too
    needleDestroy(&needle);
    assert_null(needle);
    needleDestroy(&needle);
    needleDestroy(NULL);
}

static void
needleLenReturnsZeroForNullArg(void** state)
{
    assert_int_equal(needleLen(NULL), 0);
}

static void
needleLenReturnsLengthOfOriginalStr(void** state)
{
    needle_t *needle = needleCreate("hey");
    assert_int_equal(needleLen(needle), strlen("hey"));
    needleDestroy(&needle);
}

static void
needleFindReturnsMinusOneForBadArgs(void** state)
{
    char *buf = "hey";
    needle_t *needle = needleCreate("hey");

    assert_int_equal(needleFind(NULL, buf, sizeof(buf)), -1);

    assert_int_equal(needleFind(needle, NULL, sizeof(buf)), -1);

    assert_int_equal(needleFind(needle, buf, -1), -1);

    needleDestroy(&needle);
}

static void
needleFindReturnsExpectedResultsInHappyPath(void** state)
{
    char *buf = "hey";
    needle_t *needle = needleCreate("hey");

    // exact match
    assert_int_equal(needleFind(needle, buf, strlen(buf)), 0);

    // almost match; needle longer than buf size
    assert_int_equal(needleFind(needle, buf, strlen(buf)-1), -1);

    // match at end of buf
    assert_int_equal(needleFind(needle, " hey", strlen(" hey")), 1);
    assert_int_equal(needleFind(needle, "  hey", strlen("  hey")), 2);

    // first possible match
    assert_int_equal(needleFind(needle, " heyhey", strlen(" heyhey")), 1);
    needleDestroy(&needle);

    // Minimum possible needles
    needle = needleCreate("h");
    assert_int_equal(needleFind(needle, buf, strlen(buf)), 0);
    needleDestroy(&needle);

    needle = needleCreate("e");
    assert_int_equal(needleFind(needle, buf, strlen(buf)), 1);
    needleDestroy(&needle);

    needle = needleCreate("y");
    assert_int_equal(needleFind(needle, buf, strlen(buf)), 2);
    needleDestroy(&needle);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(needleCreateReturnsNullForNullArg),
        cmocka_unit_test(needleCreateReturnsNullForEmptyString),
        cmocka_unit_test(needleCreateReturnsValidPtrInHappyPath),
        cmocka_unit_test(needleDestroyHandlesNullArg),
        cmocka_unit_test(needleLenReturnsZeroForNullArg),
        cmocka_unit_test(needleLenReturnsLengthOfOriginalStr),
        cmocka_unit_test(needleFindReturnsMinusOneForBadArgs),
        cmocka_unit_test(needleFindReturnsExpectedResultsInHappyPath),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}

