#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "strset.h"
#include "test.h"

static void
strSetCreateReturnsNonNull(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);
    strSetDestroy(&set);

    // Test that strSetDestroy changes the value of set to null
    assert_null(set);
}

static void
strSetDestroyOfNullSetDoesNotCrash(void** state)
{
    strSetDestroy(NULL);

    strset_t *set = NULL;
    strSetDestroy(&set);
}

static void
strSetAddOnNullSetReturnsFalse(void **state)
{
    assert_false(strSetAdd(NULL, "Hey"));
}

static void
strSetAddOfNullStrReturnsFalse(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_false(strSetAdd(set, NULL));
    strSetDestroy(&set);
}

static void
strSetAddNewElementReturnsTrue(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);

    assert_true(strSetAdd(set, "Hey"));

    strSetDestroy(&set);
}

static void
strSetAddDupElementReturnsFalse(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);
    assert_int_equal(strSetEntryCount(set), 0);

    assert_true(strSetAdd(set, "Hey"));
    assert_int_equal(strSetEntryCount(set), 1);

    // Try to add "Hey" again.  It should return null.
    assert_false(strSetAdd(set, "Hey"));
    assert_int_equal(strSetEntryCount(set), 1);

    // Try to add "Hey" from a string in a different location.
    // This ensures that the location of the string doesn't
    // matter at all.  Only the contents of the string should matter.
    const char hey[16] = "Hey";
    assert_false(strSetAdd(set, hey));
    assert_int_equal(strSetEntryCount(set), 1);

    strSetDestroy(&set);
    assert_int_equal(strSetEntryCount(set), 0);
}

static void
strSetAddIsNotCaseSensitive(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);

    assert_true(strSetAdd(set, "Hey"));
    assert_true(strSetAdd(set, "hey"));
    assert_true(strSetAdd(set, "HEY"));
    assert_int_equal(strSetEntryCount(set), 3);

    strSetDestroy(&set);
}


static void
strSetAddGrowsWithoutCrashing(void **state)
{
    strset_t *set = strSetCreate(0);
    assert_non_null(set);
    assert_int_equal(strSetEntryCount(set), 0);

    assert_true(strSetAdd(set, "one"));
    assert_true(strSetAdd(set, "two"));
    assert_true(strSetAdd(set, "three"));
    assert_true(strSetAdd(set, "four"));
    assert_true(strSetAdd(set, "five"));
    assert_true(strSetAdd(set, "six"));
    assert_true(strSetAdd(set, "seven"));
    assert_true(strSetAdd(set, "eight"));
    assert_true(strSetAdd(set, "nine"));
    assert_true(strSetAdd(set, "ten"));
    assert_int_equal(strSetEntryCount(set), 10);

    assert_true(strSetContains(set, "one"));
    assert_true(strSetContains(set, "ten"));
    strSetDestroy(&set);
}

static void
strSetContainsOfNullReturnsFalse(void **state)
{
    assert_false(strSetContains(NULL, "Hey"));

    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_false(strSetContains(set, NULL));
    strSetDestroy(&set);
}

static void
strSetContainsOfEmptySetReturnsFalse(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);

    assert_false(strSetContains(set, "Hey"));

    strSetDestroy(&set);
}

static void
strSetContainsOfExistingElementReturnsTrue(void** state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);

    assert_true(strSetAdd(set, "Hey"));

    // Lookup the element, it should be there
    assert_true(strSetContains(set, "Hey"));

    strSetDestroy(&set);
}

static void
strSetContainsOfNonExistingElementReturnsFalse(void **state)
{
    strset_t *set = strSetCreate(DEFAULT_SET_SIZE);
    assert_non_null(set);

    strSetAdd(set, "Hey");

    assert_false(strSetContains(set, "You"));
    assert_false(strSetContains(set, "hey"));
    assert_false(strSetContains(set, "h"));

    strSetDestroy(&set);
}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(strSetCreateReturnsNonNull),
        cmocka_unit_test(strSetDestroyOfNullSetDoesNotCrash),
        cmocka_unit_test(strSetAddOnNullSetReturnsFalse),
        cmocka_unit_test(strSetAddOfNullStrReturnsFalse),
        cmocka_unit_test(strSetAddNewElementReturnsTrue),
        cmocka_unit_test(strSetAddDupElementReturnsFalse),
        cmocka_unit_test(strSetAddIsNotCaseSensitive),
        cmocka_unit_test(strSetAddGrowsWithoutCrashing),
        cmocka_unit_test(strSetContainsOfNullReturnsFalse),
        cmocka_unit_test(strSetContainsOfEmptySetReturnsFalse),
        cmocka_unit_test(strSetContainsOfExistingElementReturnsTrue),
        cmocka_unit_test(strSetContainsOfNonExistingElementReturnsFalse),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
