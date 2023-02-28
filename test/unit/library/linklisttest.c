#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dbg.h"
#include "linklist.h"
#include "test.h"

static void
lstCreateReturnsNonNull(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);
    lstDestroy(&list);

    // Test that lstDestroy changes the value of list to null
    assert_null(list);
}

static void
lstDestroyOfNullListDoesNotCrash(void** state)
{
    lstDestroy(NULL);

    list_t* list = NULL;
    lstDestroy(&list);
}

static void
lstInsertOnNullListReturnsFalse(void **state)
{
    assert_false(lstInsert(NULL, 23, (void*)23));
}

static void
lstInsertNewElementReturnsTrue(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    assert_true(lstInsert(list, 23, (void*)23));

    lstDestroy(&list);
}

static void
lstInsertDupElementReturnsFalse(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    assert_true(lstInsert(list, 23, (void*)23));

    // Try to add 23 again.  It should return null.
    assert_false(lstInsert(list, 23, (void*)23));

    lstDestroy(&list);
}

static void
lstFindOfNullListReturnsNull(void **state)
{
    list_t* result_of_find = lstFind(NULL, 23);
    assert_null(result_of_find);
}

static void
lstFindOfEmptyListReturnsNull(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    list_t* result_of_find = lstFind(list, 23);
    assert_null(result_of_find);

    lstDestroy(&list);
}

static void
lstFindOfExistingElementReturnsTheElement(void** state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    assert_true(lstInsert(list, 23, (void*)23));

    // Lookup the element with find, it should be there
    void* result_of_find = lstFind(list, 23);

    // Verify that they're the same
    assert_ptr_equal(result_of_find, (void*)23);

    lstDestroy(&list);
}

static void
lstFindOfNonExistingElementReturnsNull(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    lstInsert(list, 23, (void*)23);

    assert_null(lstFind(list, 22));
    assert_null(lstFind(list, 24));
    assert_null(lstFind(list, 0));

    lstDestroy(&list);
}

static void
lstDeleteOnNullListReturnsFalse (void **state)
{
    assert_int_equal(lstDelete(NULL, 23), 0);

    list_t* list = NULL;
    assert_int_equal(lstDelete(list, 23), 0);
}

static void
lstDeleteExistingElementReturnsTrue(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    // Add some elements in arbitrary order
    lstInsert(list, 23, (void*)23);
    lstInsert(list, 22, (void*)22);
    lstInsert(list, 24, (void*)24);

    // Verify that they're there
    assert_non_null(lstFind(list, 22));
    assert_non_null(lstFind(list, 23));
    assert_non_null(lstFind(list, 24));

    // Delete the middle
    assert_int_equal(lstDelete(list, 23), 1);
    assert_null(lstFind(list, 23));

    // Delete the first
    assert_int_equal(lstDelete(list, 22), 1);
    assert_null(lstFind(list, 22));

    // Delete the last
    assert_int_equal(lstDelete(list, 24), 1);
    assert_null(lstFind(list, 24));

    lstDestroy(&list);
}

static void
lstDeleteNonExistingElementReturnsFalse(void **state)
{
    list_t* list = lstCreate(NULL);
    assert_non_null(list);

    assert_int_equal(lstDelete(list, 23), 0);

    lstDestroy(&list);
}

static void
test_delete_fn(void* arg)
{
    // Completely contrived example.  Assume arg is a pointer to an uint64_t.
    // Increment the value at that location just so something can be
    // observed in the following test.
    uint64_t* int_ptr = arg;
    *int_ptr = 24;
}

static void
lstDeleteCallsDeleteFn(void **state)
{
    uint64_t value = 23;

    list_t* list = lstCreate(test_delete_fn);
    assert_non_null(list);

    // Insert a pointer to value
    assert_true( lstInsert(list, 23, &value) );

    // Retrieve the pointer, verify nothing has modified value
    void* data = lstFind(list, 23);
    assert_int_equal(data, &value);
    assert_int_equal(*(uint64_t*)data, 23);

    // Delete it - this should call test_delete_fn()
    assert_true (lstDelete(list, 23));

    // See that test_delete_fn() modified the value of data.
    // The fact that value changed from 23 to 24 proves it was called.
    assert_int_equal(*(uint64_t*)data, 24);

    lstDestroy(&list);
}

static void
lstDeleteSimpleDeleteFnExample(void **state)
{
    // This is less of a test than an example;
    // it's here to show the simplest way delete_fn could be used.
    // Though, if you run this test under valgrind, valgrind
    // will show that the memory malloc'd by strdup is freed properly.

    // Pass stdlib.h free() as the delete_fn.
    list_t* list = lstCreate(free);
    assert_non_null(list);

    // strdup does dynamic memory allocation that needs to be free'd.
    assert_true( lstInsert(list, 23, strdup("Hey!")) );

    // free(data) is called during lstDelete()
    assert_true( lstDelete(list, 23) );

    lstDestroy(&list);


    // And the same thing again, but with lstDestroy() doing the
    // lstDelete of all the elements in the list.
    list = lstCreate(free);
    assert_true( lstInsert(list, 23, strdup("You!")) );
    lstDestroy(&list);
}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(lstCreateReturnsNonNull),
        cmocka_unit_test(lstDestroyOfNullListDoesNotCrash),
        cmocka_unit_test(lstInsertOnNullListReturnsFalse),
        cmocka_unit_test(lstInsertNewElementReturnsTrue),
        cmocka_unit_test(lstInsertDupElementReturnsFalse),
        cmocka_unit_test(lstFindOfNullListReturnsNull),
        cmocka_unit_test(lstFindOfEmptyListReturnsNull),
        cmocka_unit_test(lstFindOfExistingElementReturnsTheElement),
        cmocka_unit_test(lstFindOfNonExistingElementReturnsNull),
        cmocka_unit_test(lstDeleteOnNullListReturnsFalse),
        cmocka_unit_test(lstDeleteExistingElementReturnsTrue),
        cmocka_unit_test(lstDeleteNonExistingElementReturnsFalse),
        cmocka_unit_test(lstDeleteCallsDeleteFn),
        cmocka_unit_test(lstDeleteSimpleDeleteFnExample),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
