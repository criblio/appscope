#define _GNU_SOURCE
#include "circbuf.h"
#include "dbg.h"
#include "test.h"
#include <stdio.h>

static void
circbufInitGetsBuf(void **state)
{
    cbuf_handle_t ch = cbufInit(10);
    assert_non_null(ch);
    assert_non_null(ch->buffer);
    cbufFree(ch);
}

static void
circbufResetTest(void **state)
{
    cbuf_handle_t ch = cbufInit(10);
    assert_non_null(ch);
    assert_non_null(ch->buffer);
    cbufReset(ch);
    assert_int_equal(ch->head, 0);
    assert_int_equal(ch->tail, 0);
    cbufFree(ch);
}

static void
circbufCapacityTest(void **state)
{
    cbuf_handle_t ch = cbufInit(10);
    assert_non_null(ch);
    assert_int_equal(cbufCapacity(ch), 10);
    cbufFree(ch);
}

static void
circbufPutGetTest(void **state)
{
    uint64_t data;
    cbuf_handle_t ch = cbufInit(5);
    assert_non_null(ch);
    assert_non_null(ch->buffer);

    data = 1;
    assert_int_equal(cbufPut(ch, data), 0);
    data = 2;
    assert_int_equal(cbufPut(ch, data), 0);
    data = 3;
    assert_int_equal(cbufPut(ch, data), 0);
    data = 4;
    assert_int_equal(cbufPut(ch, data), 0);
    data = 5;
    assert_int_equal(cbufPut(ch, data), 0);

    // should not accept a new entry
    assert_int_equal(dbgCountMatchingLines("src/circbuf.c"), 0);
    data = 6;
    assert_int_equal(cbufPut(ch, data), -1);
    // Note we removed the DBG statement as it caused a crash with 100k Go routines
    assert_int_equal(dbgCountMatchingLines("src/circbuf.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    // Did we get the correct data?
    assert_int_equal(cbufGet(ch, &data), 0);
    assert_int_equal(data, 1);
    assert_int_equal(cbufGet(ch, &data), 0);
    assert_int_equal(data, 2);
    assert_int_equal(cbufGet(ch, &data), 0);
    assert_int_equal(data, 3);
    assert_int_equal(cbufGet(ch, &data), 0);
    assert_int_equal(data, 4);
    assert_int_equal(cbufGet(ch, &data), 0);
    assert_int_equal(data, 5);
    // should not find a new entry
    assert_int_equal(cbufGet(ch, &data), -1);

    cbufFree(ch);
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(circbufInitGetsBuf), cmocka_unit_test(circbufResetTest),           cmocka_unit_test(circbufCapacityTest),
        cmocka_unit_test(circbufPutGetTest),  cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
