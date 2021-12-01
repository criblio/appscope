#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "fn.h"
#include "mtc.h"
#include "test.h"

static void
mtcCreateReturnsValidPtr(void **state)
{
    mtc_t *mtc = mtcCreate();
    assert_non_null(mtc);
    mtcDestroy(&mtc);
    assert_null(mtc);
}

static void
mtcDestroyNullMtcDoesntCrash(void **state)
{
    mtcDestroy(NULL);
    mtc_t *mtc = NULL;
    mtcDestroy(&mtc);
    // Implicitly shows that calling mtcDestroy with NULL is harmless
}

static void
mtcEnabledSetAndGet(void **state)
{
    assert_int_equal(mtcEnabled(NULL), DEFAULT_MTC_ENABLE);

    mtc_t *mtc = mtcCreate();
    assert_int_equal(mtcEnabled(mtc), DEFAULT_MTC_ENABLE);

    mtcEnabledSet(mtc, FALSE);
    assert_false(mtcEnabled(mtc));
    mtcEnabledSet(mtc, TRUE);
    assert_true(mtcEnabled(mtc));

    mtcDestroy(&mtc);
}

static void
mtcSendForNullMtcDoesntCrash(void **state)
{
    const char *msg = "Hey, this is cool!\n";
    assert_int_equal(mtcSend(NULL, msg), -1);
}

static void
mtcSendForNullMessageDoesntCrash(void **state)
{
    mtc_t *mtc = mtcCreate();
    assert_non_null(mtc);
    transport_t *t = transportCreateSyslog();
    assert_non_null(t);
    mtcTransportSet(mtc, t);
    assert_int_equal(mtcSend(mtc, NULL), -1);
    mtcDestroy(&mtc);
}

static void
mtcTransportSetAndMtcSend(void **state)
{
    const char *file_path = "/tmp/my.path";
    mtc_t *mtc = mtcCreate();
    assert_non_null(mtc);
    transport_t *t1 = transportCreateUdp("127.0.0.1", "12345");
    transport_t *t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t *t3 = transportCreateSyslog();
    transport_t *t4 = transportCreateShm();
    transport_t *t5 = transportCreateFile(file_path, CFG_BUFFER_FULLY);
    mtcTransportSet(mtc, t1);
    mtcTransportSet(mtc, t2);
    mtcTransportSet(mtc, t3);
    mtcTransportSet(mtc, t4);
    mtcTransportSet(mtc, t5);

    // Test that transport is set by testing side effects of mtcSend
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    char *msg = strdup("Something to send\n");
    assert_int_equal(mtcSend(mtc, msg), 0);

    // With CFG_BUFFER_FULLY, this output only happens with the flush
    long file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    mtcFlush(mtc);
    file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    mtcDestroy(&mtc);
}

static void
mtcFormatSetAndMtcSendEvent(void **state)
{
    const char *file_path = "/tmp/my.path";
    mtc_t *mtc = mtcCreate();
    assert_non_null(mtc);
    transport_t *t = transportCreateFile(file_path, CFG_BUFFER_LINE);
    mtcTransportSet(mtc, t);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    mtc_fmt_t *f = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatSet(mtc, f);

    // Test that format is cleared by seeing no side effects.
    mtcFormatSet(mtc, NULL);
    long file_pos_before = fileEndPosition(file_path);
    assert_int_equal(mtcSendMetric(mtc, &e), -1);
    long file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    mtcDestroy(&mtc);
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);
    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(mtcCreateReturnsValidPtr),     cmocka_unit_test(mtcDestroyNullMtcDoesntCrash),     cmocka_unit_test(mtcEnabledSetAndGet),
        cmocka_unit_test(mtcSendForNullMtcDoesntCrash), cmocka_unit_test(mtcSendForNullMessageDoesntCrash), cmocka_unit_test(mtcTransportSetAndMtcSend),
        cmocka_unit_test(mtcFormatSetAndMtcSendEvent),  cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
