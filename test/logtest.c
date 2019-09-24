#include <stdio.h>
#include <unistd.h>
#include "log.h"

#include "test.h"

static void
logCreateReturnsValidPtr(void** state)
{
    log_t* log = logCreate();
    assert_non_null(log);
    logDestroy(&log);
    assert_null(log);
}

static void
logDestroyNullLogDoesntCrash(void** state)
{
    logDestroy(NULL);
    log_t* log = NULL;
    logDestroy(&log);
    // Implicitly shows that calling logDestroy with NULL is harmless
}

static void
logSendForNullLogDoesntCrash(void** state)
{
    const char* msg = "Hey, this is cool!\n";
    assert_int_equal(logSend(NULL, msg, DEFAULT_LOG_LEVEL), -1);
}

static void
logSendForNullMessageDoesntCrash(void** state)
{
    log_t* log = logCreate();
    assert_non_null(log);
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    logTransportSet(log, t);
    assert_int_equal(logSend(log, NULL, DEFAULT_LOG_LEVEL), -1);
    logDestroy(&log);
}

static void
logLevelVerifyDefaultLevel(void** state)
{
    log_t* log = logCreate();
    assert_int_equal(logLevel(log), DEFAULT_LOG_LEVEL);
    logDestroy(&log);
}

static void
logLevelSetAndGet(void** state)
{
    log_t* log = logCreate();
    logLevelSet(log, CFG_LOG_NONE);
    assert_int_equal(logLevel(log), CFG_LOG_NONE);
    logLevelSet(log, CFG_LOG_DEBUG);
    assert_int_equal(logLevel(log), CFG_LOG_DEBUG);
    logDestroy(&log);
}

static void
logTranportSetAndLogSend(void** state)
{
    const char* file_path = "/tmp/my.path";
    log_t* log = logCreate();
    assert_non_null(log);
    transport_t* t1 = transportCreateUdp("127.0.0.1", "12345");
    transport_t* t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t* t3 = transportCreateSyslog();
    transport_t* t4 = transportCreateShm();
    transport_t* t5 = transportCreateFile(file_path);
    logTransportSet(log, t1);
    logTransportSet(log, t2);
    logTransportSet(log, t3);
    logTransportSet(log, t4);
    logTransportSet(log, t5);

    // Test that transport is set by testing side effects of logSend
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    assert_int_equal(logSend(log, "Something to send\n", DEFAULT_LOG_LEVEL), 0);
    long file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that transport is cleared by seeing no side effects.
    logTransportSet(log, NULL);
    file_pos_before = fileEndPosition(file_path);
    assert_int_equal(logSend(log, "Something to send\n", DEFAULT_LOG_LEVEL), -1);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    logDestroy(&log);
}

static void
logSendWithLogLevelFilter(void** state)
{
    const char* file_path = "/tmp/my.path";
    log_t* log = logCreate();
    assert_non_null(log);
    transport_t* t = transportCreateFile(file_path);
    logTransportSet(log, t);

    // Test logLevel filtering by testing side effects of logSend
    // affecting the file at file_path.

    // Set the log level to CFG_LOG_TRACE and verify that every logSend
    // level results in something being added to the file
    logLevelSet(log, CFG_LOG_TRACE);
    cfg_log_level_t level;
    for (level = CFG_LOG_TRACE; level <= CFG_LOG_NONE + 1; level++) {
        long file_pos_before = fileEndPosition(file_path);
        assert_int_equal(logSend(log, "Something to send\n", level), 0);
        long file_pos_after = fileEndPosition(file_path);
        assert_int_not_equal(file_pos_before, file_pos_after);
    }

    // Set the log level to CFG_LOG_NONE and verify that no logSend
    // level results in something being added to the file.
    logLevelSet(log, CFG_LOG_NONE);
    for (level = CFG_LOG_TRACE; level <= CFG_LOG_NONE + 1; level++) {
        long file_pos_before = fileEndPosition(file_path);
        assert_int_equal(logSend(log, "Something to send\n", level), 0);
        long file_pos_after = fileEndPosition(file_path);
        assert_int_equal(file_pos_before, file_pos_after);
    }

    // Set the log level to CFG_LOG_INFO and verify that logSend
    // of INFO, WARN, or ERROR results in something being added to the file.
    logLevelSet(log, CFG_LOG_INFO);
    for (level = CFG_LOG_TRACE; level <= CFG_LOG_NONE + 1; level++) {
        long file_pos_before = fileEndPosition(file_path);
        assert_int_equal(logSend(log, "Something to send\n", level), 0);
        long file_pos_after = fileEndPosition(file_path);
        switch (level) {
            case CFG_LOG_TRACE:
            case CFG_LOG_DEBUG:
                assert_int_equal(file_pos_before, file_pos_after);
                break;
            case CFG_LOG_INFO:
            case CFG_LOG_WARN:
            case CFG_LOG_ERROR:
            case CFG_LOG_NONE:
                assert_int_not_equal(file_pos_before, file_pos_after);
        }
    }

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    logDestroy(&log);
}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(logCreateReturnsValidPtr),
        cmocka_unit_test(logDestroyNullLogDoesntCrash),
        cmocka_unit_test(logSendForNullLogDoesntCrash),
        cmocka_unit_test(logSendForNullMessageDoesntCrash),
        cmocka_unit_test(logLevelVerifyDefaultLevel),
        cmocka_unit_test(logLevelSetAndGet),
        cmocka_unit_test(logTranportSetAndLogSend),
        cmocka_unit_test(logSendWithLogLevelFilter),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
