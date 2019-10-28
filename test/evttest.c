#include <stdio.h>
#include <unistd.h>
#include "dbg.h"
#include "evt.h"

#include "test.h"

static void
evtCreateReturnsValidPtr(void** state)
{
    evt_t* evt = evtCreate();
    assert_non_null(evt);
    evtDestroy(&evt);
    assert_null(evt);
}

static void
evtDestroyNullOutDoesntCrash(void** state)
{
    evtDestroy(NULL);
    evt_t* evt = NULL;
    evtDestroy(&evt);
    // Implicitly shows that calling evtDestroy with NULL is harmless
}

static void
evtSendForNullOutDoesntCrash(void** state)
{
    const char* msg = "Hey, this is cool!\n";
    assert_int_equal(evtSend(NULL, msg), -1);
}

static void
evtSendForNullMessageDoesntCrash(void** state)
{
    evt_t* evt = evtCreate();
    assert_non_null(evt);
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    evtTransportSet(evt, t);
    assert_int_equal(evtSend(evt, NULL), -1);
    evtDestroy(&evt);
}

static void
evtTransportSetAndOutSend(void** state)
{
    const char* file_path = "/tmp/my.path";
    evt_t* evt = evtCreate();
    assert_non_null(evt);
    transport_t* t1 = transportCreateUdp("127.0.0.1", "12345");
    transport_t* t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t* t3 = transportCreateSyslog();
    transport_t* t4 = transportCreateShm();
    transport_t* t5 = transportCreateFile(file_path, CFG_BUFFER_FULLY);
    evtTransportSet(evt, t1);
    evtTransportSet(evt, t2);
    evtTransportSet(evt, t3);
    evtTransportSet(evt, t4);
    evtTransportSet(evt, t5);

    // Test that transport is set by testing side effects of evtSend
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    assert_int_equal(evtSend(evt, "Something to send\n"), 0);

    // With CFG_BUFFER_FULLY, this output only happens with the flush
    long file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    evtFlush(evt);
    file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that transport is cleared by seeing no side effects.
    evtTransportSet(evt, NULL);
    file_pos_before = fileEndPosition(file_path);
    assert_int_equal(evtSend(evt, "Something to send\n"), -1);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    evtDestroy(&evt);
}

static void
evtFormatSetAndOutSendEvent(void** state)
{
    const char* file_path = "/tmp/my.path";
    evt_t* evt = evtCreate();
    assert_non_null(evt);
    transport_t* t = transportCreateFile(file_path, CFG_BUFFER_LINE);
    evtTransportSet(evt, t);

    event_t e = {"A", 1, DELTA, NULL};
    format_t* f = fmtCreate(CFG_EXPANDED_STATSD);
    evtFormatSet(evt, f);

    // Test that format is set by testing side effects of evtSendEvent
    // affecting the file at file_path when connected to format.
    long file_pos_before = fileEndPosition(file_path);
    assert_int_equal(evtSendEvent(evt, &e), 0);
    long file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that format is cleared by seeing no side effects.
    evtFormatSet(evt, NULL);
    file_pos_before = fileEndPosition(file_path);
    assert_int_equal(evtSendEvent(evt, &e), -1);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    evtDestroy(&evt);
}

static void
evtLogFileFilterSetAndGet(void** state)
{
    evt_t* evt = evtCreate();

    // Make sure there is a filter by default
    regex_t* default_re = evtLogFileFilter(evt);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "somethinglogsomething", 0, NULL, 0), 0);
    assert_int_equal(regexec(default_re, "somethingsomething", 0, NULL, 0), REG_NOMATCH);

    // Make sure it can be changed
    evtLogFileFilterSet(evt, "(^/var/log/.*)|(.*[.]log$)");
    regex_t* new_re = evtLogFileFilter(evt);
    assert_non_null(new_re);
    assert_ptr_not_equal(default_re, new_re);
    assert_int_equal(regexec(new_re, "somethinglogsomething", 0, NULL, 0), REG_NOMATCH);
    assert_int_equal(regexec(new_re, "somethingsomething.log", 0, NULL, 0), 0);

    // Make sure default is returned for null strings
    evtLogFileFilterSet(evt, "");
    new_re = evtLogFileFilter(evt);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "somethinglogsomething", 0, NULL, 0), 0);
    assert_int_equal(regexec(new_re, "somethingsomething", 0, NULL, 0), REG_NOMATCH);

    evtLogFileFilterSet(evt, NULL);
    new_re = evtLogFileFilter(evt);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "somethinglogsomething", 0, NULL, 0), 0);
    assert_int_equal(regexec(new_re, "somethingsomething", 0, NULL, 0), REG_NOMATCH);

    // Make sure default is returned for bad regex
    evtLogFileFilterSet(evt, "W![T^F?");
    new_re = evtLogFileFilter(evt);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "somethinglogsomething", 0, NULL, 0), 0);
    assert_int_equal(regexec(new_re, "somethingsomething", 0, NULL, 0), REG_NOMATCH);

    evtDestroy(&evt);

    // Get a default filter, even if evt is NULL
    default_re = evtLogFileFilter(evt);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "somethinglogsomething", 0, NULL, 0), 0);
    assert_int_equal(regexec(default_re, "somethingsomething", 0, NULL, 0), REG_NOMATCH);
}

static void
evtSourceSetAndGet(void** state)
{
    evt_t* evt = evtCreate();

    // Set everything to 1
    int i, j;
    for (i=CFG_SRC_LOGFILE; i<CFG_SRC_MAX+1; i++) {
        evtSourceSet(evt, i, 1);
        if (i >= CFG_SRC_MAX) {
             assert_int_equal(evtSource(evt, i), DEFAULT_SRC_LOGFILE);
             assert_int_equal(dbgCountMatchingLines("src/evt.c"), 1);
             dbgInit(); // reset dbg for the rest of the tests
        } else {
             assert_int_equal(dbgCountMatchingLines("src/evt.c"), 0);
             assert_int_equal(evtSource(evt, i), 1);
        }
    }

    // Clear one at a time to see there aren't side effects
    for (i=CFG_SRC_LOGFILE; i<CFG_SRC_MAX; i++) {
        evtSourceSet(evt, i, 0); // Clear it
        for (j=CFG_SRC_LOGFILE; j<CFG_SRC_MAX; j++) {
            if (i==j)
                 assert_int_equal(evtSource(evt, j), 0);
            else
                 assert_int_equal(evtSource(evt, j), 1);
        }
        evtSourceSet(evt, i, 1); // Set it back
    }

    evtDestroy(&evt);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(evtCreateReturnsValidPtr),
        cmocka_unit_test(evtDestroyNullOutDoesntCrash),
        cmocka_unit_test(evtSendForNullOutDoesntCrash),
        cmocka_unit_test(evtSendForNullMessageDoesntCrash),
        cmocka_unit_test(evtTransportSetAndOutSend),
        cmocka_unit_test(evtFormatSetAndOutSendEvent),
        cmocka_unit_test(evtLogFileFilterSetAndGet),
        cmocka_unit_test(evtSourceSetAndGet),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
