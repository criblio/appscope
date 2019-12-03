#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "ctl.h"

#include "test.h"

static void
ctlSendMsgForNullOutDoesntCrash(void** state)
{
    char* msg = strdup("Hey, this is cool!\n");
    ctlSendMsg(NULL, msg);
}

static void
ctlSendMsgForNullMessageDoesntCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    ctlTransportSet(ctl, t);
    ctlSendMsg(ctl, NULL);
    ctlDestroy(&ctl);
}

static void
ctlTransportSetAndOutSend(void** state)
{
    const char* file_path = "/tmp/my.path";
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    transport_t* t1 = transportCreateUdp("127.0.0.1", "12345");
    transport_t* t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t* t3 = transportCreateSyslog();
    transport_t* t4 = transportCreateShm();
    transport_t* t5 = transportCreateFile(file_path, CFG_BUFFER_FULLY);
    ctlTransportSet(ctl, t1);
    ctlTransportSet(ctl, t2);
    ctlTransportSet(ctl, t3);
    ctlTransportSet(ctl, t4);
    ctlTransportSet(ctl, t5);

    // Test that transport is set by testing side effects of ctlSendMsg
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    char* msg = strdup("Something to send\n");
    ctlSendMsg(ctl, msg);

    // With CFG_BUFFER_FULLY, this output only happens with the flush
    long file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    ctlFlush(ctl);
    file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that transport is cleared by seeing no side effects.
    ctlTransportSet(ctl, NULL);
    file_pos_before = fileEndPosition(file_path);
    msg = strdup("Something else to send\n");
    ctlSendMsg(ctl, msg);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    ctlDestroy(&ctl);
}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ctlSendMsgForNullOutDoesntCrash),
        cmocka_unit_test(ctlSendMsgForNullMessageDoesntCrash),
        cmocka_unit_test(ctlTransportSetAndOutSend),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };

    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}

