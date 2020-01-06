#define _GNU_SOURCE
#include <stdio.h>
#include "com.h"
#include "ctl.h"
#include "dbg.h"
#include "test.h"

static void
cmdPostEvtMsgDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);

    assert_int_equal(-1, cmdPostEvtMsg(NULL, NULL));
    assert_int_equal(-1, cmdPostEvtMsg(NULL, cJSON_CreateString("hey")));
    assert_int_equal(-1, cmdPostEvtMsg(ctl, NULL));
    assert_int_equal(0, cmdPostEvtMsg(ctl, cJSON_CreateString("hey")));

    ctlDestroy(&ctl);
}

static void
cmdPostInfoMsgDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);

    assert_int_equal(-1, cmdPostInfoMsg(NULL, NULL));
    assert_int_equal(-1, cmdPostInfoMsg(NULL, cJSON_CreateString("hey")));
    assert_int_equal(-1, cmdPostInfoMsg(ctl, NULL));
    assert_int_equal(0, cmdPostInfoMsg(ctl, cJSON_CreateString("hey")));

    ctlDestroy(&ctl);
}

static void
cmdSendEvtMsgDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);

    assert_int_equal(-1, cmdSendEvtMsg(NULL, NULL));
    assert_int_equal(-1, cmdSendEvtMsg(NULL, cJSON_CreateString("hey")));
    assert_int_equal(-1, cmdSendEvtMsg(ctl, NULL));
    assert_int_equal(0, cmdSendEvtMsg(ctl, cJSON_CreateString("hey")));

    ctlDestroy(&ctl);
}

static void
cmdSendInfoStrDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);

    assert_int_equal(-1, cmdSendInfoStr(NULL, NULL));
    assert_int_equal(-1, cmdSendInfoStr(NULL, "hey"));
    assert_int_equal(-1, cmdSendInfoStr(ctl, NULL));
    assert_int_equal(0, cmdSendInfoStr(ctl, "hey"));

    ctlDestroy(&ctl);
}

static void
cmdSendInfoMsgDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);

    assert_int_equal(-1, cmdSendInfoMsg(NULL, NULL));
    assert_int_equal(-1, cmdSendInfoMsg(NULL, cJSON_CreateString("hey")));
    assert_int_equal(-1, cmdSendInfoMsg(ctl, NULL));
    assert_int_equal(0, cmdSendInfoMsg(ctl, cJSON_CreateString("hey")));

    ctlDestroy(&ctl);
}

static void
cmdSendResponseDoesNotCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    const char buf[] =
         "{\"type\": \"req\", \"req\": \"huh?\", \"reqId\": 3.5}";

    request_t* req = cmdParse(buf);
    assert_non_null(req);

    assert_int_equal(-1, cmdSendResponse(NULL, NULL));
    assert_int_equal(-1, cmdSendResponse(NULL, req));
    assert_int_equal(-1, cmdSendResponse(ctl, NULL));
    assert_int_equal(0, cmdSendResponse(ctl, req));

    ctlDestroy(&ctl);
    destroyReq(&req);
}

static void
cmdParseDoesNotCrash(void** state)
{
    assert_int_equal(dbgCountAllLines(), 0);
    request_t* req = cmdParse(NULL);
    assert_int_equal(dbgCountAllLines(), 1);
    dbgInit(); // reset dbg for the rest of the tests
    assert_non_null(req);
    destroyReq(&req);

    req = cmdParse("{\"type\": \"req\", \"req\": \"huh?\", \"reqId\": 3.5}");
    assert_non_null(req);
    destroyReq(&req);
}

static void
msgStartHasExpectedSubNodes(void** state)
{
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "comtest",
                      .cmd = "cmd",
                      .id = "host-comtest-cmd"};
    config_t* cfg = cfgCreateDefault();
    cJSON* json;

    json = msgStart(NULL, NULL);
    assert_null(json);
    json = msgStart(NULL, cfg);
    assert_null(json);
    json = msgStart(&proc, NULL);
    assert_null(json);
    json = msgStart(&proc, cfg);
    assert_non_null(json);

    assert_non_null(cJSON_GetObjectItem(json, "process"));
    assert_non_null(cJSON_GetObjectItem(json, "configuration"));
    assert_non_null(cJSON_GetObjectItem(json, "environment"));

    /*
    char* str = cJSON_Print(json);
    assert_non_null(str);
    printf(str);
    free(str);
    */

    cJSON_Delete(json);
    cfgDestroy(&cfg);
}

static void
msgEvtMetricDoesNotCrash(void** state)
{
    evt_t* evt = evtCreate();
    assert_non_null(evt);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "comtest",
                      .cmd = "cmd",
                      .id = "host-comtest-cmd"};

    // when enabled, we should get a non-null json
    evtSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
    cJSON* json = msgEvtMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    evtDestroy(&evt);
}

static void
msgEvtLogDoesNotCrash(void** state)
{
    evt_t* evt = evtCreate();
    assert_non_null(evt);

    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "comtest",
                      .cmd = "cmd",
                      .id = "host-comtest-cmd"};

    // when enabled, we should get a non-null json
    evtSourceEnabledSet(evt, CFG_SRC_CONSOLE, 1);
    cJSON* json = msgEvtLog(evt, "stdout", "hey", 4, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    evtDestroy(&evt);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cmdPostEvtMsgDoesNotCrash),
        cmocka_unit_test(cmdPostInfoMsgDoesNotCrash),
        cmocka_unit_test(cmdSendEvtMsgDoesNotCrash),
        cmocka_unit_test(cmdSendInfoStrDoesNotCrash),
        cmocka_unit_test(cmdSendInfoMsgDoesNotCrash),
        cmocka_unit_test(cmdSendResponseDoesNotCrash),
        cmocka_unit_test(cmdParseDoesNotCrash),
        cmocka_unit_test(msgStartHasExpectedSubNodes),
        cmocka_unit_test(msgEvtMetricDoesNotCrash),
        cmocka_unit_test(msgEvtLogDoesNotCrash),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
