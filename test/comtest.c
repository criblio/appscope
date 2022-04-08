#define _GNU_SOURCE
#include <stdio.h>
#include "com.h"
#include "ctl.h"
#include "dbg.h"
#include "test.h"
#include "runtimecfg.h"

static void
cmdPostInfoMsgDoesNotCrash(void** state)
{
    config_t* cfg = cfgCreateDefault();
    ctl_t* ctl = ctlCreate(cfg);
    assert_non_null(ctl);

    assert_int_equal(-1, cmdPostInfoMsg(NULL, NULL));
    assert_int_equal(-1, cmdPostInfoMsg(NULL, cJSON_CreateString("hey")));
    assert_int_equal(-1, cmdPostInfoMsg(ctl, NULL));
    assert_int_equal(0, cmdPostInfoMsg(ctl, cJSON_CreateString("hey")));

    ctlDestroy(&ctl);
    cfgDestroy(&cfg);
}

static void
cmdSendInfoStrDoesNotCrash(void** state)
{
    config_t* cfg = cfgCreateDefault();
    ctl_t* ctl = ctlCreate(cfg);
    assert_non_null(ctl);

    assert_int_equal(-1, cmdSendInfoStr(NULL, NULL));
    assert_int_equal(-1, cmdSendInfoStr(NULL, "hey"));
    assert_int_equal(-1, cmdSendInfoStr(ctl, NULL));
    assert_int_equal(0, cmdSendInfoStr(ctl, "hey"));

    ctlDestroy(&ctl);
    cfgDestroy(&cfg);
}

static void
cmdSendResponseDoesNotCrash(void** state)
{
    config_t* cfg = cfgCreateDefault();
    ctl_t* ctl = ctlCreate(cfg);
    assert_non_null(ctl);
    const char buf[] =
         "{\"type\": \"req\", \"req\": \"huh?\", \"reqId\": 3.5}";

    request_t* req = cmdParse(buf);
    assert_non_null(req);

    // First two arguments must be non-null.  The last arg can be null.
    assert_int_equal(-1, cmdSendResponse(NULL, NULL, NULL));
    assert_int_equal(-1, cmdSendResponse(NULL, req, NULL));
    assert_int_equal(-1, cmdSendResponse(ctl, NULL, NULL));
    assert_int_equal(0, cmdSendResponse(ctl, req, NULL));

    ctlDestroy(&ctl);
    destroyReq(&req);
    cfgDestroy(&cfg);
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
                      .gid = 1000,
                      .groupname = "test-group",
                      .uid = 1000,
                      .username = "test-user",
                      .hostname = "host",
                      .procname = "comtest",
                      .cmd = "cmd",
                      .id = "host-comtest-cmd"};
    config_t *cfg = cfgCreateDefault();
    cJSON *json;

    json = msgStart(NULL, NULL, CFG_CTL);
    assert_null(json);
    json = msgStart(NULL, cfg, CFG_CTL);
    assert_null(json);
    json = msgStart(&proc, NULL, CFG_CTL);
    assert_null(json);
    json = msgStart(&proc, cfg, CFG_CTL);
    assert_non_null(json);

    // Make sure there is a "format" field, with value "ndjson"
    cJSON *format = cJSON_GetObjectItem(json, "format");
    assert_non_null(format);
    assert_string_equal("ndjson", cJSON_GetStringValue(format));

    // Make sure there is an "info" field
    cJSON *info = cJSON_GetObjectItem(json, "info");
    assert_non_null(info);

    // Make sure the contents of "info" contain the basics
    assert_non_null(cJSON_GetObjectItem(info, "process"));
    assert_non_null(cJSON_GetObjectItem(info, "configuration"));
    assert_non_null(cJSON_GetObjectItem(info, "environment"));

    /*
    char* str = cJSON_Print(json);
    assert_non_null(str);
    printf("%s\n", str);
    free(str);
    */

    cJSON_Delete(json);
    cfgDestroy(&cfg);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cmdPostInfoMsgDoesNotCrash),
        cmocka_unit_test(cmdSendInfoStrDoesNotCrash),
        cmocka_unit_test(cmdSendResponseDoesNotCrash),
        cmocka_unit_test(cmdParseDoesNotCrash),
        cmocka_unit_test(msgStartHasExpectedSubNodes),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
