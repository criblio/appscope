#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ctl.h"
#include "dbg.h"

#include "test.h"

static void
ctlParseRxMsgNullReturnsParseError(void** state)
{
    request_t* req = ctlParseRxMsg(NULL);
    assert_non_null(req);

    assert_int_equal(req->cmd, REQ_PARSE_ERR);
    assert_null(req->cmd_str);
    assert_int_equal(req->id, 0);
    assert_null(req->cfg);

    // NULL seems like it would only be the result of a programming error
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    destroyReq(&req);
    assert_null(req);
}

static void
ctlParseRxMsgUnparseableReturnsParseError(void** state)
{
    // We expect REQ_PARSE_ERR if any of the following are true:
    const char* test[] = {
    //  o) The json is truely unparseable
        "{ \"type\": \"req\", \"re",
        "dude, seriously?",
        "",
        "\n",
    //  o) The root object is anything but an object
        "\"hey\"",
        "1",
        "[ \"Clint\", \"Ledion\", \"Dritan\" ]",
        "true",
        "false",
        "null",
        NULL
    };

    const char** msg;
    for (msg=test; *msg; msg++) {
        //printf("%s\n", *msg);
        request_t* req = ctlParseRxMsg(*msg);
        assert_non_null(req);

        assert_int_equal(req->cmd, REQ_PARSE_ERR);
        assert_null(req->cmd_str);
        assert_int_equal(req->id, 0);
        assert_null(req->cfg);

        destroyReq(&req);
    }
}

static void
ctlParseRxMsgRequiredFieldProblemsReturnsMalformed(void** state)
{
    // We expect REQ_MALFORMED if any of the following are true:
    const char* test[] = {
    //  o)  value of type field is anything but "req"
        "{ \"type\": \"info\", \"req\": \"GetCfg\", \"reqId\": 1 }",
        "{ \"type\": \"resp\", \"req\": \"GetCfg\", \"reqId\": 1 }",
        "{ \"type\": \"evt\",  \"req\": \"GetCfg\", \"reqId\": 1 }",

    //  o)  any of these fields are missing: type, req, reqId
        "{ \"req\": \"GetCfg\", \"reqId\": 1 }",
        "{ \"type\": \"req\", \"reqId\": 1 }",
        "{ \"type\": \"req\", \"req\": \"GetCfg\" }",

    //  o)  any of these fields are wrong types:
    //      type (string), req (string), reqId (number)
        "{ \"type\": 1, \"req\": \"GetCfg\", \"reqId\": 1 }",
        "{ \"type\": \"req\", \"req\": 1, \"reqId\": 1 }",
        "{ \"type\": \"req\", \"req\": \"GetCfg\", \"reqId\": \"hey\" }",
        NULL
    };

    const char** msg;
    for (msg=test; *msg; msg++) {
        //printf("%s\n", *msg);
        request_t* req = ctlParseRxMsg(*msg);
        assert_non_null(req);
        assert_int_equal(req->cmd, REQ_MALFORMED);
        destroyReq(&req);
    }
}

static void
ctlParseRxMsgBogusReqReturnsUnknown(void** state)
{
    const char buf[] =
         "{"
         "    \"type\": \"req\","
         "    \"req\": \"huh?\","
         "    \"reqId\": 3.5"
         "}";

    request_t* req = ctlParseRxMsg(buf);
    assert_non_null(req);

    assert_int_equal(req->cmd, REQ_UNKNOWN);
    assert_string_equal(req->cmd_str, "huh?");
    assert_int_equal(req->id, 3.5);
    assert_null(req->cfg);

    destroyReq(&req);
}

static void
ctlParseRxMsgSetCfgWithoutDataObjectReturnsParamErr(void** state)
{
    char* msg;

    const char base[] =
         "{"
         "    \"type\": \"req\","
         "    \"req\": \"SetCfg\","
         "    \"reqId\": 987413948756391"
         "    %s"
         "}";

    const char* data[] = {
    //  o) data field absent
        "",
    //  o) data is not object
        ",\"data\": \"hey\"",
        ",\"data\": 1",
        ",\"data\": [ \"Clint\", \"Ledion\", \"Dritan\" ]",
        ",\"data\": true",
        ",\"data\": false",
        ",\"data\": null",
        NULL
    };

    const char** test;
    for (test=data; *test; test++) {
        assert_return_code(asprintf(&msg, base, *test), errno);
        //printf("%s\n", msg);
        request_t* req = ctlParseRxMsg(msg);
        free(msg);
        assert_non_null(req);

        // data is missing, so expect REQ_PARAM_ERR
        assert_int_equal(req->cmd, REQ_PARAM_ERR);
        assert_string_equal(req->cmd_str, "SetCfg");
        assert_int_equal(req->id, 987413948756391);
        assert_null(req->cfg);
        destroyReq(&req);
    }
}

static void
ctlParseRxMsgSetCfg(void** state)
{
    char* msg;

    const char base[] =
         "{"
         "    \"type\": \"req\","
         "    \"req\": \"SetCfg\","
         "    \"reqId\": 3,"
         "    \"data\": %s"
         "}";

    const char* data[] = {
    //  o) Any object will be accepted, this will return *all* defaults
        "{}",
    //  o) This should create a cfg object that is different than default
        "\n"
        "{\n"
        "  \"metric\": {\n"
        "    \"format\": {\n"
        "      \"type\": \"metricjson\",\n"
        "      \"statsdprefix\": \"cribl.scope\",\n"
        "      \"statsdmaxlen\": \"42\",\n"
        "      \"verbosity\": \"0\",\n"
        "      \"tags\": [\n"
        "        {\"tagA\": \"val1\"},\n"
        "        {\"tagB\": \"val2\"},\n"
        "        {\"tagC\": \"val3\"}\n"
        "      ]\n"
        "    },\n"
        "    \"transport\": {\n"
        "      \"type\": \"file\",\n"
        "      \"path\": \"/var/log/scope.log\"\n"
        "    }\n"
        "  },\n"
        "  \"event\": {\n"
        "    \"format\": {\n"
        "      \"type\": \"ndjson\"\n"
        "    },\n"
        "    \"watch\" : [\n"
        "      {\"type\":\"file\", \"name\":\".*[.]log$\"},\n"
        "      {\"type\":\"console\"},\n"
        "      {\"type\":\"syslog\"},\n"
        "      {\"type\":\"metric\"}\n"
        "    ]\n"
        "  },\n"
        "  \"libscope\": {\n"
        "    \"transport\": {\n"
        "      \"type\": \"file\",\n"
        "      \"path\": \"/var/log/event.log\"\n"
        "    },\n"
        "    \"summaryperiod\": \"13\",\n"
        "    \"log\": {\n"
        "      \"level\": \"debug\",\n"
        "      \"transport\": {\n"
        "        \"type\": \"shm\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n",
        NULL
    };

    const char** test;
    int run=1;
    for (test=data; *test; test++) {
        assert_return_code(asprintf(&msg, base, *test), errno);
        //printf("%s\n", msg);
        request_t* req = ctlParseRxMsg(msg);
        free(msg);
        assert_non_null(req);

        // data exists! expect REQ_SET_CFG, and non-null req->cfg
        assert_int_equal(req->cmd, REQ_SET_CFG);
        assert_string_equal(req->cmd_str, "SetCfg");
        assert_int_equal(req->id, 3);
        assert_non_null(req->cfg);

        // Verify StatsDMaxLen just to do some crude verification that
        // the data made it into the req->cfg object...
        // Run 1: verify that StatsDMaxLen is default
        // Run 2: verify that StatsDMaxLen is 42
        int expected_statsd_val;
        switch (run++) {
            case 1:
                expected_statsd_val = DEFAULT_STATSD_MAX_LEN;
                break;
            case 2:
                expected_statsd_val = 42;
                break;
            default:
                fail(); // must have added a test to data[] above.
        }
        assert_int_equal(cfgOutStatsDMaxLen(req->cfg), expected_statsd_val);
        destroyReq(&req);
    }
}

static void
ctlParseRxMsgGetCfg(void** state)
{
    const char buf[] =
         "{"
         "    \"type\": \"req\","
         "    \"req\": \"GetCfg\","
         "    \"reqId\": 987413948756391"
         "}";

    request_t* req = ctlParseRxMsg(buf);
    assert_non_null(req);

    assert_int_equal(req->cmd, REQ_GET_CFG);
    assert_string_equal(req->cmd_str, "GetCfg");
    assert_int_equal(req->id, 987413948756391);
    assert_null(req->cfg);

    destroyReq(&req);
}

static void
ctlParseRxMsgGetDiags(void** state)
{
    const char buf[] =
         "{"
         "    \"type\": \"req\","
         "    \"req\": \"GetDiag\","
         "    \"reqId\": 1983457849314789"
         "}";

    request_t* req = ctlParseRxMsg(buf);
    assert_non_null(req);

    assert_int_equal(req->cmd, REQ_GET_DIAG);
    assert_string_equal(req->cmd_str, "GetDiag");
    assert_int_equal(req->id, 1983457849314789);
    assert_null(req->cfg);

    destroyReq(&req);
}


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
        cmocka_unit_test(ctlParseRxMsgNullReturnsParseError),
        cmocka_unit_test(ctlParseRxMsgUnparseableReturnsParseError),
        cmocka_unit_test(ctlParseRxMsgRequiredFieldProblemsReturnsMalformed),
        cmocka_unit_test(ctlParseRxMsgBogusReqReturnsUnknown),
        cmocka_unit_test(ctlParseRxMsgSetCfgWithoutDataObjectReturnsParamErr),
        cmocka_unit_test(ctlParseRxMsgSetCfg),
        cmocka_unit_test(ctlParseRxMsgGetCfg),
        cmocka_unit_test(ctlParseRxMsgGetDiags),
        cmocka_unit_test(ctlSendMsgForNullOutDoesntCrash),
        cmocka_unit_test(ctlSendMsgForNullMessageDoesntCrash),
        cmocka_unit_test(ctlTransportSetAndOutSend),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };

    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}

