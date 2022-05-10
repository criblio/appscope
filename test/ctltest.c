#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ctl.h"
#include "circbuf.h"
#include "dbg.h"
#include "cfgutils.h"
#include "state.h"
#include "fn.h"
#include "scopestdlib.h"
#include "test.h"

#define BUFSIZE 500
static char cbuf_data[BUFSIZE];
static bool enable_cbuf_data;

static
void allow_copy_buf_data(bool enable)
{
  enable_cbuf_data = enable;
}

static
char* get_cbuf_data(void)
{
  return cbuf_data;
}

static
void set_cbuf_data(char* val, unsigned long long new_size)
{
  if (new_size > BUFSIZE) fail();
  memcpy(cbuf_data, val, new_size);
}

// These signatures satisfy --wrap=cbufGet in the Makefile
#ifdef __linux__
int __real_cbufGet(cbuf_handle_t, uint64_t*);
int __wrap_cbufGet(cbuf_handle_t cbuf, uint64_t *data)
#endif // __linux__
#ifdef __APPLE__
int cbufGet(cbuf_handle_t cbuf, uint64_t *data)
#endif // __APPLE__
{
    int res = __real_cbufGet(cbuf, data);
    if (enable_cbuf_data && res == 0) {
      log_event_t *event = (log_event_t*) *data;
      set_cbuf_data(event->data, event->datalen);
    }

    return res;
}

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
    //  o) The json is truly unparsable
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

    const char* body[] = {
    //  o) body field absent
        "",
    //  o) body is not object
        ",\"body\": \"hey\"",
        ",\"body\": 1",
        ",\"body\": [ \"Clint\", \"Ledion\", \"Dritan\" ]",
        ",\"body\": true",
        ",\"body\": false",
        ",\"body\": null",
        NULL
    };

    const char** test;
    for (test=body; *test; test++) {
        assert_return_code(scope_asprintf(&msg, base, *test), scope_errno);
        //printf("%s\n", msg);
        request_t* req = ctlParseRxMsg(msg);
        scope_free(msg);
        assert_non_null(req);

        // body is missing, so expect REQ_PARAM_ERR
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
         "    \"body\": %s"
         "}";

    const char* body[] = {
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
    for (test=body; *test; test++) {
        assert_return_code(scope_asprintf(&msg, base, *test), errno);
        //printf("%s\n", msg);
        request_t* req = ctlParseRxMsg(msg);
        scope_free(msg);
        assert_non_null(req);

        // body exists! expect REQ_SET_CFG, and non-null req->cfg
        assert_int_equal(req->cmd, REQ_SET_CFG);
        assert_string_equal(req->cmd_str, "SetCfg");
        assert_int_equal(req->id, 3);
        assert_non_null(req->cfg);

        // Verify StatsDMaxLen just to do some crude verification that
        // the body made it into the req->cfg object...
        // Run 1: verify that StatsDMaxLen is default
        // Run 2: verify that StatsDMaxLen is 42
        int expected_statsd_val = -1;
        switch (run++) {
            case 1:
                expected_statsd_val = DEFAULT_STATSD_MAX_LEN;
                break;
            case 2:
                expected_statsd_val = 42;
                break;
            default:
                fail(); // must have added a test to body[] above.
        }
        assert_int_equal(cfgMtcStatsDMaxLen(req->cfg), expected_statsd_val);
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


typedef struct {
    const char*    req;
    unsigned short port;
} test_port_t;

static void
ctlParseRxMsgBlockPort(void** state)
{
    test_port_t test[] = {
        // if body isn't present, clear the port
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 0}",
          .port = 0},
        // body with one json number in range should set the port
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 1,\"body\":0}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 2,\"body\":54321}",
          .port = 54321},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 3,\"body\":65535}",
          .port = 65535},
        // body of other json types should clear the port
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 4,\"body\":{\"huh\":\"what\"}}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 5,\"body\":[ 1, 2, 3 ]}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 6,\"body\":\"hey\"}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 7,\"body\":true}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 8,\"body\":false}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\": 9,\"body\":null}",
          .port = 0},
        // body with out of range json numbers should clear the port
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\":10,\"body\":65536}",
          .port = 0},
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\":11,\"body\":-1}",
          .port = 0},
        // fractional numbers are weird, but will work... (set the port)
        { .req = "{ \"type\": \"req\", \"req\": \"BlockPort\",\"reqId\":12,\"body\":1.667}",
          .port = 1},
    };

    int i;
    for (i=0; i<sizeof(test)/sizeof(test[0]); i++) {
        test_port_t* test_case = &test[i];

        request_t* req = ctlParseRxMsg(test_case->req);
        assert_non_null(req);

        //printf("%s\n", test_case->req);

        assert_int_equal(req->cmd, REQ_BLOCK_PORT);
        assert_string_equal(req->cmd_str, "BlockPort");
        assert_int_equal(req->id, i);
        assert_null(req->cfg);
        assert_int_equal(req->port, test_case->port);

        destroyReq(&req);
    }
}

typedef struct {
    const char*    req;
    cmd_t cmd;
    switch_action_t action;
} test_switch_t;

static void
ctlParseRxMsgSwitch(void** state)
{
    test_switch_t test[] = {
        // if body isn't present, there's nothing to do
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 0}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        // body with these specific strings should work
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 1,\"body\":\"redirect-on\"}",
          .cmd = REQ_SWITCH, .action = URL_REDIRECT_ON},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 2,\"body\":\"redirect-off\"}",
          .cmd = REQ_SWITCH, .action = URL_REDIRECT_OFF},
        // body of other json types should do nothing
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 3,\"body\":0}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 4,\"body\":{\"huh\":\"what\"}}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 5,\"body\":[ 1, 2, 3 ]}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 6,\"body\":true}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 7,\"body\":false}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 8,\"body\":null}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        // body with strings of the wrong case should do nothing
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\": 9,\"body\":\"Redirect-On\"}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
        { .req = "{ \"type\": \"req\", \"req\": \"Switch\",\"reqId\":10,\"body\":\"REDIRECT-ON\"}",
          .cmd = REQ_PARAM_ERR, .action = NO_ACTION},
    };

    int i;
    for (i=0; i<sizeof(test)/sizeof(test[0]); i++) {
        test_switch_t* test_case = &test[i];

        request_t* req = ctlParseRxMsg(test_case->req);
        assert_non_null(req);

        //printf("%s\n", test_case->req);

        assert_int_equal(req->cmd, test_case->cmd);
        assert_string_equal(req->cmd_str, "Switch");
        assert_int_equal(req->id, i);
        assert_null(req->cfg);
        assert_int_equal(req->port, 0);
        assert_int_equal(req->action, test_case->action);

        destroyReq(&req);
    }
}
static void
ctlCreateTxMsgReturnsNullForNullUpload(void** state)
{
    assert_null(ctlCreateTxMsg(NULL));
}

static void
ctlCreateTxMsgInfo(void** state)
{
    upload_t upload = {0};
    upload.type = UPLD_INFO;

    // If body is null, msg should be null
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 0);
    char* msg = ctlCreateTxMsg(&upload);
    assert_null(msg);
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    // If body is non-null, msg should exist
    upload.body = cJSON_Parse("\"yeah, dude\"");
    msg = ctlCreateTxMsg(&upload);
    assert_non_null(msg);

    char expected_msg[] =
        "{\"type\":\"info\",\"body\":\"yeah, dude\"}";
    assert_string_equal(msg, expected_msg);

    scope_free(msg);
}

typedef struct {
    char*      in;
    struct {
        char* req;
        int reqId;
        int status;
        char* message;
    } out;
} test_t;

static void
ctlCreateTxMsgResp(void** state)
{
    upload_t upload = {0};
    upload.type = UPLD_RESP;

    // If req is null, tx_msg should be null
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 0);
    char* tx_msg = ctlCreateTxMsg(&upload);
    assert_null(tx_msg);
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    test_t test[] = {
        // REQ_PARSE_ERR
        { .in = "{ \"type\": \"req\", \"re",
          .out = {.req=NULL,      .reqId=0, .status=400,
                  .message="Request could not be parsed as a json object"}},
        // REQ_MALFORMED
        { .in = "{ \"type\": \"info\", \"req\": \"GetCfg\", \"reqId\": 1 }",
          .out = {.req="GetCfg",  .reqId=1, .status=400,
                  .message="Type was not request, required fields were missing or of wrong type"}},
        // REQ_UNKNOWN
        { .in = "{ \"type\": \"req\", \"req\": \"huh?\",\"reqId\": 2}",
          .out = {.req="huh?",    .reqId=2, .status=400,
                  .message="Req field was not expected value"}},
        // REQ_PARAM_ERR (body field is required for SetCfg)
        { .in = "{ \"type\": \"req\", \"req\": \"SetCfg\",\"reqId\": 3}",
          .out = {.req="SetCfg",  .reqId=3, .status=400,
                  .message="Based on the req field, expected fields were missing"}},
        // REQ_SET_CFG
        { .in = "{ \"type\": \"req\", \"req\": \"SetCfg\",\"reqId\": 4, \"body\": {}}",
          .out = {.req="SetCfg",  .reqId=4, .status=200, .message=NULL}},
        // REQ_GET_CFG
        { .in = "{ \"type\": \"req\", \"req\": \"GetCfg\",\"reqId\": 5}",
          .out = {.req="GetCfg",  .reqId=5, .status=200, .message=NULL}},
        // REQ_GET_DIAG
        { .in = "{ \"type\": \"req\", \"req\": \"GetDiag\",\"reqId\": 6}",
          .out = {.req="GetDiag", .reqId=6, .status=200, .message=NULL}},
    };

    int i;
    for (i=0; i<sizeof(test)/sizeof(test[0]); i++) {
        test_t* test_case = &test[i];
        char expected[512];

        // ctlParseRxMsg is not under test here, but is used to create upload.req
        //printf("%s\n", test_case->in);
        upload.req = ctlParseRxMsg(test_case->in);
        assert_non_null(upload.req);

        // If req is non-null, tx_msg should exist
        tx_msg = ctlCreateTxMsg(&upload);
        assert_non_null(tx_msg);
        //printf("%s\n", tx_msg);

        // verify type field (constant value of resp)
        assert_non_null(strstr(tx_msg, "\"type\":\"resp\""));

        // verify body field (does not exist in these test cases)
        assert_null(strstr(tx_msg, "\"body\":"));

        // verify req field
        if (test_case->out.req) {
            snprintf(expected, sizeof(expected), "\"req\":\"%s\"", test_case->out.req);
            assert_non_null(strstr(tx_msg, expected));
        } else {
            assert_null(strstr(tx_msg, "\"req\":"));
        }

        // verify reqId field
        snprintf(expected, sizeof(expected), "\"reqId\":%d", test_case->out.reqId);
        assert_non_null(strstr(tx_msg, expected));

        // verify status field
        snprintf(expected, sizeof(expected), "\"status\":%d", test_case->out.status);
        assert_non_null(strstr(tx_msg, expected));

        // verify message field
        if (test_case->out.message) {
            snprintf(expected, sizeof(expected), "\"message\":\"%s\"", test_case->out.message);
            assert_non_null(strstr(tx_msg, expected));
        } else {
            assert_null(strstr(tx_msg, "\"message\":"));
        }

        scope_free(tx_msg);
        destroyReq(&upload.req);
    }

    // Verify body is sent if it exists in the upload
    upload.req = ctlParseRxMsg("{ \"type\": \"req\", \"req\": \"GetCfg\",\"reqId\": 7}");
    upload.body = cJSON_Parse("\"Gnarly\"");
    tx_msg = ctlCreateTxMsg(&upload);
    //printf("%s\n", tx_msg);
    assert_non_null(tx_msg);
    assert_non_null(strstr(tx_msg, "\"body\":\"Gnarly\""));
    scope_free(tx_msg);
    destroyReq(&upload.req);
}

static void
ctlCreateTxMsgEvt(void** state)
{
    upload_t upload = { .type = UPLD_EVT, .body = NULL, .req = NULL, .uid = 0, .proc = NULL };
    
    // If body is null, msg should be null
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 0);
    char* msg = ctlCreateTxMsg(&upload);
    assert_null(msg);
    assert_int_equal(dbgCountMatchingLines("src/ctl.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

    // If body is non-null, msg should exist
    upload.body = cJSON_Parse("\"yeah, dude\"");
    msg = ctlCreateTxMsg(&upload);
    assert_non_null(msg);

    char expected_msg[] =
        "{\"type\":\"evt\",\"_channel\":\"none\",\"body\":\"yeah, dude\"}";
    assert_string_equal(msg, expected_msg);

    scope_free(msg);
}


static void
ctlSendMsgForNullMtcDoesntCrash(void** state)
{
    char* msg = scope_strdup("Hey, this is cool!\n");
    ctlSendMsg(NULL, msg);
}

static void
ctlSendMsgForNullMessageDoesntCrash(void** state)
{
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    ctlTransportSet(ctl, t, CFG_CTL);
    ctlSendMsg(ctl, NULL);
    ctlDestroy(&ctl);
}

static void
ctlTransportSetAndMtcSend(void** state)
{
    const char* file_path = "/tmp/my.path";
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    transport_t* t1 = transportCreateUdp("127.0.0.1", "12345");
    transport_t* t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t* t3 = transportCreateSyslog();
    transport_t* t4 = transportCreateShm();
    transport_t* t5 = transportCreateFile(file_path, CFG_BUFFER_FULLY);
    ctlTransportSet(ctl, t1, CFG_CTL);
    ctlTransportSet(ctl, t2, CFG_CTL);
    ctlTransportSet(ctl, t3, CFG_CTL);
    ctlTransportSet(ctl, t4, CFG_CTL);
    ctlTransportSet(ctl, t5, CFG_CTL);

    // Test that transport is set by testing side effects of ctlSendMsg
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    char* msg = scope_strdup("Something to send\n");
    ctlSendMsg(ctl, msg);

    // With CFG_BUFFER_FULLY, this output only happens with the flush
    long file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    ctlFlush(ctl);
    file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that transport is cleared by seeing no side effects.
    ctlTransportSet(ctl, NULL, CFG_CTL);
    file_pos_before = fileEndPosition(file_path);
    msg = scope_strdup("Something else to send\n");
    ctlSendMsg(ctl, msg);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (scope_unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    ctlDestroy(&ctl);
}

static void
ctlAddProtocol(void** state)
{
    char dummy[] = "{\"type\": \"req\",\"req\": \"AddProto\",\"reqId\":6393,\"body\":{\"binary\":\"false\",\"regex\":\"^[*][[:digit:]]+\",\"pname\":\"Dummy\",\"len\":12}}";

    request_t *req = ctlParseRxMsg(dummy);
    assert_non_null(req);

    assert_int_equal(req->cmd, REQ_ADD_PROTOCOL);
    assert_string_equal(req->cmd_str, "AddProto");
    assert_string_equal(req->protocol->protname, "Dummy");

    destroyProtEntry(req->protocol);
    destroyReq(&req);
}

static void
ctlDelProtocol(void** state)
{
    char deldummy[] = "{\"type\": \"req\",\"req\": \"DelProto\",\"reqId\":6394,\"body\":{\"pname\":\"Dummy\"}}";

    request_t *req = ctlParseRxMsg(deldummy);
    assert_non_null(req);
    assert_string_equal(req->cmd_str, "DelProto");

    destroyProtEntry(req->protocol);
    destroyReq(&req);
}

static void
ctlSendLogConsoleAsciiData(void **state)
{
    initState();
    const char* console_path = "stdout";
    const char* ascii_text = "hello world<>?% _";
    proc_id_t proc = {.pid = 1,
                      .ppid = 1,
                      .hostname = "foo",
                      .procname = "foo",
                      .cmd = "foo",
                      .id = "foo"};
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    bool b_res = ctlEvtSourceEnabled(ctl, CFG_SRC_CONSOLE);
    assert_true(b_res);
    allow_copy_buf_data(TRUE);

    ctlSendLog(ctl, STDOUT_FILENO, console_path, ascii_text, strlen(ascii_text), 0, &proc);
    ctlFlushLog(ctl);
    const char *val = get_cbuf_data();
    assert_string_equal(ascii_text, val);
    ctlDestroy(&ctl);
    allow_copy_buf_data(FALSE);
}

static void
ctlSendLogConsoleNoneAsciiData(void **state)
{
    initState();
    const char* console_path = "stdout";
    char* non_basic_ascii_text = scope_malloc(sizeof(char)*4);
    assert_true(non_basic_ascii_text);
    non_basic_ascii_text[0] = 128;
    non_basic_ascii_text[1] = 157;
    non_basic_ascii_text[2] = 234;
    non_basic_ascii_text[3] = '\0';

    const char* binary_data_event_msg = "-- binary data ignored --";
    proc_id_t proc = {.pid = 1,
                      .ppid = 1,
                      .hostname = "foo",
                      .procname = "foo",
                      .cmd = "foo",
                      .id = "foo"};
    ctl_t* ctl = ctlCreate();
    assert_non_null(ctl);
    bool b_res = ctlEvtSourceEnabled(ctl, CFG_SRC_CONSOLE);
    assert_true(b_res);
    allow_copy_buf_data(TRUE);

    ctlSendLog(ctl, STDOUT_FILENO, console_path, non_basic_ascii_text, strlen(non_basic_ascii_text), 0, &proc);
    ctlFlushLog(ctl);
    const char *val = get_cbuf_data();
    assert_string_not_equal(non_basic_ascii_text, val);
    assert_string_equal(binary_data_event_msg, val);
    ctlDestroy(&ctl);

    // do this again, with ALLOW_BINARY true
    // and verify that the binary_data_event_msg does *not* appear.
    memset(cbuf_data, '\0', sizeof(cbuf_data));

    setenv("SCOPE_ALLOW_BINARY_CONSOLE", "true", 1);
    ctl = ctlCreate();
    assert_non_null(ctl);
    b_res = ctlEvtSourceEnabled(ctl, CFG_SRC_CONSOLE);
    assert_true(b_res);
    ctlSendLog(ctl, STDOUT_FILENO, console_path, non_basic_ascii_text, strlen(non_basic_ascii_text), 0, &proc);
    ctlFlushLog(ctl);
    val = get_cbuf_data();
    assert_string_not_equal(binary_data_event_msg, val);
    assert_string_equal(non_basic_ascii_text, val);
    ctlDestroy(&ctl);
    unsetenv("SCOPE_ALLOW_BINARY_CONSOLE");

    scope_free(non_basic_ascii_text);
    allow_copy_buf_data(FALSE);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ctlParseRxMsgNullReturnsParseError),
        cmocka_unit_test(ctlParseRxMsgUnparseableReturnsParseError),
        cmocka_unit_test(ctlParseRxMsgRequiredFieldProblemsReturnsMalformed),
        cmocka_unit_test(ctlParseRxMsgBogusReqReturnsUnknown),
        cmocka_unit_test(ctlParseRxMsgSetCfgWithoutDataObjectReturnsParamErr),
        cmocka_unit_test(ctlParseRxMsgSetCfg),
        cmocka_unit_test(ctlParseRxMsgGetCfg),
        cmocka_unit_test(ctlParseRxMsgGetDiags),
        cmocka_unit_test(ctlParseRxMsgBlockPort),
        cmocka_unit_test(ctlParseRxMsgSwitch),
        cmocka_unit_test(ctlCreateTxMsgReturnsNullForNullUpload),
        cmocka_unit_test(ctlCreateTxMsgInfo),
        cmocka_unit_test(ctlCreateTxMsgResp),
        cmocka_unit_test(ctlCreateTxMsgEvt),
        cmocka_unit_test(ctlSendMsgForNullMtcDoesntCrash),
        cmocka_unit_test(ctlSendMsgForNullMessageDoesntCrash),
        cmocka_unit_test(ctlTransportSetAndMtcSend),
        cmocka_unit_test(ctlAddProtocol),
        cmocka_unit_test(ctlDelProtocol),
        cmocka_unit_test(ctlSendLogConsoleAsciiData),
        cmocka_unit_test(ctlSendLogConsoleNoneAsciiData),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };

    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}

