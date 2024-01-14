#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "circbuf.h"
#include "cfgutils.h"
#include "com.h"
#include "ctl.h"
#include "dbg.h"
#include "com.h"
#include "evtutils.h"
#include "fn.h"
#include "state.h"
#include "scopestdlib.h"
#include "utils.h"

#define FS_ENTRIES 1024
#define DEFAULT_LOG_MAX_AGG_BYTES 32768
#define DEFAULT_LOG_FLUSH_PERIOD_IN_MS 2000

#define CHANNEL "_channel"
#define ID "id"

#define BINARY_DATA_MSG "-- binary data ignored --"
#define DEFAULT_BINARY_DATA_SAMPLE_SIZE (256U)
#define ESC_CHARACTER (0x1B)

typedef struct {
    char *buf;
    size_t bufsize;
    FILE *stream;
    unsigned long long tot_size;
    log_id_t id;
} streambuf_t;

struct _ctl_t
{
    // This transport serves two purposes. When AppScope is configured to
    // connect to LogStream, this is is that connection. Otherwise, this is
    // the event connection.
    transport_t *transport;

    // This is the transport for payloads when AppScope is configured to
    // connect to LogStream. Otherwise, this isn't used because we handle
    // payloads differently - to separate local files.
    transport_t *paytrans;

    evt_fmt_t *evt;
    cbuf_handle_t events;
    unsigned enhancefs;
    bool allow_binary_console;
    bool stop_aggregating;

    // Used to buffer (aggregate) log and console data
    struct {
        // queuing from their thread to our own
        cbuf_handle_t ringbuf;

        // storage for aggregating log and console data
        streambuf_t streamAgg[FS_ENTRIES];

        // limits for how much raw data to aggregate
        // and how long to aggregate without reporting
        unsigned long max_agg_bytes;
        unsigned long flush_period_in_ms;
    } log;

    struct {
        payload_status_t status;
        char * dir;
        char * dirRepr;         // human-representation of dir variable: dir://<dir value> (TODO: Unify this with dir)
        cbuf_handle_t ringbuf;
    } payload;

    // Temporary, I believe...  only used for command/response w/cribl
    cbuf_handle_t msgbuf;
};

typedef struct {
    const char *str;
    cmd_t cmd;
} cmd_map_t;

static cmd_map_t cmd_map[] = {
    {"SetCfg",           REQ_SET_CFG},
    {"GetCfg",           REQ_GET_CFG},
    {"GetDiag",          REQ_GET_DIAG},
    {"BlockPort",        REQ_BLOCK_PORT},
    {"Switch",           REQ_SWITCH},
    {"AddProto",         REQ_ADD_PROTOCOL},
    {"DelProto",         REQ_DEL_PROTOCOL},
    {NULL,               REQ_UNKNOWN}
};

typedef struct {
    const char *str;
    switch_action_t action;
} switch_map_t;

static switch_map_t switch_map[] = {
    {"detach",           FUNC_DETACH},
    {"attach",           FUNC_ATTACH},
    {NULL,               NO_ACTION}
};

static bool srcEnabledDefault[] = {
    DEFAULT_SRC_FILE,
    DEFAULT_SRC_CONSOLE,
    DEFAULT_SRC_SYSLOG,
    DEFAULT_SRC_METRIC,
    DEFAULT_SRC_HTTP,
    DEFAULT_SRC_NET,
    DEFAULT_SRC_FS,
    DEFAULT_SRC_DNS,
};

static void
grab_supplemental_for_block_port(cJSON *json_root, request_t *req)
{
    cJSON *json;

    if (!json_root || !req) return;

    // If "body" exists, is a number, and is in range
    // then set port (starts blocking tcp connections on that port).
    // Interpret everything else as an attempt to clear port (stops
    // the "block tcp connections" feature).

    json = cJSON_GetObjectItem(json_root, "body");
    if (!json || !cJSON_IsNumber(json) ||
        (json->valuedouble < 0) ||
        (json->valuedouble > USHRT_MAX)) {
        // Turn off the port blocking
        req->port = DEFAULT_PORTBLOCK;
    } else {
        // Everything looks good!  Grab the value and return.
        req->port = json->valuedouble;
    }
}

static void
grab_supplemental_for_set_cfg(cJSON * json_root, request_t *req)
{
    cJSON *json;
    char *string = NULL;

    if (!json_root || !req) goto error;

    // This expects a json version of our scope.yml file
    // See cfgReadGoodJson() in test/cfgutilstest.c or
    // ctlParseRxMsgSetCfg() in test/ctltest.c for an example
    // of what we expect to find in the "body" json node.
    json = cJSON_GetObjectItem(json_root, "body");
    if (!json || !cJSON_IsObject(json)) goto error;

    // Create a string from the "body" json object
    string = cJSON_PrintUnformatted(json);
    if (!string) goto error;

    // Feed the string to the yaml parser to get a cfg
    req->cfg = cfgFromString(string);
    scope_free(string);

    if (!req->cfg) goto error;

    // Everything worked!
    return;

error:
    // body is required for REQ_SET_CFG
    req->cmd=REQ_PARAM_ERR;
}

static void
grab_supplemental_for_switch(cJSON *json_root, request_t *req)
{
    cJSON *json;
    char *string = NULL;
    switch_map_t *map;

    if (!json_root || !req) goto error;

    // This expects a single string in the "body" field
    json = cJSON_GetObjectItem(json_root, "body");
    if (!json || !(string = cJSON_GetStringValue(json))) goto error;

    req->action = NO_ACTION;

    // search switch_map for commands we handle
    for (map=switch_map; map->str; map++) {
        if (!scope_strcmp(string, map->str)) {
            req->action = map->action;
            break;
        }
    }

    // If we didn't find anything we handle, consider it an error
    if (req->action == NO_ACTION) goto error;

    // Everything worked!
    return;

error:
    // body is required for REQ_SET_CFG
    req->cmd=REQ_PARAM_ERR;
}

static void
grab_supplemental_for_def_protocol(cJSON * json_root, request_t *req)
{
    cJSON *json, *body;
    char *str;
    protocol_def_t *prot = NULL;

    if (!json_root || !req) goto err;

    // The body includes the definition of a protocol
    body = cJSON_GetObjectItem(json_root, "body");
    if (!body || !cJSON_IsObject(body)) goto err;

    if ((prot = scope_calloc(1, sizeof(protocol_def_t))) == NULL) goto err;

    req->protocol = prot;

    json = cJSON_GetObjectItem(body, "binary");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    if (scope_strncmp("false", str, scope_strlen(str)) == 0) {
        prot->binary = FALSE;
    } else {
        prot->binary = TRUE;
    }

    // len is optional
    json = cJSON_GetObjectItem(body, "len");
    if (!json || !cJSON_IsNumber(json)) {
        prot->len = 0;
    } else {
        prot->len = json->valueint;
    }

    json = cJSON_GetObjectItem(body, "regex");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    prot->regex = scope_strdup(str);

    json = cJSON_GetObjectItem(body, "pname");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    prot->protname = scope_strdup(str);

    return;

err:
    if (req) req->cmd=REQ_PARAM_ERR;
    if (prot && prot->regex) scope_free(prot->regex);
    if (prot && prot->protname) scope_free(prot->protname);
    if (prot) scope_free(prot);
}

static void
grab_supplemental_for_del_protocol(cJSON * json_root, request_t *req)
{
    cJSON *json, *body;
    char *str;
    protocol_def_t *prot = NULL;

    if (!json_root || !req) goto err;

    // The body includes the definition of a protocol
    body = cJSON_GetObjectItem(json_root, "body");
    if (!body || !cJSON_IsObject(body)) goto err;

    if ((prot = scope_calloc(1, sizeof(protocol_def_t))) == NULL) goto err;

    req->protocol = prot;

    json = cJSON_GetObjectItem(body, "pname");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    prot->protname = scope_strdup(str);

    return;

err:
    if (req) req->cmd=REQ_PARAM_ERR;
    if (prot && prot->protname) scope_free(prot->protname);
    if (prot) scope_free(prot);
}

request_t *
ctlParseRxMsg(const char *msg)
{
    cJSON *json_root = NULL;
    cJSON *json;
    char *str;
    cmd_map_t *map;

    request_t *req = scope_calloc(1, sizeof(request_t));
    if (!req) {
        DBG(NULL);
        goto out;
    }

    //
    // phase 1, just try to parse msg as a json object
    //
    req->cmd = REQ_PARSE_ERR;
    if (!msg) {
        DBG(NULL);
        goto out;
    }

    if (!(json_root = cJSON_Parse(msg))) goto out;

    // Top level should be an object
    if (!cJSON_IsObject(json_root)) goto out;

    //
    // phase 2, grab required fields
    //
    req->cmd = REQ_MALFORMED;

    // grab reqId field first so we'll have it even if some other
    // part of the json isn't usable for some reason.
    json = cJSON_GetObjectItem(json_root, "reqId");
    if (!json || !cJSON_IsNumber(json)) goto out;
    req->id = json->valuedouble;

    // grab req field
    json = cJSON_GetObjectItem(json_root, "req");
    if (!json) goto out;
    if (!(str = cJSON_GetStringValue(json))) goto out;
    req->cmd_str = scope_strdup(str);

    // verify that type field exists, with required value "req"
    json = cJSON_GetObjectItem(json_root, "type");
    if (!json) goto out;
    if (!(str = cJSON_GetStringValue(json))) goto out;
    if (scope_strcmp(str, "req")) goto out;

    //
    // phase 3, interpret what we grabbed
    //
    req->cmd = REQ_UNKNOWN;

    // make sure that the req field is a value we expect
    for (map=cmd_map; map->str; map++) {
        if (!scope_strcmp(req->cmd_str, map->str)) {
            req->cmd = map->cmd;
            break;
        }
    }
    if (req->cmd == REQ_UNKNOWN) goto out;

    //
    // phase 4, grab supplemental info from body field as required
    //
    switch (req->cmd) {
        case REQ_PARSE_ERR:
        case REQ_MALFORMED:
        case REQ_UNKNOWN:
        case REQ_PARAM_ERR:
            // Shouldn't be checking these for body field
            DBG("Unexpected Cmd: %d", req->cmd);
            break;
        case REQ_SET_CFG:
            // Right now, only REQ_SET_CFG has required body field
            grab_supplemental_for_set_cfg(json_root, req);
            break;
        case REQ_GET_CFG:
        case REQ_GET_DIAG:
            // body field not used
            break;
        case REQ_BLOCK_PORT:
            grab_supplemental_for_block_port(json_root, req);
            break;
        case REQ_SWITCH:
            grab_supplemental_for_switch(json_root, req);
            break;
        case REQ_ADD_PROTOCOL:
            grab_supplemental_for_def_protocol(json_root, req);
            break;
        case REQ_DEL_PROTOCOL:
            grab_supplemental_for_del_protocol(json_root, req);
            break;
        default:
            DBG("Unknown Cmd: %d", req->cmd);
    }

out:
    if (json_root) cJSON_Delete(json_root);
    return req;
}

void
destroyReq(request_t **request)
{
    if (!request || !*request) return;

    request_t *req = *request;

    if (req->cmd_str) scope_free(req->cmd_str);
    if (req->cfg) cfgDestroy(&req->cfg);
    // Note: don't mess with the protocol object here
    scope_free(req);

    *request=NULL;
}

static cJSON *
create_info_json(upload_t *upld)
{
    cJSON *json_root = NULL;
    if (!upld || !upld->body) goto err;

    if (!(json_root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(json_root, "type", "info")) goto err;
    cJSON_AddItemToObjectCS(json_root, "body", upld->body);

    return json_root;
err:
    DBG("upld:%p upld->body:%p json_root:%p",
        upld, (upld)?upld->body:NULL, json_root);
    if (json_root) cJSON_Delete(json_root);
    return NULL;
}

static cJSON *
create_resp_json(upload_t *upld)
{
    cJSON *json_root = NULL;
    if (!upld || !upld->req) goto err;

    if (!(json_root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(json_root, "type", "resp")) goto err;

    // upld->body is optional
    if (upld->body) {
        // Move the upld->body from the upld to json_root
        cJSON_AddItemToObjectCS(json_root, "body", upld->body);
        upld->body = NULL;
    }

    // If we had trouble parsing, we might not have cmd_str
    if (upld->req->cmd_str) {
        if (!cJSON_AddStringToObjLN(json_root, "req", upld->req->cmd_str)) goto err;
    }
    if (!cJSON_AddNumberToObjLN(json_root, "reqId", upld->req->id)) goto err;

    int status = 200;
    char *message = NULL;
     switch (upld->req->cmd) {
        case REQ_PARSE_ERR:
            status = 400;
            message = "Request could not be parsed as a json object";
            break;
        case REQ_MALFORMED:
            status = 400;
            message = "Type was not request, required fields were missing or of wrong type";
            break;
        case REQ_UNKNOWN:
            status = 400;
            message = "Req field was not expected value";
            break;
        case REQ_PARAM_ERR:
            status = 400;
            message = "Based on the req field, expected fields were missing";
            break;
        case REQ_SET_CFG:
        case REQ_GET_CFG:
        case REQ_GET_DIAG:
        case REQ_BLOCK_PORT:
        case REQ_SWITCH:
        case REQ_ADD_PROTOCOL:
        case REQ_DEL_PROTOCOL:
            break;
        default:
            DBG(NULL);
    }
    if (!cJSON_AddNumberToObjLN(json_root, "status", status)) goto err;
    if (message) {
        if (!cJSON_AddStringToObjLN(json_root, "message", message)) goto err;
    }

    return json_root;
err:
    DBG("upld:%p upld->body:%p upld->req:%p json_root:%p",
        upld, (upld)?upld->body:NULL, (upld)?upld->req:NULL, json_root);
    if (json_root) cJSON_Delete(json_root);
    return NULL;
}

static cJSON *
create_evt_json(upload_t *upld)
{
    cJSON *json_root = NULL;
    char numbuf[32];
    if (!upld || !upld->body) goto err;

    if (!(json_root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(json_root, "type", "evt")) goto err;
    if (upld->proc && !cJSON_AddStringToObjLN(json_root, ID, upld->proc->id)) goto err;
    if (upld->uid) {
        if (scope_snprintf(numbuf, sizeof(numbuf), "%llu", upld->uid) < 0) goto err;
        if (!cJSON_AddStringToObjLN(json_root, CHANNEL, numbuf)) goto err;
    } else {
        if (!cJSON_AddStringToObjLN(json_root, CHANNEL, "none")) goto err;
    }
    cJSON_AddItemToObjectCS(json_root, "body", upld->body);
    return json_root;
err:
    DBG("upld:%p upld->body:%p json_root:%p",
        upld, (upld)?upld->body:NULL, json_root);
    if (json_root) cJSON_Delete(json_root);
    return NULL;
}

static char *
prepMessage(upload_t *upld)
{
    char *streamMsg;

    if (!upld) return NULL;

    streamMsg = ctlCreateTxMsg(upld);
    if (!streamMsg) return NULL;

    // Add the newline delimiter to the msg.
    int strsize = scope_strlen(streamMsg);
    char *temp = scope_realloc(streamMsg, strsize+2); // room for "\n\0"
    if (!temp) {
        DBG(NULL);
        scopeLogInfo("CTL scope_realloc error");
        scope_free(streamMsg);
        return NULL;
    }

    streamMsg = temp;
    streamMsg[strsize] = '\n';
    streamMsg[strsize+1] = '\0';

    return streamMsg;
}

char *
ctlCreateTxMsg(upload_t *upld)
{
    cJSON *json = NULL;
    char *msg = NULL;

    if (!upld) goto out;

    switch (upld->type) {
        case UPLD_INFO:
            json = create_info_json(upld);
            break;
        case UPLD_RESP:
            json = create_resp_json(upld);
            break;
        case UPLD_EVT:
            json = create_evt_json(upld);
            break;
        default:
            DBG(NULL);
            goto out;
    }
    if (!json) goto out;

    msg = cJSON_PrintUnformatted(json);

out:
    if (json) cJSON_Delete(json);
    return msg;
}

ctl_t *
ctlCreate(void)
{
    ctl_t *ctl = scope_calloc(1, sizeof(ctl_t));
    if (!ctl) {
        DBG(NULL);
        goto err;
    }

    size_t buf_size = DEFAULT_CBUF_SIZE;
    char *qlen_str;
    if ((qlen_str = fullGetEnv("SCOPE_QUEUE_LENGTH")) != NULL) {
        unsigned long qlen;
        scope_errno = 0;
        qlen = scope_strtoul(qlen_str, NULL, 10);
        if (!scope_errno && qlen) {
            buf_size = qlen;
        }
    }

    ctl->log.ringbuf = cbufInit(buf_size);
    if (!ctl->log.ringbuf) {
        DBG(NULL);
        goto err;
    }
    ctl->log.max_agg_bytes = DEFAULT_LOG_MAX_AGG_BYTES;
    ctl->log.flush_period_in_ms = DEFAULT_LOG_FLUSH_PERIOD_IN_MS;

    ctl->events = cbufInit(buf_size);
    if (!ctl->events) {
        DBG(NULL);
        goto err;
    }

    ctl->enhancefs = DEFAULT_ENHANCE_FS;
    ctl->allow_binary_console = DEFAULT_ALLOW_BINARY_CONSOLE;
    ctl->stop_aggregating = FALSE;

    ctl->payload.status = PAYLOAD_STATUS_DISABLE;
    ctl->payload.dir = (DEFAULT_PAYLOAD_DIR) ? scope_strdup(DEFAULT_PAYLOAD_DIR) : NULL;
    ctl->payload.dirRepr = NULL;
    if (DEFAULT_PAYLOAD_DIR_REPR) {
        if (scope_asprintf(&ctl->payload.dirRepr, "dir://%s", ctl->payload.dir) < 0) {
            ctl->payload.dirRepr = NULL;
        }
    }
    ctl->payload.ringbuf = cbufInit(buf_size);

    if (!ctl->payload.ringbuf) {
        DBG(NULL);
        goto err;
    }

    ctl->msgbuf = cbufInit(1000);
    if (!ctl->msgbuf) {
        DBG(NULL);
        goto err;
    }

    return ctl;
err:
    ctlDestroy(&ctl);
    return ctl;
}

void
ctlDestroy(ctl_t **ctl)
{
    if (!ctl || !*ctl) return;

    ctlStopAggregating(*ctl);
    ctlFlush(*ctl);
    cbufFree((*ctl)->log.ringbuf);
    cbufFree((*ctl)->msgbuf);
    cbufFree((*ctl)->events);

    if ((*ctl)->payload.dir) {
        scope_free((*ctl)->payload.dir);
    }
    if ((*ctl)->payload.dirRepr) {
        scope_free((*ctl)->payload.dirRepr);
    }

    cbufFree((*ctl)->payload.ringbuf);

    transportDestroy(&(*ctl)->transport);
    transportDestroy(&(*ctl)->paytrans);
    evtFormatDestroy(&(*ctl)->evt);

    scope_free(*ctl);
    *ctl = NULL;
}

void
ctlSendMsg(ctl_t *ctl, char *msg)
{
    if (!msg) return;
    if (!ctl) {
        scope_free(msg);
        return;
    }

    if (cbufPut(ctl->msgbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        scope_free(msg);
    }
}

// send raw json (no envelope/messaging protocol), no buffering
int
ctlSendJson(ctl_t *ctl, cJSON *json, which_transport_t who)
{
    if (!json) return -1;
    if (!ctl) {
        cJSON_Delete(json);
        return -1;
    }

    char *msg = cJSON_PrintUnformatted(json);

    int rc = -1;

    if (msg && ((msg = msgAddNewLine(msg)) != NULL)) {
        if (who == CFG_LS) {
            rc = transportSend(ctl->paytrans, msg, scope_strlen(msg));
        } else {
            rc = transportSend(ctl->transport, msg, scope_strlen(msg));
        }
    }

    if (msg) scope_free(msg);
    cJSON_Delete(json);
    return rc;
}

int
ctlPostMsg(ctl_t *ctl, cJSON *body, upload_type_t type, request_t *req, bool now)
{
    int rc = -1;
    char *streamMsg;
    upload_t upld;

    if (type == UPLD_RESP) {
        // req is required for UPLD_RESP type
        if (!req || !ctl) return rc;
    } else {
        // body is required for all other types
        if (!body) return rc;
        if (!ctl) {
            cJSON_Delete(body);
            return rc;
        }
    }

    upld.type = type;
    upld.body = body;
    upld.req = req;
    upld.proc = NULL;
    upld.uid = 0;
    streamMsg = ctlCreateTxMsg(&upld);

    if (streamMsg) {
        // on the ring buffer
        ctlSendMsg(ctl, streamMsg);

        // send it now or periodic
        if (now) ctlFlush(ctl);
        rc = 0;
    }

    return rc;
}

int
ctlSendHttp(ctl_t *ctl, event_t *evt, uint64_t uid, proc_id_t *proc)
{
    int rc;
    char *streamMsg;
    cJSON *json;
    upload_t upld;

    if (!ctl || !evt || !proc) return -1;

    // get a cJSON object for the given event
    if ((json = evtFormatHttp(ctl->evt, evt, uid, proc)) == NULL) return -1;

    // send it
    upld.type = UPLD_EVT;
    upld.body = json;
    upld.req = NULL;
    upld.uid = uid;
    upld.proc = proc;
    streamMsg = prepMessage(&upld);

    rc = transportSend(ctl->transport, streamMsg, scope_strlen(streamMsg));
    if (streamMsg) scope_free(streamMsg);
    return rc;
}

int
ctlSendEvent(ctl_t *ctl, event_t *evt, uint64_t uid, proc_id_t *proc)
{
    int rc;
    char *streamMsg;
    cJSON *json;
    upload_t upld;

    if (!ctl || !evt || !proc) return -1;

    // get a cJSON object for the given event
    if ((json = evtFormatMetric(ctl->evt, evt, uid, proc)) == NULL) return -1;

    // send it
    upld.type = UPLD_EVT;
    upld.body = json;
    upld.req = NULL;
    upld.uid = uid;
    upld.proc = proc;
    streamMsg = prepMessage(&upld);

    rc = transportSend(ctl->transport, streamMsg, scope_strlen(streamMsg));
    if (streamMsg) scope_free(streamMsg);
    return rc;

}

int
ctlPostEvent(ctl_t *ctl, char *event)
{
    if (!event) return -1;
    if (!ctl) {
        evtFree((evt_type *)event);
        return -1;
    }

    if (cbufPut(ctl->events, (uint64_t)event) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        evtFree((evt_type *)event);
        return -1;
    }
    return 0;
}

static log_event_t *
createInternalLogEvent(int fd, const char *path, const void *buf, size_t count, uint64_t uid, proc_id_t *proc, watch_t logType, regex_t *valfilter)
{
    log_event_t *event = scope_calloc(1, sizeof(*event));
    char *data = scope_malloc(count);
    char *src = scope_strdup(path);

    if (!event || !data || !path) {
        DBG("event = %p, data = %p, src = %p", event, data, src);
        if (event) scope_free(event);
        if (data) scope_free(data);
        if (src) scope_free(src);
        return NULL;
    }

    scope_memcpy(data, buf, count);

    struct timeval tv;
    scope_gettimeofday(&tv, NULL);
    event->fd = fd;
    event->id.uid = uid;
    event->id.timestamp = tv.tv_sec + tv.tv_usec/1e6;
    event->id.path = src;
    event->id.proc = proc;
    event->id.sourcetype = logType;
    event->id.valuefilter = valfilter;
    event->data = data;
    event->datalen = count;

    return event;
}

static void
destroyInternalLogEvent(log_event_t **eventptr)
{
    if (!eventptr || !*eventptr) return;

    log_event_t *event = *eventptr;
    if (event->id.path) scope_free(event->id.path);
    if (event->data)    scope_free(event->data);
    if (event)          scope_free(event);
    *eventptr = NULL;
}

static bool
is_data_binary(const void *buf, size_t count)
{
    const char* b_buf = (const char *)buf;
    size_t min_len = (count < DEFAULT_BINARY_DATA_SAMPLE_SIZE) ? count : DEFAULT_BINARY_DATA_SAMPLE_SIZE;
    size_t i;
    for (i = 0; i < min_len; i++) {
        if (!scope_isprint(b_buf[i]) && !scope_isspace(b_buf[i]) && b_buf[i] != ESC_CHARACTER) {
            return TRUE;
        }
    }
    return FALSE;
}

int
ctlSendLog(ctl_t *ctl, int fd, const char *path, const void *buf, size_t count, uint64_t uid, proc_id_t *proc)
{
    if (!ctl || !path || !buf || !proc) return -1;

    regex_t *filter;
    watch_t logType;
    if (evtFormatSourceEnabled(ctl->evt, CFG_SRC_CONSOLE) &&
       (filter = evtFormatNameFilter(ctl->evt, CFG_SRC_CONSOLE)) &&
       (!regexec_wrapper(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_CONSOLE;
    } else if (evtFormatSourceEnabled(ctl->evt, CFG_SRC_FILE) &&
       (filter = evtFormatNameFilter(ctl->evt, CFG_SRC_FILE)) &&
       (!regexec_wrapper(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_FILE;
    } else {
        return 0;
    }

    // We can't run the value filter on what might be raw binary data.
    // Grab the correct one for our logType, and send a pointer of
    // it to be used later, after we've created a string from the data.
    filter = evtFormatValueFilter(ctl->evt, logType);

    log_event_t *logevent = NULL;
    if (!ctl->allow_binary_console && (logType == CFG_SRC_CONSOLE)) {

        // Grab previous data_content, then compute and save new data_content
        fs_content_type_t prev_data_content = getFSContentType(fd);
        fs_content_type_t cur_data_content = is_data_binary(buf, count) ? FS_CONTENT_BINARY : FS_CONTENT_TEXT;
        setFSContentType(fd, cur_data_content);

        // Report only first event of binary data, drop and ignore rest
        if (cur_data_content == FS_CONTENT_BINARY) {
            if (prev_data_content != FS_CONTENT_BINARY) {
                logevent = createInternalLogEvent(fd, path, BINARY_DATA_MSG, C_STRLEN(BINARY_DATA_MSG), uid, proc, logType, filter);
            } else {
                return -1;
            }
        }
    }

    // This will be true for CFG_SRC_FILE, or if CFG_SRC_CONSOLE is TEXT.
    if (!logevent) {
        logevent = createInternalLogEvent(fd, path, buf, count, uid, proc, logType, filter);
    }

    if (cbufPut(ctl->log.ringbuf, (uint64_t)logevent) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        destroyInternalLogEvent(&logevent);
        return -1;
    }
    return 0;
}

static cJSON *
createLogEventJson(ctl_t *ctl, streambuf_t *stmbuf)
{
    cJSON *root = NULL;
    cJSON *data;
    int successful = FALSE;

    event_format_t event;
    event.timestamp = stmbuf->id.timestamp;
    event.src = stmbuf->id.path;
    event.proc = stmbuf->id.proc;
    event.uid = stmbuf->id.uid;
    event.sourcetype = stmbuf->id.sourcetype;

    scope_fclose(stmbuf->stream);  // updates stmbuf->buf, stmbuf->bufsize
    stmbuf->stream = NULL;

    if (!(root = cJSON_CreateObject())) goto out;
    if (!(data = cJSON_CreateStringFromBuffer(stmbuf->buf, stmbuf->bufsize))) goto out;
    cJSON_AddItemToObjectCS(root, "message", data);
    event.data = root;

    if (data && data->valuestring) {
        regex_t *filter = stmbuf->id.valuefilter;
        if (filter && regexec_wrapper(filter, data->valuestring, 0, NULL, 0)) {
            // This event doesn't match.  Drop it on the floor.
            goto out;
        }
    }

    if (!(root = fmtEventJson(ctl->evt, &event))) goto out;

    successful = TRUE;
out:
    if (!successful && root) {
        cJSON_Delete(root);
        root = NULL;
    }
    scope_free(stmbuf->buf);
    scope_free(stmbuf->id.path);

    return root;
}

static void
sendBufferedMessages(ctl_t *ctl)
{
    uint64_t data;
    while (cbufGet(ctl->msgbuf, &data) == 0) {
        if (data) {
            char *msg = (char*) data;

            // Add the newline delimiter to the msg.
            {
                int strsize = scope_strlen(msg);
                char* temp = scope_realloc(msg, strsize+2); // room for "\n\0"
                if (!temp) {
                    DBG(NULL);
                    scope_free(msg);
                    msg = NULL;
                    continue;
                }
                msg = temp;
                msg[strsize] = '\n';
                msg[strsize+1] = '\0';
            }
            transportSend(ctl->transport, msg, scope_strlen(msg));
            scope_free(msg);
        }
    }
}

static void
sendAggregatedLogData(ctl_t *ctl, streambuf_t *stmbuf)
{
    // Create json for log/console event and run the regex valuefilter
    cJSON *json = createLogEventJson(ctl, stmbuf);
    if (!json) return;

    // Create message
    upload_t upld;
    upld.type = UPLD_EVT;
    upld.body = json;
    upld.req = NULL;
    upld.proc = stmbuf->id.proc;
    upld.uid = stmbuf->id.uid;
    char *msg = prepMessage(&upld);
    if (!msg) return;

    // Send it.
    transportSend(ctl->transport, msg, scope_strlen(msg));
    scope_free(msg);
}

static void
ctlSendAllAggregatedLogData(ctl_t *ctl)
{
    if (!ctl) return;
    static unsigned long long count = 0;

    // If our process is exiting or this ctl is going away, report all now.
    // Otherwise, send the data once out of every flush_period_in_ms.
    // (Note that this code assumes it's called once per ms.)
    int report_now = ctlProcessAllQueuedEventsNow(ctl);
    report_now |= !(count++ % ctl->log.flush_period_in_ms);
    if (!report_now) return;

    int i;
    for (i=0; i<FS_ENTRIES; i++) {
        streambuf_t *stmbuf = &ctl->log.streamAgg[i];
        if (stmbuf->stream) {
            sendAggregatedLogData(ctl, stmbuf);
        }
    }
}

void
ctlFlushLog(ctl_t *ctl)
{
    if (!ctl) return;

    // aggregate the data queued by ctlSendLog
    uint64_t data;
    while (cbufGet(ctl->log.ringbuf, &data) == 0) {
        if (!data) continue;
        log_event_t *event = (log_event_t*) data;

        if ((event->fd >= 0) && (event->fd < FS_ENTRIES)) {

            streambuf_t *stmbuf = &ctl->log.streamAgg[event->fd];

            // See if something new is on the same FD or
            // if adding this event would exceed our stream buffer data limit.
            // In either of these cases, send what we have so far.
            // The act of sending the data closes the stream buffer.
            if (stmbuf->stream &&
                 ((stmbuf->id.uid != event->id.uid) ||
                 (stmbuf->tot_size + event->datalen > ctl->log.max_agg_bytes))) {
                sendAggregatedLogData(ctl, stmbuf);
            }

            // Open a new stream buffer if needed
            if (!stmbuf->stream) {
                stmbuf->buf = NULL;
                stmbuf->bufsize = 0;
                stmbuf->tot_size = 0;
                stmbuf->stream = scope_open_memstream(&stmbuf->buf, &stmbuf->bufsize);
                if (!stmbuf->stream) {
                    DBG("log buffer create error for fd %d, path %s", event->fd, event->id.path);
                } else {
                    stmbuf->id = event->id;
                    event->id.path = NULL; // Tranferring alloc'd path from event to stmbuf.
                }
            }

            // Append the current event data onto the stream buffer
            if (stmbuf->stream) {
                size_t actual = scope_fwrite(event->data, 1, event->datalen, stmbuf->stream);
                stmbuf->tot_size += actual;
                if (event->datalen != actual) {
                    DBG("log buffer write error for fd %d, path %s. tried to "
                        "buffer %zu, but only buffered %zu", event->fd, event->id.path,
                        event->datalen, actual);
                }
            }
        }

        destroyInternalLogEvent(&event);
    }
}

int
ctlSendBin(ctl_t *ctl, char *buf, size_t len)
{
    if (!ctl || !buf) return -1;

    if (ctl->paytrans) {
        return transportSend(ctl->paytrans, buf, len);
    }

    return transportSend(ctl->transport, buf, len);
}

void
ctlStopAggregating(ctl_t *ctl)
{
    if (!ctl) return;
    ctl->stop_aggregating = TRUE;
}

bool
ctlProcessAllQueuedEventsNow(ctl_t *ctl)
{
    if (!ctl) return FALSE;
    return ctl->stop_aggregating;
}

void
ctlFlush(ctl_t *ctl)
{
    if (!ctl) return;
    sendBufferedMessages(ctl);
    ctlSendAllAggregatedLogData(ctl);
    transportFlush(ctl->transport);
    transportFlush(ctl->paytrans);
}

int
ctlNeedsConnection(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return (who == CFG_LS) ?
        transportNeedsConnection(ctl->paytrans) :
        transportNeedsConnection(ctl->transport);
}

int
ctlConnection(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return (who == CFG_LS) ?
        transportConnection(ctl->paytrans) :
        transportConnection(ctl->transport);
}

int
ctlConnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return (who == CFG_LS) ?
        transportConnect(ctl->paytrans) :
        transportConnect(ctl->transport);
}

int
ctlDisconnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return (who == CFG_LS) ?
        transportDisconnect(ctl->paytrans) :
        transportDisconnect(ctl->transport);
}

int
ctlReconnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return (who == CFG_LS) ?
        transportReconnect(ctl->paytrans) :
        transportReconnect(ctl->transport);
}

void
ctlTransportSet(ctl_t *ctl, transport_t *transport, which_transport_t who)
{
    if (!ctl) return;

    if (who == CFG_LS) {
        transportDestroy(&ctl->paytrans);
        ctl->paytrans = transport;
    } else {
        // Don't leak if ctlTransportSet is called repeatedly
        transportDestroy(&ctl->transport);
        ctl->transport = transport;
    }
}

transport_t *
ctlTransport(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return NULL;

    return (who == CFG_LS) ? ctl->paytrans : ctl->transport;
}

void
ctlEvtSet(ctl_t *ctl, evt_fmt_t *evt)
{
    if (!ctl) return;

    // Don't leak if ctlEvtSet is called repeatedly
    evtFormatDestroy(&ctl->evt);
    ctl->evt = evt;
}

evt_fmt_t *
ctlEvtGet(ctl_t *ctl)
{
    return ctl ? ctl->evt : NULL;
}

transport_status_t
ctlPayloadConnectionStatus(ctl_t *ctl) {
    // retrieve the information about source 
    payload_status_t payStatus = ctlPayStatus(ctl);
    transport_status_t status = {
        .configString = NULL,
        .isConnected = FALSE,
        .connectAttemptCount = 0,
        .failureString = NULL};

    switch (payStatus) {
        case PAYLOAD_STATUS_DISABLE:
            return status;
        case PAYLOAD_STATUS_CRIBL:
            return transportConnectionStatus(ctl->paytrans);
        case PAYLOAD_STATUS_CTL:
            return transportConnectionStatus(ctl->transport);
        case PAYLOAD_STATUS_DISK:
            status.configString = ctl->payload.dirRepr;
            status.isConnected = TRUE;
            return status;
        default:
            DBG(NULL);
            return status;
    }
}

transport_status_t
ctlConnectionStatus(ctl_t *ctl) {
    return transportConnectionStatus(ctl->transport);
}

bool
ctlEvtSourceEnabled(ctl_t *ctl, watch_t src)
{
    if (src < CFG_SRC_MAX) {
        if (ctl && ctl->evt) return evtFormatSourceEnabled(ctl->evt, src);
        return srcEnabledDefault[src];
    }

    DBG("%d", src);
    return srcEnabledDefault[CFG_SRC_FILE];
}

unsigned
ctlEnhanceFs(ctl_t *ctl)
{
    return (ctl) ? ctl->enhancefs : DEFAULT_ENHANCE_FS;
}

void
ctlEnhanceFsSet(ctl_t *ctl, unsigned val)
{
    if (!ctl) return;
    ctl->enhancefs = val;
}

payload_status_t
ctlPayStatus(ctl_t *ctl)
{
    return (ctl) ? ctl->payload.status : PAYLOAD_STATUS_DISABLE;
}

void
ctlPayStatusSet(ctl_t *ctl, payload_status_t val)
{
    if (!ctl) return;
    ctl->payload.status = val;
}

const char *
ctlPayDir(ctl_t *ctl)
{
    return (ctl) ? ctl->payload.dir : DEFAULT_PAYLOAD_DIR;
}

void
ctlPayDirSet(ctl_t *ctl, const char *dir)
{
    if (!ctl) return;
    if (ctl->payload.dir) {
        scope_free(ctl->payload.dir);
    }
    if (ctl->payload.dirRepr) {
        scope_free(ctl->payload.dirRepr);
    }
    if (!dir || (dir[0] == '\0')) {
        ctl->payload.dir = (DEFAULT_PAYLOAD_DIR) ? scope_strdup(DEFAULT_PAYLOAD_DIR) : NULL;
        ctl->payload.dirRepr = NULL;
        if (DEFAULT_PAYLOAD_DIR_REPR) {
            if (scope_asprintf(&ctl->payload.dirRepr, "dir://%s", ctl->payload.dir) < 0) {
                ctl->payload.dirRepr = NULL;
            }
        }
        return;
    }

    ctl->payload.dir = scope_strdup(dir);
    if (scope_asprintf(&ctl->payload.dirRepr, "dir://%s", ctl->payload.dir) < 0) {
        ctl->payload.dirRepr = NULL;
    }
}

void
ctlAllowBinaryConsoleSet(ctl_t *ctl, unsigned val)
{
    if (!ctl || val < 0 || val > 1) return;
    ctl->allow_binary_console = val;
}


uint64_t
ctlGetEvent(ctl_t *ctl)
{
    uint64_t data;

    if (cbufGet(ctl->events, &data) == 0) {
        return data;
    } else {
        return (uint64_t)-1;
    }
}

bool
ctlCbufEmpty(ctl_t *ctl)
{
    return cbufEmpty(ctl->log.ringbuf);
}

int
ctlPostPayload(ctl_t *ctl, char *pay)
{
    if (!pay || !ctl) return -1;

    if ((ctl->payload.ringbuf) &&
        (cbufPut(ctl->payload.ringbuf, (uint64_t)pay) == -1)) {
        // Full; drop and ignore
        DBG(NULL);
        return -1;
    }
    return 0;
}

uint64_t
ctlGetPayload(ctl_t *ctl)
{
    uint64_t data;

    if ((ctl->payload.ringbuf) &&
        (cbufGet(ctl->payload.ringbuf, &data) == 0)) {
        return data;
    } else {
        return (uint64_t)-1;
    }
}

