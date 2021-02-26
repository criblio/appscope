#define _GNU_SOURCE
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "circbuf.h"
#include "cfgutils.h"
#include "ctl.h"
#include "dbg.h"

struct _ctl_t
{
    transport_t *transport;
    transport_t *paytrans;
    evt_fmt_t *evt;
    cbuf_handle_t events;
    cbuf_handle_t evbuf;
    unsigned enhancefs;

    struct {
        unsigned int enable;
        char * dir;
        cbuf_handle_t ringbuf;
    } payload;
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
    {"redirect-on",      URL_REDIRECT_ON},
    {"redirect-off",     URL_REDIRECT_OFF},
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
    free(string);

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
        if (!strcmp(string, map->str)) {
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

    if ((prot = calloc(1, sizeof(protocol_def_t))) == NULL) goto err;

    req->protocol = prot;

    json = cJSON_GetObjectItem(body, "binary");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    if (strncmp("false", str, strlen(str)) == 0) {
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
    prot->regex = strdup(str);

    json = cJSON_GetObjectItem(body, "pname");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    prot->protname = strdup(str);

    return;

err:
    if (req) req->cmd=REQ_PARAM_ERR;
    if (prot && prot->regex) free(prot->regex);
    if (prot && prot->protname) free(prot->protname);
    if (prot) free(prot);
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

    if ((prot = calloc(1, sizeof(protocol_def_t))) == NULL) goto err;

    req->protocol = prot;

    json = cJSON_GetObjectItem(body, "pname");
    if (!json) goto err;
    if (!(str = cJSON_GetStringValue(json))) goto err;
    prot->protname = strdup(str);

    return;

err:
    if (req) req->cmd=REQ_PARAM_ERR;
    if (prot && prot->protname) free(prot->protname);
    if (prot) free(prot);
}

request_t *
ctlParseRxMsg(const char *msg)
{
    cJSON *json_root = NULL;
    cJSON *json;
    char *str;
    cmd_map_t *map;

    request_t *req = calloc(1, sizeof(request_t));
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
    // part of the json isn't useable for some reason.
    json = cJSON_GetObjectItem(json_root, "reqId");
    if (!json || !cJSON_IsNumber(json)) goto out;
    req->id = json->valuedouble;

    // grab req field
    json = cJSON_GetObjectItem(json_root, "req");
    if (!json) goto out;
    if (!(str = cJSON_GetStringValue(json))) goto out;
    req->cmd_str = strdup(str);

    // verify that type field exists, with required value "req"
    json = cJSON_GetObjectItem(json_root, "type");
    if (!json) goto out;
    if (!(str = cJSON_GetStringValue(json))) goto out;
    if (strcmp(str, "req")) goto out;

    //
    // phase 3, interpret what we grabbed
    //
    req->cmd = REQ_UNKNOWN;

    // make sure that the req field is a value we expect
    for (map=cmd_map; map->str; map++) {
        if (!strcmp(req->cmd_str, map->str)) {
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

    if (req->cmd_str) free(req->cmd_str);
    if (req->cfg) cfgDestroy(&req->cfg);
    // Note: don't mess with the protocol object here
    free(req);

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
    if (!upld || !upld->body) goto err;

    if (!(json_root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(json_root, "type", "evt")) goto err;
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
    int strsize = strlen(streamMsg);
    char *temp = realloc(streamMsg, strsize+2); // room for "\n\0"
    if (!temp) {
        DBG(NULL);
        scopeLog("CTL realloc error", -1, CFG_LOG_INFO);
        free(streamMsg);
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
ctlCreate()
{
    ctl_t *ctl = calloc(1, sizeof(ctl_t));
    if (!ctl) {
        DBG(NULL);
        return NULL;
    }

    ctl->evbuf = cbufInit(DEFAULT_CBUF_SIZE);
    if (!ctl->evbuf) {
        DBG(NULL);
        return NULL;
    }

    ctl->events = cbufInit(DEFAULT_CBUF_SIZE);
    if (!ctl->events) {
        DBG(NULL);
        return NULL;
    }

    ctl->enhancefs = DEFAULT_ENHANCE_FS;

    ctl->payload.enable = DEFAULT_PAYLOAD_ENABLE;
    ctl->payload.dir = (DEFAULT_PAYLOAD_DIR) ? strdup(DEFAULT_PAYLOAD_DIR) : NULL;
    ctl->payload.ringbuf = cbufInit(DEFAULT_PAYLOAD_RING_SIZE);
    if (!ctl->payload.ringbuf) {
        DBG(NULL);
        return NULL;
    }

    return ctl;
}

void
ctlDestroy(ctl_t **ctl)
{
    if (!ctl || !*ctl) return;

    ctlFlush(*ctl);
    cbufFree((*ctl)->evbuf);
    cbufFree((*ctl)->events);

    if ((*ctl)->payload.dir) free((*ctl)->payload.dir);
    cbufFree((*ctl)->payload.ringbuf);

    transportDestroy(&(*ctl)->transport);
    transportDestroy(&(*ctl)->paytrans);
    evtFormatDestroy(&(*ctl)->evt);

    free(*ctl);
    *ctl = NULL;
}

void
ctlSendMsg(ctl_t *ctl, char *msg)
{
if (!msg) return;
    if (!ctl) {
        free(msg);
        return;
    }

    if (cbufPut(ctl->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free(msg);
    }
}

void
ctlSendJson(ctl_t *ctl, cJSON *json)
{
    // This sends raw json (no envelope/messaging protocol)

    if (!json) return;
    if (!ctl) {
        cJSON_Delete(json);
        return;
    }

    char *streamMsg = cJSON_PrintUnformatted(json);
    cJSON_Delete(json);
    if (streamMsg) {
        // put it on the ring buffer and send it
        ctlSendMsg(ctl, streamMsg);
    }
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
    streamMsg = prepMessage(&upld);

    rc = transportSend(ctl->transport, streamMsg, strlen(streamMsg));
    if (streamMsg) free(streamMsg);
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
    streamMsg = prepMessage(&upld);

    rc = transportSend(ctl->transport, streamMsg, strlen(streamMsg));
    if (streamMsg) free(streamMsg);
    return rc;

}

int
ctlPostEvent(ctl_t *ctl, char *event)
{
    if (!event) return -1;
    if (!ctl) {
        free(event);
        return -1;
    }

    if (cbufPut(ctl->events, (uint64_t)event) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free(event);
        return -1;
    }
    return 0;
}

int
ctlSendLog(ctl_t *ctl, const char *path, const void *buf, size_t count, uint64_t uid, proc_id_t *proc)
{
    if (!ctl || !path || !buf || !proc) return -1;

    upload_t upld;

    // get a cJSON object for the given log msg
    cJSON *json = evtFormatLog(ctl->evt, path, buf, count, uid, proc);
    if (!json) return -1;

    upld.type = UPLD_EVT;
    upld.body = json;
    upld.req = NULL;
    char *msg = ctlCreateTxMsg(&upld);
    if (!msg) return -1;

    if (cbufPut(ctl->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        if (msg) free(msg);
        return -1;
    }

    return 0;
}

static void
sendBufferedMessages(ctl_t *ctl)
{
    if (!ctl) return;

    uint64_t data;
    while (cbufGet(ctl->evbuf, &data) == 0) {
        if (data) {
            char *msg = (char*) data;

            // Add the newline delimiter to the msg.
            {
                int strsize = strlen(msg);
                char* temp = realloc(msg, strsize+2); // room for "\n\0"
                if (!temp) {
                    DBG(NULL);
                    free(msg);
                    msg = NULL;
                    continue;
                }
                msg = temp;
                msg[strsize] = '\n';
                msg[strsize+1] = '\0';
            }
            transportSend(ctl->transport, msg, strlen(msg));
            free(msg);
        }
    }
}

void
ctlFlushLog(ctl_t *ctl)
{
    return sendBufferedMessages(ctl);
}

int
ctlSendBin(ctl_t *ctl, char *buf, size_t len)
{
    if (!ctl || !buf) return -1;

    if (ctl->paytrans) {
        if (ctlNeedsConnection(ctl, CFG_LS)) {
            // wait for a connection?
            // put the data back on the queue?
            // else, we drop packets until connected
            ctlConnect(ctl, CFG_LS);
        }

        return transportSend(ctl->paytrans, buf, len);
    }

    return transportSend(ctl->transport, buf, len);
}

bool
ctlHeaderMatch(ctl_t *ctl, const char *match)
{
    if (!ctl || !match) return FALSE;

    return evtHeaderMatch(ctl->evt, match);
}

void
ctlFlush(ctl_t *ctl)
{
    if (!ctl) return;

    sendBufferedMessages(ctl);
    transportFlush(ctl->transport);
    transportFlush(ctl->paytrans);
}

int
ctlNeedsConnection(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
        transportNeedsConnection(ctl->paytrans) :
        transportNeedsConnection(ctl->transport);
}

int
ctlConnection(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
        transportConnection(ctl->paytrans) :
        transportConnection(ctl->transport);
}

int
ctlConnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
        transportConnect(ctl->paytrans) :
        transportConnect(ctl->transport);
}

int
ctlDisconnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
        transportDisconnect(ctl->paytrans) :
        transportDisconnect(ctl->transport);
}

int
ctlReconnect(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return 0;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
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

cfg_transport_t
ctlTransportType(ctl_t *ctl, which_transport_t who)
{
    if (!ctl) return (cfg_transport_t)-1;

    return ((who == CFG_LS) && (ctl->paytrans)) ?
        transportType(ctl->paytrans) :
        transportType(ctl->transport);
}

void
ctlEvtSet(ctl_t *ctl, evt_fmt_t *evt)
{
    if (!ctl) return;

    // Don't leak if ctlEvtSet is called repeatedly
    evtFormatDestroy(&ctl->evt);
    ctl->evt = evt;
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

unsigned int
ctlPayEnable(ctl_t *ctl)
{
    return (ctl) ? ctl->payload.enable : DEFAULT_PAYLOAD_ENABLE;
}

void
ctlPayEnableSet(ctl_t *ctl, unsigned int val)
{
    if (!ctl) return;
    ctl->payload.enable = val;
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
    if (ctl->payload.dir) free(ctl->payload.dir);
    if (!dir || (dir[0] == '\0')) {
        ctl->payload.dir = (DEFAULT_PAYLOAD_DIR) ? strdup(DEFAULT_PAYLOAD_DIR) : NULL;
        return;
    }

    ctl->payload.dir = strdup(dir);
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
    return cbufEmpty(ctl->evbuf);
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

