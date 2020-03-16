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
    evt_fmt_t *evt;
    cbuf_handle_t evbuf;
    cbuf_handle_t events;
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
    cJSON* json_root = NULL;
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

    return ctl;
}

void
ctlDestroy(ctl_t **ctl)
{
    if (!ctl || !*ctl) return;

    ctlFlush(*ctl);
    cbufFree((*ctl)->evbuf);
    cbufFree((*ctl)->events);

    transportDestroy(&(*ctl)->transport);
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
ctlSendEvent(ctl_t *ctl, event_t *evt, uint64_t uid, proc_id_t *proc)
{
    if (!ctl || !evt || !proc) return -1;

    // get a cJSON object for the given event
    cJSON *json;
    if ((json = evtFormatMetric(ctl->evt, evt, uid, proc)) == NULL) return -1;

    // send it
    return ctlPostMsg(ctl, json, UPLD_EVT, NULL, FALSE);
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

    // get a cJSON object for the given log msg
    cJSON *json = evtFormatLog(ctl->evt, path, buf, count, uid, proc);

    // send it
    return ctlPostMsg(ctl, json, UPLD_EVT, NULL, FALSE);
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
            transportSend(ctl->transport, msg);
            free(msg);
        }
    }
}

void
ctlFlush(ctl_t *ctl)
{
    if (!ctl) return;
    sendBufferedMessages(ctl);
    transportFlush(ctl->transport);
}

int
ctlNeedsConnection(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportNeedsConnection(ctl->transport);
}

int
ctlConnection(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportConnection(ctl->transport);
}

int
ctlConnect(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportConnect(ctl->transport);
}

int
ctlClose(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportDisconnect(ctl->transport);
}

void
ctlTransportSet(ctl_t *ctl, transport_t *transport)
{
    if (!ctl) return;

    // Don't leak if ctlTransportSet is called repeatedly
    transportDestroy(&ctl->transport);
    ctl->transport = transport;
}

void
ctlEvtSet(ctl_t *ctl, evt_fmt_t *evt)
{
    if (!ctl) return;

    // Don't leak if ctlEvtSet is called repeatedly
    // TODO: need to ensure that previous object is no longer in use
    // evtFormatDestroy(&ctl->evt);
    ctl->evt = evt;
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
