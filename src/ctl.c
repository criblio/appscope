#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "circbuf.h"
#include "cfgutils.h"
#include "ctl.h"
#include "dbg.h"


typedef struct {
    const char* str;
    cmd_t cmd;
} cmd_map_t;

static cmd_map_t cmd_map[] = {
    {"SetCfg",           REQ_SET_CFG},
    {"GetCfg",           REQ_GET_CFG},
    {"GetDiag",          REQ_GET_DIAG},
    {"BlockPort",        REQ_BLOCK_PORT},
    {NULL,               REQ_UNKNOWN}
};


static void
grab_supplemental_for_block_port(cJSON* json_root, request_t* req)
{
    cJSON* json;

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
grab_supplemental_for_set_cfg(cJSON* json_root, request_t* req)
{
    cJSON* json;
    char* string = NULL;

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

request_t*
ctlParseRxMsg(const char* msg)
{
    cJSON* json_root = NULL;
    cJSON* json;
    char* str;
    cmd_map_t* map;

    request_t* req = calloc(1, sizeof(request_t));
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
        default:
            DBG("Unknown Cmd: %d", req->cmd);
    }

out:
    if (json_root) cJSON_Delete(json_root);
    return req;
}

void
destroyReq(request_t** request)
{
    if (!request || !*request) return;

    request_t* req = *request;

    if (req->cmd_str) free(req->cmd_str);
    if (req->cfg) cfgDestroy(&req->cfg);

    free(req);

    *request=NULL;
}

static cJSON*
create_info_json(upload_t* upld)
{
    cJSON* json_root = NULL;
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

static cJSON*
create_resp_json(upload_t* upld)
{
    cJSON* json_root = NULL;
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
    char* message = NULL;
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

static cJSON*
create_evt_json(upload_t* upld)
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

char*
ctlCreateTxMsg(upload_t* upld)
{
    cJSON* json = NULL;
    char* msg = NULL;

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




struct _ctl_t
{
    transport_t* transport;
    cbuf_handle_t evbuf;
};

ctl_t*
ctlCreate()
{
    ctl_t* ctl = calloc(1, sizeof(ctl_t));
    if (!ctl) {
        DBG(NULL);
        return NULL;
    }

    ctl->evbuf = cbufInit(DEFAULT_CBUF_SIZE);
    if (!ctl->evbuf) {
        DBG(NULL);
        return NULL;
    }

    return ctl;
}

void
ctlDestroy(ctl_t** ctl)
{
    if (!ctl || !*ctl) return;

    ctlFlush(*ctl);
    cbufFree((*ctl)->evbuf);

    transportDestroy(&(*ctl)->transport);

    free(*ctl);
    *ctl = NULL;
}

void
ctlSendMsg(ctl_t* ctl, char * msg)
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

static void
sendBufferedMessages(ctl_t* ctl)
{
    if (!ctl) return;

    uint64_t data;
    while (cbufGet(ctl->evbuf, &data) == 0) {
        if (data) {
            char *msg = (char*) data;
            transportSend(ctl->transport, msg);
            transportSend(ctl->transport, "\n");
            free(msg);
        }
    }
}

void
ctlFlush(ctl_t* ctl)
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
ctlClose(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportDisconnect(ctl->transport);
}

int
ctlConnect(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportConnect(ctl->transport);
}

void
ctlTransportSet(ctl_t* ctl, transport_t* transport)
{
    if (!ctl) return;

    // Don't leak if ctlTransportSet is called repeatedly
    transportDestroy(&ctl->transport);
    ctl->transport = transport;
}


