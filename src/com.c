#include "com.h"


static int
postMsg(ctl_t *ctl, cJSON *body, upload_type_t type, request_t *req, bool now)
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
cmdPostEvtMsg(ctl_t *ctl, cJSON *json)
{
    return postMsg(ctl, json, UPLD_EVT, NULL, FALSE);
}

int
cmdSendInfoStr(ctl_t *ctl, const char *str)
{
    cJSON* json;
    if (!str || !(json = cJSON_CreateString(str))) return -1;

    return postMsg(ctl, json, UPLD_INFO, NULL, TRUE);
}

int
cmdPostInfoMsg(ctl_t *ctl, cJSON *json)
{
    return postMsg(ctl, json, UPLD_INFO, NULL, FALSE);
}

int
cmdSendEvtMsg(ctl_t *ctl, cJSON *json)
{
    return postMsg(ctl, json, UPLD_EVT, NULL, TRUE);
}

int
cmdSendInfoMsg(ctl_t *ctl, cJSON *json)
{
    return postMsg(ctl, json, UPLD_INFO, NULL, TRUE);
}

int
cmdSendResponse(ctl_t *ctl, request_t *req, cJSON *body)
{
    return postMsg(ctl, body, UPLD_RESP, req, TRUE);
}

request_t *
cmdParse(const char *cmd)
{
    return ctlParseRxMsg(cmd);
}

static cJSON*
jsonProcessObject(proc_id_t* proc)
{
    cJSON* root = NULL;

    if (!proc) goto err;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!(cJSON_AddStringToObjLN(root, "libscopever", SCOPE_VER))) goto err;

    if (!(cJSON_AddNumberToObjLN(root, "pid", proc->pid))) goto err;
    if (!(cJSON_AddNumberToObjLN(root, "ppid", proc->ppid))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "hostname", proc->hostname))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "procname", proc->procname))) goto err;
    if (proc->cmd) {
        if (!(cJSON_AddStringToObjLN(root, "cmd", proc->cmd))) goto err;
    }
    if (!(cJSON_AddStringToObjLN(root, "id", proc->id))) goto err;
    // starttime

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

cJSON*
jsonConfigurationObject(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* current;

    if (!cfg) goto err;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!(current = jsonObjectFromCfg(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, "current", current);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
jsonEnvironmentObject()
{
    return cJSON_CreateObject();
    // config file???
    // env variables???
}

cJSON*
msgStart(proc_id_t* proc, config_t* cfg)
{
    cJSON* json_root = NULL;
    cJSON* json_proc, *json_cfg, *json_env;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (!(json_proc = jsonProcessObject(proc))) goto err;
    cJSON_AddItemToObjectCS(json_root, "process", json_proc);

    if (!(json_cfg = jsonConfigurationObject(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, "configuration", json_cfg);

    if (!(json_env = jsonEnvironmentObject())) goto err;
    cJSON_AddItemToObjectCS(json_root, "environment", json_env);

    return json_root;
err:
    if (json_root) cJSON_Delete(json_root);
    return NULL;
}

cJSON *
msgEvtMetric(evt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc)
{
    return evtMetric(evt, metric, uid, proc);
}

cJSON *
msgEvtLog(evt_t *evt, const char *path, const void *buf, size_t len,
          uint64_t uid, proc_id_t *proc)
{
    return evtLog(evt, path, buf, len, uid, proc);

}

