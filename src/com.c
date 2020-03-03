#include "com.h"

int
cmdSendEvent(ctl_t *ctl, event_t* e, uint64_t time, proc_id_t* proc)
{
    return ctlSendEvent(ctl, e, time, proc);
}

int
cmdSendMetric(mtc_t *mtc, event_t* e)
{
    return mtcSendMetric(mtc, e);
}


int
cmdPostEvtMsg(ctl_t *ctl, cJSON *json)
{
    return ctlPostMsg(ctl, json, UPLD_EVT, NULL, FALSE);
}

int
cmdSendInfoStr(ctl_t *ctl, const char *str)
{
    cJSON* json;
    if (!str || !(json = cJSON_CreateString(str))) return -1;

    return ctlPostMsg(ctl, json, UPLD_INFO, NULL, TRUE);
}

int
cmdPostInfoMsg(ctl_t *ctl, cJSON *json)
{
    return ctlPostMsg(ctl, json, UPLD_INFO, NULL, FALSE);
}

int
cmdSendEvtMsg(ctl_t *ctl, cJSON *json)
{
    return ctlPostMsg(ctl, json, UPLD_EVT, NULL, TRUE);
}

int
cmdSendInfoMsg(ctl_t *ctl, cJSON *json)
{
    return ctlPostMsg(ctl, json, UPLD_INFO, NULL, TRUE);
}

int
cmdSendResponse(ctl_t *ctl, request_t *req, cJSON *body)
{
    return ctlPostMsg(ctl, body, UPLD_RESP, req, TRUE);
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
