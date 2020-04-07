#include "com.h"

extern rtconfig g_cfg;

// for reporttest on mac __attribute__((weak))
/*
 * Note: there are 2 config values to check to see if
 * events; metrics, logs, console, http are enabled.
 * 1) the overall event enable/disable
 * 2) the individual source; metric, log, console, http
 *
 * The function ctlEvtSourceEnabled() tests for both conditions.
 * That is enabled in cfgutils.c:initEvtFormat()
 */
int
cmdSendEvent(ctl_t *ctl, event_t *event, uint64_t time, proc_id_t *proc)
{
    if (!ctlEvtSourceEnabled(ctl, CFG_SRC_METRIC)) return 0;
    return ctlSendEvent(ctl, event, time, proc);
}

int
cmdSendHttp(ctl_t *ctl, event_t *event, uint64_t time, proc_id_t *proc)
{
    if (!ctlEvtSourceEnabled(ctl, CFG_SRC_HTTP)) return 0;
    return ctlSendHttp(ctl, event, time, proc);
}

// for reporttest on mac __attribute__((weak))
int
cmdSendMetric(mtc_t *mtc, event_t *evt)
{
    if (!mtcEnabled(mtc)) return 0;
    return mtcSendMetric(mtc, evt);
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
cmdPostEvent(ctl_t *ctl, char *event)
{
    return ctlPostEvent(ctl, event);
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

int
cmdSendBin(ctl_t *ctl, char *buf, size_t len)
{
    return ctlSendBin(ctl, buf, len);
}

request_t *
cmdParse(const char *cmd)
{
    return ctlParseRxMsg(cmd);
}

static cJSON *
jsonProcessObject(proc_id_t *proc)
{
    cJSON *root = NULL;

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

cJSON *
jsonConfigurationObject(config_t *cfg)
{
    cJSON *root = NULL;
    cJSON *current;

    if (!cfg) goto err;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!(current = jsonObjectFromCfg(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, "current", current);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON *
jsonEnvironmentObject()
{
    return cJSON_CreateObject();
    // config file???
    // env variables???
}

cJSON *
msgStart(proc_id_t *proc, config_t *cfg)
{
    cJSON *json_root = NULL;
    cJSON *json_proc, *json_cfg, *json_env;

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

uint64_t
msgEventGet(ctl_t *ctl)
{
    if (!ctl) return (uint64_t) -1;
    return ctlGetEvent(ctl);
}

