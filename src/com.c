#define _GNU_SOURCE
#include "com.h"
#include "dbg.h"

extern rtconfig g_cfg;

bool g_need_stack_expand = FALSE;

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
cmdSendPayload(ctl_t *ctl, char *data, size_t len)
{
    if (!ctl || !data) return 0;
    return ctlSendBin(ctl, data, len);
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
cmdSendResponse(ctl_t *ctl, request_t *req, cJSON *body)
{
    return ctlPostMsg(ctl, body, UPLD_RESP, req, TRUE);
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
    cJSON *json_info;
    cJSON *json_proc, *json_cfg, *json_env;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(json_root, "format", "ndjson")) goto err;

    if (!(json_info = cJSON_AddObjectToObjLN(json_root, "info"))) goto err;

    if (!(json_proc = jsonProcessObject(proc))) goto err;
    cJSON_AddItemToObjectCS(json_info, "process", json_proc);

    if (!(json_cfg = jsonConfigurationObject(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_info, "configuration", json_cfg);

    if (!(json_env = jsonEnvironmentObject())) goto err;
    cJSON_AddItemToObjectCS(json_info, "environment", json_env);

    char *cfg_text = cJSON_PrintUnformatted(json_cfg);
    if (cfg_text) {
        scopeLog(cfg_text, -1, CFG_LOG_INFO);
        free(cfg_text);
    }
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

int
pcre2_match_wrapper(pcre2_code *re, PCRE2_SPTR data, PCRE2_SIZE size,
                    PCRE2_SIZE startoffset, uint32_t options,
                    pcre2_match_data *match_data, pcre2_match_context *mcontext)
{
    int rc;
    char *pcre_stack, *tstack, *gstack;

    if (g_need_stack_expand == FALSE) {
        return pcre2_match(re, data, size, startoffset, options, match_data, mcontext);
    }

    if ((pcre_stack = malloc(PCRE_STACK_SIZE)) == NULL) {
        scopeLog("ERROR; pcre2_match_wrapper: malloc", -1, CFG_LOG_ERROR);
        return -1;
    }

    tstack = pcre_stack + PCRE_STACK_SIZE;

    // save the original stack, switch to the tstack
    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        : "=r"(rc)                   // output
        : "m"(tstack), "m"(gstack)   // input
        :                            // clobbered register
        );

    rc = pcre2_match(re, data, size, startoffset, options, match_data, mcontext);

    // Switch stack back to the original stack
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(rc)                        // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    if (pcre_stack) free(pcre_stack);
    return rc;
}

int
regexec_wrapper(const regex_t *preg, const char *string, size_t nmatch,
                regmatch_t *pmatch, int eflags)
{
    int rc, arc;
    char *pcre_stack = NULL, *tstack = NULL, *gstack = NULL;

    if (g_need_stack_expand == FALSE) {
        return regexec(preg, string, nmatch, pmatch, eflags);
    }

    if ((pcre_stack = malloc(PCRE_STACK_SIZE)) == NULL) {
        scopeLog("ERROR; regexec_wrapper: malloc", -1, CFG_LOG_ERROR);
        return -1;
    }

    tstack = pcre_stack + PCRE_STACK_SIZE;

    // save the original stack, switch to the tstack
    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        : "=r"(arc)                   // output
        : "m"(tstack), "m"(gstack)   // input
        :                            // clobbered register
        );

    rc = regexec(preg, string, nmatch, pmatch, eflags);

    // Switch stack back to the original stack
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(arc)                        // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    if (pcre_stack) free(pcre_stack);
    return rc;
}

bool
cmdCbufEmpty(ctl_t *ctl)
{
    return ctlCbufEmpty(ctl);
}

int
cmdPostPayload(ctl_t *ctl, char *pay)
{
    return ctlPostPayload(ctl, pay);
}

uint64_t
msgPayloadGet(ctl_t *ctl)
{
    if (!ctl) return (uint64_t) -1;
    return ctlGetPayload(ctl);
}

