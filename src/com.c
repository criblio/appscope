#define _GNU_SOURCE
#include <string.h>

#include "com.h"
#include "dbg.h"
#include "utils.h"

bool g_need_stack_expand = FALSE;
unsigned g_sendprocessstart = 0;

// interfaces
mtc_t *g_mtc = NULL;
ctl_t *g_ctl = NULL;

// Add a newline delimiter to a msg
char *
msgAddNewLine(char *msg)
{
    if (!msg) return NULL;

    int strsize = strlen(msg);
    char *temp = realloc(msg, strsize + 2); // room for "\n\0"
    if (!temp) {
        DBG(NULL);
        free(msg);
        return NULL;
    }

    msg = temp;
    msg[strsize] = '\n';
    msg[strsize+1] = '\0';

    return msg;
}

void
sendProcessStartMetric()
{
    char *urlEncodedCmd = NULL;
    char *command = g_proc.cmd; // default is no encoding

    // If we're reporting in statsd format, url encode the cmd
    // to avoid characters that could cause parsing problems
    // for statsd aggregators  :|@#,
    if (cfgMtcFormat(g_cfg.staticfg) == CFG_FMT_STATSD) {
        urlEncodedCmd = fmtUrlEncode(g_proc.cmd);
        command = urlEncodedCmd;
    }

    event_field_t fields[] = {
        STRFIELD("proc", (g_proc.procname), 4, TRUE),
        NUMFIELD("pid", (g_proc.pid), 4, TRUE),
        STRFIELD("host", (g_proc.hostname), 4, TRUE),
        STRFIELD("args", (command), 7, TRUE),
        STRFIELD("unit", ("process"), 1, TRUE),
        FIELDEND
    };
    event_t evt = INT_EVENT("proc.start", 1, DELTA, fields);
    cmdSendMetric(g_mtc, &evt);
    if (urlEncodedCmd) free(urlEncodedCmd);
}

/*
 * This is called in 3 contexts/use cases
 * From the constructor
 * From the child on a fork, before return to the child from fork
 * From the periodic thread
 *
 * In all cases we send the json direct over the configured transport.
 * No in-memory buffer or delay. Given this context it should be safe
 * to send direct like this.
 */
void
reportProcessStart(ctl_t *ctl, bool init, which_transport_t who)
{
    // 1) Send a startup msg
    if (g_sendprocessstart && ((who == CFG_CTL) || (who == CFG_WHICH_MAX))) {
        cJSON *json = msgStart(&g_proc, g_cfg.staticfg, CFG_CTL);
        ctlSendJson(ctl, json, CFG_CTL);
    }

    // 2) send a payload start msg
    if (g_sendprocessstart && ((who == CFG_LS) || (who == CFG_WHICH_MAX)) &&
        cfgLogStream(g_cfg.staticfg)) {
        cJSON *json = msgStart(&g_proc, g_cfg.staticfg, CFG_LS);
        ctlSendJson(ctl, json, CFG_LS);
    }

    // only emit metric and log msgs at init time
    if (init) {
        // 3) Log it at startup, provided the loglevel is set to allow it
        scopeLog("Constructor (Scope Version: " SCOPE_VER ")", -1, CFG_LOG_INFO);
        char *cmd_w_args = NULL;
        if (asprintf(&cmd_w_args, "command w/args: %s", g_proc.cmd) != -1) {
            scopeLog(cmd_w_args, -1, CFG_LOG_INFO);
            if (cmd_w_args) free(cmd_w_args);
        }

        msgLogConfig(g_cfg.staticfg);

        // 4) Send a metric start; proc.start
        sendProcessStartMetric();
    }
}

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

void
msgLogConfig(config_t *cfg)
{
    cJSON *json;

    if (!(json = jsonConfigurationObject(cfg))) return;

    char *cfg_text = cJSON_PrintUnformatted(json);

    if (cfg_text) {
        scopeLog(cfg_text, -1, CFG_LOG_INFO);
        free(cfg_text);
    }

    cJSON_Delete(json);
}

cJSON *
msgStart(proc_id_t *proc, config_t *cfg, which_transport_t who)
{
    cJSON *json_root = NULL;
    cJSON *json_info;
    cJSON *json_proc, *json_cfg, *json_env;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (cfgAuthToken(cfg)) {
        if (!cJSON_AddStringToObjLN(json_root, "authToken", cfgAuthToken(cfg))) goto err;
    }

    if (who == CFG_LS) {
        if (!cJSON_AddStringToObjLN(json_root, "format", "scope")) goto err;
    } else {
        if (!cJSON_AddStringToObjLN(json_root, "format", "ndjson")) goto err;
        if (checkEnv("SCOPE_CRIBL_NO_BREAKER", "true")) {
                if (!cJSON_AddStringToObjLN(json_root, "breaker",
                                    "Cribl - Do Not Break Ruleset")) goto err;
        }
    }

    if (!(json_info = cJSON_AddObjectToObjLN(json_root, "info"))) goto err;

    if (!(json_proc = jsonProcessObject(proc))) goto err;
    cJSON_AddItemToObjectCS(json_info, "process", json_proc);

    if (!(json_cfg = jsonConfigurationObject(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_info, "configuration", json_cfg);

    if (!(json_env = jsonEnvironmentObject())) goto err;
    cJSON_AddItemToObjectCS(json_info, "environment", json_env);

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
    int rc, arc;
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
        : "=r"(arc)                  // output
        : "m"(tstack), "m"(gstack)   // input
        :                            // clobbered register
        );

    rc = pcre2_match(re, data, size, startoffset, options, match_data, mcontext);

    // Switch stack back to the original stack
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(arc)                       // output
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

