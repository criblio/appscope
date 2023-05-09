#define _GNU_SOURCE
#include <string.h>

#include "atomic.h"
#include "com.h"
#include "dbg.h"
#include "os.h"
#include "utils.h"
#include "scopestdlib.h"

unsigned g_sendprocessstart = 0;
bool g_exitdone = FALSE;

// interfaces
mtc_t *g_mtc = NULL;
ctl_t *g_ctl = NULL;

list_t *g_protlist;
unsigned int g_prot_sequence = 0;

// Add a newline delimiter to a msg
char *
msgAddNewLine(char *msg)
{
    if (!msg) return NULL;

    int strsize = scope_strlen(msg);
    char *temp = scope_realloc(msg, strsize + 2); // room for "\n\0"
    if (!temp) {
        DBG(NULL);
        scope_free(msg);
        return NULL;
    }

    msg = temp;
    msg[strsize] = '\n';
    msg[strsize+1] = '\0';

    return msg;
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
        cfgLogStreamEnable(g_cfg.staticfg)) {
        cJSON *json = msgStart(&g_proc, g_cfg.staticfg, CFG_LS);
        ctlSendJson(ctl, json, CFG_LS);
    }

    // only emit metric and log msgs at init time
    if (init) {
        // 3) Log it at startup, provided the loglevel is set to allow it
        scopeLogInfo("Constructor (Scope Version: " SCOPE_VER ")");
        scopeLogInfo("command w/args: %s", g_proc.cmd);

        msgLogConfig(g_cfg.staticfg);
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
    if (!(cJSON_AddNumberToObjLN(root, "gid", proc->gid))) goto err;
    if (proc->groupname) {
        if (!(cJSON_AddStringToObjLN(root, "groupname", proc->groupname))) goto err;
    }
    if (!(cJSON_AddNumberToObjLN(root, "uid", proc->uid))) goto err;
    if (proc->username) {
        if (!(cJSON_AddStringToObjLN(root, "username", proc->username))) goto err;
    }
    if (!(cJSON_AddStringToObjLN(root, "hostname", proc->hostname))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "procname", proc->procname))) goto err;
    if (proc->cmd) {
        if (!(cJSON_AddStringToObjLN(root, "cmd", proc->cmd))) goto err;
    }
    if (!(cJSON_AddStringToObjLN(root, "id", proc->id))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "cgroup", proc->cgroup))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "machine_id", proc->machine_id))) goto err;
    if (!(cJSON_AddStringToObjLN(root, "uuid", proc->uuid))) goto err;
 
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
    cJSON* root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    char *env_cribl_k8s_pod = getenv("CRIBL_K8S_POD");
    if (env_cribl_k8s_pod) {
        if (!cJSON_AddStringToObjLN(root, "CRIBL_K8S_POD",
                                        env_cribl_k8s_pod)) goto err;
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

void
msgLogConfig(config_t *cfg)
{
    cJSON *json;

    if (!(json = jsonConfigurationObject(cfg))) return;

    char *cfg_text = cJSON_PrintUnformatted(json);

    if (cfg_text) {
        scopeLogInfo("%s", cfg_text);
        scope_free(cfg_text);
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

// We saw a performance issue with malloc/free of memory
// used for stacks in pcre2_match_wrapper and regexec_wrapper.
// Every malloc was an mmap, every free an unmmap (both syscalls).
//
// To address this, have a lazily-allocated pool of stacks.
// Once allocated, never free stacks in this pool, instead just
// keep track of whether each stack is available for reuse.

// Why 48? Primarily to support 48 concurrent threads,
// but also to provide some slop. Slop for the unlikely case
// where a thread is killed asynchronously while the stack
// is marked as used.  When this happens we'll lose the
// ability to reuse that stack for the life of this process.

#define POOL_MAX 48
typedef struct {
    uint64_t used;
    char *addr;
} pool_t;

static pool_t g_stack_pool[POOL_MAX] = {0};

static bool
grab_unused(pool_t *entry)
{
    // Done atomically so two concurrent threads will never try
    // to use a stack at one time.
    return atomicCasU64(&entry->used, (uint64_t)FALSE, (uint64_t)TRUE);
}

static char *
get_stack(void)
{
    int i;
    for (i=0; i<POOL_MAX; i++) {
        pool_t *entry = &g_stack_pool[i];

        if (!entry->used && grab_unused(entry)) {

            // Sweet!  Our thread grabbed an allocated spot!
            if (entry->addr) return entry->addr;

            // We have a spot, but we need to allocate a stack here.
            entry->addr = scope_malloc(PCRE_STACK_SIZE);
            if (entry->addr) return entry->addr;

            // We got a spot, but our malloc failed. Put the spot back
            // into the unused pool. Stop looping if this happens.
            if (!atomicCasU64(&entry->used, (uint64_t)TRUE, (uint64_t)FALSE)) {
                 scopeLogError("get_stack failed to set used to FALSE");
            }
            break;
        }
    }

    // Our attempt to use the pool failed.
    // All pool entries were probably in use.
    // As a fall-back, do an allocation that's not from the pool
    return scope_malloc(PCRE_STACK_SIZE);
}

static void
free_stack(char *addr)
{
    int i;
    for (i=0; i<POOL_MAX; i++) {
        pool_t *entry = &g_stack_pool[i];

        if (entry->addr == addr) {

            // Addr is in the pool. Don't free addr, but set used to false
            // to allow reuse.
            if (!atomicCasU64(&entry->used, (uint64_t)TRUE, (uint64_t)FALSE)) {
                 scopeLogError("free_stack failed to set used to FALSE");
            }
            return;
        }
    }

    // We didn't find this addr in the pool.
    // It must not be from the pool.  Free it.
    scope_free(addr);
}


int
pcre2_match_wrapper(pcre2_code *re, PCRE2_SPTR data, PCRE2_SIZE size,
                    PCRE2_SIZE startoffset, uint32_t options,
                    pcre2_match_data *match_data, pcre2_match_context *mcontext)
{
    int rc;
    char *pcre_stack = NULL, *tstack = NULL, *gstack = NULL;
    if ((pcre_stack = get_stack()) == NULL) {
        scopeLogError("ERROR; pcre2_match_wrapper: get_stack");
        return -1;
    }

    tstack = pcre_stack + PCRE_STACK_SIZE;

    // save the original stack, switch to the tstack
#if defined (__x86_64__)
    int arc;

    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        : "=r"(arc)                     // output
        : "m"(tstack), "m"(gstack)      // input
        :                               // clobbered register
        );

    rc = pcre2_match(re, data, size, startoffset, options, match_data, mcontext);

    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(arc)                       // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );
#elif defined (__aarch64__)
    __asm__ volatile (
        "ldr  x0, %3 \n"                 // get params from the stack before switching
        "ldr  x1, %4 \n"
        "ldr  x2, %5 \n"
        "ldr  x3, %6 \n"
        "ldr  w4, %7 \n"
        "ldr  x5, %8 \n"
        "ldr  x6, %9 \n"
        "mov  x14, sp \n"
        "str  x14, %2 \n"
        "ldr  x15, %1 \n"
        "mov  sp, x15 \n"                // increase stack size

        "str  x14, [sp, #-16]! \n"

        // Note: the symbol name below may need to change if the pcre2 lib is updated
        "bl   pcre2_match_8 \n"          // call the regexec function
        "ldr x15, [sp, #0] \n"
        "mov  sp, x15 \n"
        "str  w0, %0 \n"                 // save the return value
        : "=&m"(rc)                      // output
        : "m"(tstack), "m"(gstack), "m" (re), "m" (data),"m" (size), "m" (startoffset), "m" (options), "m" (match_data), "m" (mcontext)
        :                               // clobbered register
        );
#else
   #error Bad arch defined
#endif

    if (pcre_stack) free_stack(pcre_stack);
    return rc;
}

int
regexec_wrapper(const regex_t *preg, const char *string, size_t nmatch,
                regmatch_t *pmatch, int eflags)
{
    int rc;
    char *pcre_stack = NULL, *tstack = NULL, *gstack = NULL;

     if ((pcre_stack = get_stack()) == NULL) {
        scopeLogError("ERROR; regexec_wrapper: get_stack");
        return -1;
    }

    tstack = pcre_stack + PCRE_STACK_SIZE;

    // save the original stack, switch to the tstack
#if defined (__x86_64__)
    int arc;

    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        : "=r"(arc)                     // output
        : "m"(tstack), "m"(gstack)      // input
        :                               // clobbered register
        );

    rc = regexec(preg, string, nmatch, pmatch, eflags);

    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(arc)                       // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );
#elif defined (__aarch64__)
    __asm__ volatile (
        "ldr  x0, %3 \n"                 // get params from the stack before switching
        "ldr  x1, %4 \n"
        "ldr  x2, %5 \n"
        "ldr  x3, %6 \n"
        "ldr  w4, %7 \n"
        "mov  x14, sp \n"
        "str  x14, %2 \n"
        "ldr  x15, %1 \n"
        "mov  sp, x15 \n"                // increase stack size

        "str  x14, [sp, #-16]! \n"

        "bl   pcre2_regexec \n"          // call the regexec function
        "ldr x15, [sp, #0] \n"
        "mov  sp, x15 \n"
        "str  w0, %0 \n"                 // save the return value
        : "=&m"(rc)                      // output
        : "m"(tstack), "m"(gstack), "m" (preg), "m" (string),"m" (nmatch), "m" (pmatch), "m" (eflags)
        :                               // clobbered register
        );
#else
   #error Bad arch defined
#endif

    if (pcre_stack) free_stack(pcre_stack);
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

