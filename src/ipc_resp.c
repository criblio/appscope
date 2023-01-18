#define _GNU_SOURCE

#include "com.h"
#include "dbg.h"
#include "ipc_resp.h"
#include "scopestdlib.h"
#include "runtimecfg.h"

#define ARRAY_SIZE(arr) ((sizeof(arr))/(sizeof(arr[0])))

static const char* cmdMetaName[] = {
    [META_REQ_JSON]         = "completeRequestJson",
    [META_REQ_JSON_PARTIAL] = "incompleteRequestJson",
};

#define CMD_META_SIZE  (ARRAY_SIZE(cmdMetaName))

static const char* cmdScopeName[] = {
    [IPC_CMD_GET_SUPPORTED_CMD] = "getSupportedCmd",
    [IPC_CMD_GET_SCOPE_STATUS]  = "getScopeStatus",
    [IPC_CMD_GET_SCOPE_CFG]     = "getScopeCfg",
    [IPC_CMD_SET_SCOPE_CFG]     = "setScopeCfg",
};

#define CMD_SCOPE_SIZE  (ARRAY_SIZE(cmdScopeName))

extern void doAndReplaceConfig(void *);

#define WRAP_PRIV_SIZE (2)

// Wrapper for scope message response
struct scopeRespWrapper{
    cJSON *resp;                // Scope mesage response
    void *priv[WRAP_PRIV_SIZE]; // Additional resources allocated to create response
};

/*
 * Creates the scope response wrapper object
 */
static scopeRespWrapper *
respWrapperCreate(void) {
    scopeRespWrapper *wrap = scope_calloc(1, sizeof(scopeRespWrapper));
    if (!wrap) {
        return NULL;
    }
    wrap->resp = NULL;
    for (int i = 0; i < WRAP_PRIV_SIZE; ++i) {
        wrap->priv[i] = NULL;
    }
    return wrap;
}

/*
 * Destroys the scope response wrapper object
 */
void
ipcRespWrapperDestroy(scopeRespWrapper *wrap) {
    if (wrap->resp) {
        cJSON_free(wrap->resp);
    }
    for (int i = 0; i < WRAP_PRIV_SIZE; ++i) {
        if (wrap->priv[i]) {
            cJSON_free(wrap->priv[i]);
        }  
    }

    scope_free(wrap);
}

/*
 * Returns the scope message response string representation
 */
char *
ipcRespScopeRespStr(scopeRespWrapper *wrap) {
    return cJSON_PrintUnformatted(wrap->resp);
}

/*
 * Creates the wrapper for generic response (scope message and ipc message)
 * Used by following requests: IPC_CMD_UNKNOWN, IPC_CMD_SET_SCOPE_CFG
 */
scopeRespWrapper *
ipcRespStatus(ipc_resp_status_t status) {
    scopeRespWrapper *wrap = respWrapperCreate();
    if (!wrap) {
        return NULL;
    }
    cJSON *resp = cJSON_CreateObject();
    if (!resp) {
        goto allocFail;
    }
    wrap->resp = resp;
    if (!cJSON_AddNumberToObjLN(resp, "status", status)) {
        goto allocFail;
    }

    return wrap;

allocFail:
    ipcRespWrapperDestroy(wrap);
    return NULL; 
}

/*
 * Creates descriptor for meta and scope command used in IPC_CMD_GET_SUPPORTED_CMD
 */
static cJSON*
createCmdDesc(int id, const char *name) {
    cJSON *cmdDesc = cJSON_CreateObject();
    if (!cmdDesc) {
        return NULL;
    }

    if (!cJSON_AddNumberToObject(cmdDesc, "id", id)) {
        cJSON_free(cmdDesc);
        return NULL;
    }

    if (!cJSON_AddStringToObject(cmdDesc, "name", name)) {
        cJSON_free(cmdDesc);
        return NULL;
    }

    return cmdDesc;
}

/*
 * Creates the wrapper for response to IPC_CMD_GET_SUPPORTED_CMD
 * TODO: use unused attribute later
 */
scopeRespWrapper *
ipcRespGetScopeCmds(const cJSON * unused) {
    SCOPE_BUILD_ASSERT(IPC_CMD_UNKNOWN == CMD_SCOPE_SIZE, "cmdScopeName must be inline with ipc_scope_req_t");

    scopeRespWrapper *wrap = respWrapperCreate();
    if (!wrap) {
        return NULL;
    }
    cJSON *resp = cJSON_CreateObject();
    if (!resp) {
        goto allocFail;
    }
    wrap->resp = resp;
    if (!cJSON_AddNumberToObjLN(resp, "status", IPC_RESP_OK)) {
        goto allocFail;
    }

    cJSON *metaCmds = cJSON_CreateArray();
    if (!metaCmds) {
        goto allocFail;
    }

    wrap->priv[0] = metaCmds;
    for (int id = 0; id < CMD_META_SIZE; ++id){
        cJSON *singleCmd = createCmdDesc(id, cmdMetaName[id]);
        if (!singleCmd) {
            goto allocFail;
        }
        cJSON_AddItemToArray(metaCmds, singleCmd);
    }
    cJSON_AddItemToObjectCS(resp, "commands_meta", metaCmds);

    cJSON *scopeCmds = cJSON_CreateArray();
    if (!scopeCmds) {
        goto allocFail;
    }

    wrap->priv[1] = scopeCmds;
    for (int id = 0; id < CMD_SCOPE_SIZE; ++id){
        cJSON *singleCmd = createCmdDesc(id, cmdScopeName[id]);
        if (!singleCmd) {
            goto allocFail;
        }
        cJSON_AddItemToArray(scopeCmds, singleCmd);
    }
    cJSON_AddItemToObjectCS(resp, "commands_scope", scopeCmds);

    return wrap;

allocFail:
    ipcRespWrapperDestroy(wrap);
    return NULL; 
}

/*
 * Creates the wrapper for response to IPC_CMD_GET_SCOPE_STATUS
 * TODO: use unused attribute later
 */
scopeRespWrapper *
ipcRespGetScopeStatus(const cJSON *unused) {
    scopeRespWrapper *wrap = respWrapperCreate();
    if (!wrap) {
        return NULL;
    }
    cJSON *resp = cJSON_CreateObject();
    if (!resp) {
        goto allocFail;
    }
    wrap->resp = resp;
    if (!cJSON_AddNumberToObjLN(resp, "status", IPC_RESP_OK)) {
        goto allocFail;
    }
    if (!cJSON_AddBoolToObjLN(resp, "scoped", (g_cfg.funcs_attached))) {
        goto allocFail;
    }
    return wrap;

allocFail:
    ipcRespWrapperDestroy(wrap);
    return NULL; 
}

/*
 * Creates the wrapper for response to IPC_CMD_GET_SCOPE_CFG
 * TODO: use unused attribute later
 */
scopeRespWrapper *
ipcRespGetScopeCfg(const cJSON *unused) {
    scopeRespWrapper *wrap = respWrapperCreate();
    if (!wrap) {
        return NULL;
    }
    cJSON *resp = cJSON_CreateObject();
    if (!resp) {
        goto allocFail;
    }
    wrap->resp = resp;

    cJSON *cfg = jsonConfigurationObject(g_cfg.staticfg);
    if (!cfg) {
        if (!cJSON_AddNumberToObjLN(resp, "status", IPC_RESP_SERVER_ERROR)) {
            goto allocFail;
        }
        return wrap;
    }
    wrap->priv[0] = cfg;

    cJSON_AddItemToObjectCS(resp, "cfg", cfg);
    
    if (!cJSON_AddNumberToObjLN(resp, "status", IPC_RESP_OK)) {
        goto allocFail;
    }
    
    return wrap;

allocFail:
    ipcRespWrapperDestroy(wrap);
    return NULL;
}

/*
 * Creates the wrapper for response to IPC_CMD_UNKNOWN
 * TODO: use unused attribute later
 */
scopeRespWrapper *
ipcRespStatusNotImplemented(const cJSON *unused) {
    return ipcRespStatus(IPC_RESP_NOT_IMPLEMENTED);
}

/*
 * Process the request IPC_CMD_SET_SCOPE_CFG
 */
static bool
ipcProcessSetCfg(const cJSON *scopeReq) {
    bool res = FALSE;
    // Verify if scope request is based on JSON-format
    cJSON *cfgKey = cJSON_GetObjectItem(scopeReq, "cfg");
    if (!cfgKey || !cJSON_IsObject(cfgKey)) {
        return res;
    }
    char *cfgStr = cJSON_PrintUnformatted(cfgKey);
    config_t *cfg = cfgFromString(cfgStr);
    doAndReplaceConfig(cfg);
    res = TRUE;
    return res;
}

/*
 * Creates the wrapper for response to IPC_CMD_SET_SCOPE_CFG
 */
scopeRespWrapper *
ipcRespSetScopeCfg(const cJSON *scopeReq) {
    if (ipcProcessSetCfg(scopeReq)) {
        return ipcRespStatus(IPC_RESP_OK);
    }
    return ipcRespStatus(IPC_RESP_SERVER_ERROR);
}

/*
 * Creates the wrapper for failed case in processing scope msg
 */
scopeRespWrapper *
ipcRespStatusScopeError(ipc_resp_status_t status) {
    return ipcRespStatus(status);
}
