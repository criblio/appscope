#define _GNU_SOURCE

#include "com.h"
#include "ipc_resp.h"
#include "scopestdlib.h"
#include "cJSON.h"
#include "runtimecfg.h"

extern void doAndReplaceConfig(void *);

// Wrapper for scope message response
struct scopeRespWrapper{
    cJSON *resp;        // Scope mesage response
    void *priv;         // Additional resource allocated to create response
};

/*
 * Creates the scope response wrapper object
 */
static scopeRespWrapper *
respWrapperCreate(void) {
    return scope_calloc(1, sizeof(scopeRespWrapper)); 
}

/*
 * Destroys the scope response wrapper object
 */
void
ipcRespWrapperDestroy(scopeRespWrapper *wrap) {
    if (wrap->resp) {
        cJSON_free(wrap->resp);
    }
    if (wrap->priv) {
        cJSON_free(wrap->priv);
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
 * Creates the wrapper for response to IPC_CMD_GET_SCOPE_STATUS
 * TODO: use unused attribute later
 */
scopeRespWrapper *
ipcRespGetScopeStatus(cJSON *unused) {
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
ipcRespGetScopeCfg(cJSON *unused) {
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
    wrap->priv = cfg;

    cJSON_AddItemToObjectCS(resp, "cfg", cfg);
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
ipcRespStatusNotImplemented(cJSON *unused) {
    return ipcRespStatus(IPC_RESP_NOT_IMPLEMENTED);
}

/*
 * Process the request IPC_CMD_SET_SCOPE_CFG
 */
static bool
ipcProcessSetCfg(cJSON *scopeReq) {
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
ipcRespSetScopeCfg(cJSON *scopeReq) {
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