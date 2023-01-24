#ifndef __IPC_RESP_H__
#define __IPC_RESP_H__

#include "cJSON.h"

/*
 * meta_req_t describes the metadata part of ipc request command retrieves from IPC communication
 * IMPORTANT NOTE:
 * meta_req_t must be inline with client: metaReqCmd
 * Please extend `cmdMetaName` structure in ipc_resp.c
 */
typedef enum {
    META_REQ_JSON,            // JSON request (complete)
    META_REQ_JSON_PARTIAL,    // JSON request (partial)
} meta_req_t;

/*
 * ipc_scope_req_t describes the scope request command retrieves from IPC communication
 * IMPORTANT NOTE:
 * ipc_scope_req_t must be inline with client: scopeReqCmd
 * NEW VALUES MUST BE PLACED AFTER LAST SUPPORTED CMD AND BEFORE IPC_CMD_UNKNOWN
 * Please extend `cmdScopeName` structure in ipc_resp.c
 */
typedef enum {
    IPC_CMD_GET_SUPPORTED_CMD,    // Retrieves the supported commands, introduced in: 1.3.0
    IPC_CMD_GET_SCOPE_STATUS,     // Retrieves scope status of application (enabled or disabled), introduced in: 1.3.0
    IPC_CMD_GET_SCOPE_CFG,        // Retrieves the current configuration, introduced in: 1.3.0
    IPC_CMD_SET_SCOPE_CFG,        // Update the current configuration, introduced in: 1.3.0
    IPC_CMD_GET_TRANSPORT_STATUS, // Retrieves the transport status, introduced in: 1.3.0
    IPC_CMD_CORE_DUMP,            // Trigger the core dump operation (testing)
    // Place to add new message
    IPC_CMD_UNKNOWN,              // MUST BE LAST - points to unsupported message
} ipc_scope_req_t;

/*
 * Internal status of parsing the incoming message request, which can be:
 * - success after joining all the frames in the message
 * - error during parsing frames (processing will not wait for all frames)
 * - status is used both for ipc message and scope message
 */
typedef enum {
    REQ_PARSE_GENERIC_ERROR,            // Error: unknown error
    REQ_PARSE_ALLOCATION_ERROR,         // Error: memory allocation fails or empty queue
    REQ_PARSE_RECEIVE_ERROR,            // Error: general error during receive the message
    REQ_PARSE_RECEIVE_TIMEOUT_ERROR,    // Error: timeout during receive the message
    REQ_PARSE_MISSING_SCOPE_DATA_ERROR, // Error: missing scope frame in request
    REQ_PARSE_JSON_ERROR,               // Error: request it not based on JSON format
    REQ_PARSE_REQ_ERROR,                // Error: msg frame issue with req field
    REQ_PARSE_UNIQ_ERROR,               // Error: msg frame issue with uniq field
    REQ_PARSE_REMAIN_ERROR,             // Error: msg frame issue with remain field
    REQ_PARSE_SCOPE_REQ_ERROR,          // Error: scope frame issue with uniq field
    REQ_PARSE_SCOPE_SIZE_ERROR,         // Error: scope frame issue with size
    REQ_PARSE_PARTIAL,                  // Request was succesfully parsed partial
    REQ_PARSE_OK,                       // Request was succesfully parsed
} req_parse_status_t;

// Forward declaration
typedef struct scopeRespWrapper scopeRespWrapper;

// This must be inline with respStatus in ipccmd.go
typedef enum {
    IPC_RESP_OK = 200,              // Response OK
    IPC_RESP_OK_PARTIAL_DATA = 206, // Response OK Partial Data
    IPC_BAD_REQUEST = 400,          // Invalid message syntax from client
    IPC_RESP_SERVER_ERROR = 500,    // Internal Server Error
    IPC_RESP_NOT_IMPLEMENTED = 501, // Method not implemented
} ipc_resp_status_t;

// String representation of scope response
char *ipcRespScopeRespStr(scopeRespWrapper *);

// Wrapper for Scope responses
scopeRespWrapper *ipcRespGetScopeCmds(const cJSON *);
scopeRespWrapper *ipcRespGetScopeStatus(const cJSON *);
scopeRespWrapper *ipcRespGetScopeCfg(const cJSON *);
scopeRespWrapper *ipcRespSetScopeCfg(const cJSON *);
scopeRespWrapper *ipcRespGetTransportStatus(const cJSON *);
scopeRespWrapper *ipcRespCoreDumpTrigger(const cJSON *);
scopeRespWrapper *ipcRespStatusNotImplemented(const cJSON *);
scopeRespWrapper *ipcRespStatusScopeError(ipc_resp_status_t);

// Wrapper destructor
void ipcRespWrapperDestroy(scopeRespWrapper *);

#endif // __IPC_RESP_H__
