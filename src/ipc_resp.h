#ifndef __IPC_RESP_H__
#define __IPC_RESP_H__

#include "cJSON.h"

/*
 * Internal status of parsing the incoming message request, which can be:
 * - success after joining all the frames in the message
 * - error during parsing frames (processing will not wait for all frames)
 * - status is used both for ipc message and scope message
 */
typedef enum {
    REQ_PARSE_GENERIC_ERROR,            // Error: unknown error
    REQ_PARSE_ALLOCATION_ERROR,         // Error: memory allocation fails or empty queue
    REQ_PARSE_RECEIVE_ERROR,            // Error: during receive the message
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
scopeRespWrapper *ipcRespGetScopeStatus(cJSON *);
scopeRespWrapper *ipcRespGetScopeCfg(cJSON *);
scopeRespWrapper *ipcRespSetScopeCfg(cJSON *);
scopeRespWrapper *ipcRespStatusNotImplemented(cJSON *);
scopeRespWrapper *ipcRespStatusScopeError(ipc_resp_status_t);

// Wrapper destructor
void ipcRespWrapperDestroy(scopeRespWrapper *);

#endif // __IPC_RESP_H__
