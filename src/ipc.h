#ifndef __IPC_H__
#define __IPC_H__

#include "scopestdlib.h"
#include "scopetypes.h"
#include "ipc_resp.h"


// Manage Inter-process connection
mqd_t ipcOpenReadConnection(const char *);
mqd_t ipcOpenWriteConnection(const char *);
int ipcCloseConnection(mqd_t);
bool ipcIsActive(mqd_t, size_t *, long *);

/*
 * Internal status of sending the response
 */
typedef enum {
    RESP_RESULT_GENERIC_ERROR,        // Error: unknown error
    RESP_ALLOCATION_ERROR,            // Error: memory allocation error
    RESP_UNSUFFICENT_MSGBUF_ERROR,    // Error: unsufficient space in msgbuf
    RESP_SEND_RETRY_LIMIT,            // Error: sending: retry limit hit
    RESP_SEND_OTHER,                  // Error: sending: other
    RESP_RESULT_OK,                   // Response was succesfully sended
} ipc_resp_result_t;

/* IPC Request Handler
 *
 * Handler for message queue request
 * IMPORTANT NOTE:
 * Message queue request MUST be inline with client code definition: metaRequest
 * 
 * - req - number - message queue request command type
 * - uniq - number - unique identifier of message request
 * - remain - number - number of bytes data remaining (including present frame request)
 */
char *ipcRequestHandler(mqd_t, size_t, req_parse_status_t *, int *);

/* IPC Response Handlers
 *
 * Handler for message queue response
 * IMPORTANT NOTE:
 * Message queue response MUST be inline with client code definition: ipcResponse
 * 
 * - status - number - message queue response status
 * - uniq - number - unique identifier of message request
 * - remain - number - number of bytes data remaining (including present frame response)
 */
ipc_resp_result_t ipcSendResponseOnly(mqd_t, size_t, req_parse_status_t, int);

ipc_resp_result_t ipcSendResponseWithScopeData(mqd_t, size_t, const char *, int);

#endif // __IPC_H__
