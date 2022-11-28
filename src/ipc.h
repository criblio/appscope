#ifndef __IPC_H__
#define __IPC_H__

#include "scopestdlib.h"
#include "scopetypes.h"


typedef enum {
    IPC_CMD_GET_SCOPE_STATUS = 0U, // Retrieve attach status
    IPC_CMD_UNKNOWN,               // Unknown command
} ipc_cmd_t;

// Manage Inter-process connection
mqd_t ipcCreateNonBlockReadConnection(const char *);
mqd_t ipcOpenWriteConnection(const char *);
int ipcCloseConnection(mqd_t);
int ipcDestroyConnection(const char *);
long ipcInfoMsgCount(mqd_t);
bool ipcIsActive(mqd_t, size_t *);

// IPC Request/Response Handler
bool ipcRequestMsgHandler(mqd_t, size_t, ipc_cmd_t *);
bool ipcResponseMsgHandler(mqd_t, size_t, ipc_cmd_t);

#endif // __IPC_H__
