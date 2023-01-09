#define _GNU_SOURCE

#include "ipc.h"
#include "ipc_resp.h"
#include "com.h"

#include "cJSON.h"
#include "dbg.h"
#include <errno.h>
#include <time.h>

/* Inter-process communication module based on the message-queue
 *
 * Message-queue system limits which are defined in following files:
 *
 * "/proc/sys/fs/mqueue/msg_max"
 * - describes maximum number of messsages in a queue
 *
 * "/proc/sys/fs/mqueue/msgsize_max"
 * - describes maximum message size in a queue
 *
 * "/proc/sys/fs/mqueue/queues_max"
 * - describes system-wide limit on the number of message queues that can be created
 *
 * See details in: https://man7.org/linux/man-pages/man7/mq_overview.7.html
 */

// Send message retry count
#define RETRY_COUNT 50
#define INPUT_MSG_ALLOC_LIMIT (1024*1024) // 1 MB
/*
 * meta_req_t describes the metadata part of ipc request command retrieves from IPC communication
 * IMPORTANT NOTE:
 * meta_req_t must be inline with client: metaReqCmd
 */

typedef enum {
    META_REQ_JSON,            // JSON request (complete)
    META_REQ_JSON_PARTIAL,    // JSON request (partial)
} meta_req_t;

/*
 * Translates the internal status of parsing request to the response output status
 */
static ipc_resp_status_t
translateParseStatusToResp(req_parse_status_t status) {
    switch (status) {
    case REQ_PARSE_OK:
        return IPC_RESP_OK;
    case REQ_PARSE_ALLOCATION_ERROR:
    case REQ_PARSE_RECEIVE_ERROR:
        return IPC_RESP_SERVER_ERROR;
    case REQ_PARSE_JSON_ERROR:
    case REQ_PARSE_REQ_ERROR:
    case REQ_PARSE_UNIQ_ERROR:
    case REQ_PARSE_SCOPE_REQ_ERROR:
    case REQ_PARSE_MISSING_SCOPE_DATA_ERROR:
    case REQ_PARSE_SCOPE_SIZE_ERROR:
        return IPC_BAD_REQUEST;
    default:
        UNREACHABLE();
        DBG(NULL);
        return IPC_RESP_SERVER_ERROR;
    }
}

/*
 * ipc_scope_req_t describes the scope request command retrieves from IPC communication
 * IMPORTANT NOTE:
 * ipc_scope_req_t must be inline with client: scopeReqCmd
 */
typedef enum {
    IPC_CMD_GET_SCOPE_STATUS, // Retrieves scope status of application (enabled or disabled)
    IPC_CMD_GET_SCOPE_CFG,    // Retrieves the current configuration
    IPC_CMD_SET_SCOPE_CFG,    // Update the current configuration
    IPC_CMD_UNKNOWN,          // Should be last - points to unsupported message
} ipc_scope_req_t;


typedef enum {
    MSG_RECV_RETRY_LIMIT,            // Error: receive: retry limit hit
    MSG_RECV_OTHER,                  // Error: receive: other
    MSG_RECV_OK,                     // Request was succesfully received
} ipc_receive_result;

/*
 * Retrieves IPC frame to specific message queue descriptor
 */
static ipc_receive_result
ipcReceiveFrameWithRetry(mqd_t mqDes, char *mqMsgBuf, size_t mqMaxMsgSize, ssize_t *mqMsgLen) {
    for (int retryCount = 1; retryCount <= RETRY_COUNT; ++retryCount) {
        *mqMsgLen = scope_mq_receive(mqDes, mqMsgBuf, mqMaxMsgSize, 0);
        if (*mqMsgLen != -1) {
            return MSG_RECV_OK;
        } else if(scope_errno == EAGAIN) {
            /*
            * Message queue is empty wait 50 us and retry since
            * the communication here is non-block
            */
            scope_nanosleep((const struct timespec[]){{0, 50000L}}, NULL);
        } else {
            /*
            * Other error
            */
            return MSG_RECV_OTHER;
        }
    }
    return MSG_RECV_RETRY_LIMIT;
}

/*
 * Sends IPC frame to specific message queue descriptor
 */
static ipc_resp_result_t
ipcSendFrameWithRetry(mqd_t mqDes, void *frame, size_t frameLen) {
    for (int retryCount = 1; retryCount <= RETRY_COUNT; ++retryCount) {
        int sendRes = scope_mq_send(mqDes, frame, frameLen, 0);
        if (sendRes == 0) {
            return RESP_RESULT_OK;
        } else if(scope_errno == EAGAIN) {
            /*
            * Message queue is full wait 50 us and retry since
            * the communication here is non-block
            */
            scope_nanosleep((const struct timespec[]){{0, 50000L}}, NULL);
        } else {
            /*
            * Other error
            */
            return RESP_SEND_OTHER;
        }
    }
    return RESP_SEND_RETRY_LIMIT;
}

/*
 * Opens the IPC connection for sending data as non-block IO
 */
mqd_t ipcOpenWriteConnection(const char *name) {
    return scope_mq_open(name, O_WRONLY | O_NONBLOCK);
}

/*
 * Open the IPC connection for receiving data as non-block IO
 */
mqd_t
ipcOpenReadConnection(const char *name) {
    return scope_mq_open(name, O_RDONLY | O_NONBLOCK);
}

/*
 * Closes the specific IPC connection
 */
int
ipcCloseConnection(mqd_t mqdes) {
    return scope_mq_close(mqdes);
}

/*
 * Checks if specific message queue is active,
 * Returns TRUE if Active, FALSE otherwise.
 * If message queue is active, function additionally
 * will returns the maximum message size and numbers of messages in the queue
 */
bool
ipcIsActive(mqd_t mqdes, size_t *maxMsgSize, long *msgCount) {
    struct mq_attr attr;
    if (mqdes == (mqd_t)-1) {
        return FALSE;
    }

    if (scope_mq_getattr(mqdes, &attr) == -1) {
        return FALSE;
    }

    *maxMsgSize = attr.mq_msgsize;
    *msgCount = attr.mq_curmsgs;
    return TRUE;
}

/*
 * Parse single frame placed in message queue.
 * Returns scope data from frame, the status of parsing the frame (parseStatus) and unique identifer of message request (uniqVal)
 */
static char *
ipcParseSingleFrame(const char *msgBuf, ssize_t msgLen, req_parse_status_t *parseStatus, int *uniqVal, size_t *scopeFrameLen, size_t *remainLen) {
    char *scopeMsg = NULL;

    // Verify if frame is based on JSON-format
    cJSON *msgJson = cJSON_Parse(msgBuf);
    if (!msgJson) {
        *parseStatus = REQ_PARSE_JSON_ERROR;
        goto end;
    }

    if (!cJSON_IsObject(msgJson)) {
        *parseStatus = REQ_PARSE_JSON_ERROR;
        goto cleanJson;
    }

    // Check the req in message queue frame
    cJSON *reqKey = cJSON_GetObjectItemCaseSensitive(msgJson, "req");
    if (!reqKey || !cJSON_IsNumber(reqKey)) {
        *parseStatus = REQ_PARSE_REQ_ERROR;
        goto cleanJson;
    }

    if (reqKey->valueint != META_REQ_JSON && reqKey->valueint != META_REQ_JSON_PARTIAL) {
        *parseStatus = REQ_PARSE_REQ_ERROR;
        goto cleanJson;
    }

    // Get the unique request id in message queue frame
    cJSON *uniqKey = cJSON_GetObjectItemCaseSensitive(msgJson, "uniq");
    if (!uniqKey || !cJSON_IsNumber(uniqKey)) {
        *parseStatus = REQ_PARSE_UNIQ_ERROR;
        goto cleanJson;
    }
    *uniqVal = uniqKey->valueint;

    // Get the remain data in the message queue frame
    cJSON *remainKey = cJSON_GetObjectItemCaseSensitive(msgJson, "remain");
    if (!remainKey || !cJSON_IsNumber(remainKey)) {
        *parseStatus = REQ_PARSE_REMAIN_ERROR;
        goto cleanJson;
    }

    // Compare remaining data with previouse frame 
    if (remainKey->valueint >= *remainLen) {
        *parseStatus = REQ_PARSE_REMAIN_ERROR;
        goto cleanJson;
    }
    *remainLen = remainKey->valueint;

    char *metaData = cJSON_PrintUnformatted(msgJson);
    size_t metaDataLen = scope_strlen(metaData);
    size_t dataOffset = metaDataLen + 1;
    // There is no scope data
    if (msgLen <= dataOffset) {
        *parseStatus = REQ_PARSE_MISSING_SCOPE_DATA_ERROR;
        goto cleanJson;
    }
    // Calculate the scope data length
    size_t dataLen = msgLen - dataOffset;
    // Allocate place for scope data in the current frame
    scopeMsg = scope_calloc(1, sizeof(char) * dataLen);
    if (!scopeMsg) {
        *parseStatus = REQ_PARSE_ALLOCATION_ERROR;
        goto cleanJson;
    }
    // Skip NUL char separator
    scope_memcpy(scopeMsg, msgBuf + dataOffset, dataLen);
    *parseStatus = REQ_PARSE_OK;
    *scopeFrameLen = dataLen;
    if (reqKey->valueint == META_REQ_JSON_PARTIAL ) {
        *parseStatus = REQ_PARSE_PARTIAL;
    } 

cleanJson:
    cJSON_Delete(msgJson);

end:
    return scopeMsg;
}

static char *
scopeMsgCreate(const char *frame, size_t frameLen) {
    char *msg = scope_calloc(1, frameLen * sizeof(char));
    if (!msg) {
        return NULL;
    }
    scope_memcpy(msg, frame, frameLen);

    return msg;
}

static char *
scopeMsgAppend(char *msg, size_t msgLen, const char *frame, size_t frameLen, req_parse_status_t *parseStatus) {
    // When we append we will overwrite the last byte
    size_t newMsgLen = msgLen - 1 + frameLen;
    if (newMsgLen > INPUT_MSG_ALLOC_LIMIT) {
        *parseStatus = REQ_PARSE_SCOPE_SIZE_ERROR;
        return NULL;
    }
    char *temp = scope_realloc(msg, newMsgLen * sizeof(char));
    if (!temp) {
        *parseStatus = REQ_PARSE_ALLOCATION_ERROR;
        return NULL;
    }
    // overwrite the NUL byte
    int lastMsgIndx = msgLen - 1;
    scope_memcpy(temp + lastMsgIndx, frame, frameLen);

    return temp;
}

/* 
 * ipcRequestHandler performs parsing of incoming frame in message queue
 * Returns scope msg
 */
char *
ipcRequestHandler(mqd_t mqDes, size_t mqMaxMsgSize, req_parse_status_t *parseStatus, int *uniqueReq) {
    char *frameRes = NULL;
    char *msgRes = NULL;
    size_t msgLen = 0;


    // Allocate maximum buffer for single meesage in message queue
    char *mqMsgBuf = scope_malloc(mqMaxMsgSize);
    if (!mqMsgBuf) {
        *parseStatus = REQ_PARSE_ALLOCATION_ERROR;
        return msgRes;
    }
    
    bool listenForResponseTransmission = TRUE;
    while (listenForResponseTransmission) {
        size_t frameLen = 0;
        size_t remainLen = SIZE_MAX;
        ssize_t mqMsgLen = -1;

        ipc_receive_result recvStatus = ipcReceiveFrameWithRetry(mqDes, mqMsgBuf, mqMaxMsgSize, &mqMsgLen);
        if (recvStatus != MSG_RECV_OK) {
            *parseStatus = REQ_PARSE_RECEIVE_ERROR;
            scope_free(mqMsgBuf);
            scope_free(msgRes);
            return NULL;
        }
        // Data from single frame
        frameRes = ipcParseSingleFrame(mqMsgBuf, mqMsgLen, parseStatus, uniqueReq, &frameLen, &remainLen);
        if (!frameRes) {
            scope_free(mqMsgBuf);
            scope_free(msgRes);
            return NULL;
        }

        // First frame 
        if (!msgRes) {
            msgRes = scopeMsgCreate(frameRes, frameLen);
            if (!msgRes) {
                *parseStatus = REQ_PARSE_ALLOCATION_ERROR;
                scope_free(frameRes);
                scope_free(mqMsgBuf);
                return NULL;
            }
            msgLen += frameLen;
        } else {
            char *temp = scopeMsgAppend(msgRes, msgLen, frameRes, frameLen, parseStatus);
            if (!temp) {
                scope_free(frameRes);
                scope_free(msgRes);
                scope_free(mqMsgBuf);
                return NULL;
            }
            msgRes = temp;
            msgLen = msgLen - 1 + frameLen;
        }
        scope_free(frameRes);

        if (*parseStatus != REQ_PARSE_PARTIAL) {
            listenForResponseTransmission = FALSE;
        }
    }
    scope_free(mqMsgBuf);

    return msgRes;
}



/*
 * Create metadata in message response
 */
static cJSON *
createMetaResp(ipc_resp_status_t status, int uniqReq, int remainLen) {
    cJSON *meta = cJSON_CreateObject();
    if (!meta) {
        return NULL;
    }
    if (!cJSON_AddNumberToObjLN(meta, "status", status)) {
        goto err;
    }

    if (!cJSON_AddNumberToObjLN(meta, "uniq", uniqReq)) {
        goto err;
    }

    if (!cJSON_AddNumberToObjLN(meta, "remain", remainLen)) {
        goto err;
    }

    return meta;

err:
    cJSON_Delete(meta);
    return NULL;
}

/*
 * Sends mq Response status, returns status of sending operation
 */
ipc_resp_result_t
ipcSendResponseOnly(mqd_t mqDes, size_t msgBufSize, req_parse_status_t parseStatus, int uniqReq) {    
    ipc_resp_result_t res = RESP_ALLOCATION_ERROR;
    cJSON *meta = createMetaResp(translateParseStatusToResp(parseStatus), uniqReq, 0);
    if (!meta) {
        return RESP_ALLOCATION_ERROR;
    }

    char *metadataBytes = cJSON_PrintUnformatted(meta);
    size_t metadataLen = scope_strlen(metadataBytes);
    // There is not sufficient place to use msg buffer 
    if (metadataLen >= msgBufSize) {
        res = RESP_UNSUFFICENT_MSGBUF_ERROR;
        goto end;
    }

    res = ipcSendFrameWithRetry(mqDes, metadataBytes, metadataLen);

end:
    cJSON_Delete(meta);
    return res;
}

typedef scopeRespWrapper *responseProcessor(cJSON *);

static responseProcessor *supportedResp[] = {
    [IPC_CMD_GET_SCOPE_STATUS] = ipcRespGetScopeStatus,
    [IPC_CMD_GET_SCOPE_CFG] = ipcRespGetScopeCfg, 
    [IPC_CMD_SET_SCOPE_CFG]  =  ipcRespSetScopeCfg,
    [IPC_CMD_UNKNOWN] = ipcRespStatusNotImplemented
};

/* 
 * ipcProcessRequestAndPrepareResponse
 * - parse the scope request
 * - prepare the response
 * It will create response based on:
 * - scopeReq scope request
 * Returns scope wrapper which contains the scope message respnose
 */
static scopeRespWrapper *
ipcProcessRequestAndPrepareResponse(const char *scopeReq) {

    req_parse_status_t status = REQ_PARSE_JSON_ERROR;

    // Verify if scope request is based on JSON-format
    cJSON *scopeReqJson = cJSON_Parse(scopeReq);
    if (!scopeReqJson) {
        goto errJson;
    }

    if (!cJSON_IsObject(scopeReqJson)) {
        goto errJson;
    }

    cJSON *cmdReq = cJSON_GetObjectItemCaseSensitive(scopeReqJson, "req");
    if (!cmdReq || !cJSON_IsNumber(cmdReq)) {
        status = REQ_PARSE_SCOPE_REQ_ERROR;
        goto errJson;
    }

    ipc_scope_req_t supportedCmd = IPC_CMD_UNKNOWN;
    for(supportedCmd = 0; supportedCmd < IPC_CMD_UNKNOWN; ++supportedCmd) {
        if (cmdReq->valueint == supportedCmd) {
            break;
        }
    }

    scopeRespWrapper *resp = supportedResp[supportedCmd](scopeReqJson);

    cJSON_Delete(scopeReqJson);

    return resp;

errJson:
    cJSON_Delete(scopeReqJson);

    return ipcRespStatusScopeError(translateParseStatusToResp(status));

}

/*
 * Sends IPC response with scope msg, returns status of sending operation
 */
ipc_resp_result_t
ipcSendResponseWithScopeData(mqd_t mqDes, size_t msgBufSize, const char *scopeDataReq, int uniqReq) {

    ipc_resp_result_t res = RESP_ALLOCATION_ERROR;
    // Proceed incoming scope request 
    scopeRespWrapper *scopeRespWrap = ipcProcessRequestAndPrepareResponse(scopeDataReq);
    if (!scopeRespWrap) {
        return res;
    }
    char *scopeRespBytes = ipcRespScopeRespStr(scopeRespWrap);
    size_t scopeDataRemainLen = scope_strlen(scopeRespBytes);
    size_t scopeDataOffset = 0;

    // Allocate buffer to send out
    void *frame = scope_malloc(msgBufSize * sizeof(char));
    if (!frame) {
        goto destroyWrap;
    }

    while (scopeDataRemainLen) {
        // Create basic metadata for response
        cJSON *metadataJson = createMetaResp(IPC_RESP_OK, uniqReq, scopeDataRemainLen);
        if (!metadataJson) {
            goto destroyFrame;
        }

        char *metadataBytes = cJSON_PrintUnformatted(metadataJson);
        size_t metadataLen = scope_strlen(metadataBytes) + 1;

        // There is not sufficient place to use msg buffer 
        if (metadataLen >= msgBufSize) {
            res = RESP_UNSUFFICENT_MSGBUF_ERROR;
            cJSON_Delete(metadataJson);
            goto destroyFrame;
        }
        // Calculate the scope data offset and length including NUL terminator byte
        size_t maxDataLen = msgBufSize - metadataLen;
        size_t dataSendLen = maxDataLen;
        if (scopeDataRemainLen < maxDataLen) {
            dataSendLen = scopeDataRemainLen;
        }
        scopeDataRemainLen -= dataSendLen;

        /*
        * If there is still remaining data we want to change status
        * from 200 -> 206, but because of how we construct the message
        * we cannot do this prior
        */
        if (scopeDataRemainLen != 0) {
            // Locate the status field
            char *statusStrs = scope_strstr(metadataBytes, "\"status\":200");
            // Get the offset for change 200 -> 206 minus 2 escape "\" characters
            size_t offset = sizeof("\"status\":200") - 2;
            statusStrs[offset] = '6';
        }

        scope_memcpy(frame, metadataBytes, metadataLen);
        cJSON_Delete(metadataJson);

        // Copy the scope frame data
        scope_memcpy(frame + metadataLen, scopeRespBytes + scopeDataOffset, dataSendLen);

        res = ipcSendFrameWithRetry(mqDes, frame, metadataLen + dataSendLen);
        if (res != RESP_RESULT_OK) {
            goto destroyFrame;
        }

        scopeDataOffset += dataSendLen;
    }

    res = RESP_RESULT_OK;

destroyFrame:
    scope_free(frame);

destroyWrap:
    ipcRespWrapperDestroy(scopeRespWrap);

    return res;
}

