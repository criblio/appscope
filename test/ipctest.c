#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfgutils.h"
#include "ipc.h"
#include "test.h"
#include "runtimecfg.h"
#include "cJSON.h"

static config_t * configTest;

// These signatures satisfy --wrap=jsonConfigurationObject in the Makefile
#ifdef __linux__
cJSON * __real_jsonConfigurationObject(config_t *cfg);
cJSON * __wrap_jsonConfigurationObject(config_t *cfg)
#endif // __linux__
#ifdef __APPLE__
cJSON * jsonConfigurationObject(config_t *cfg)
#endif // __APPLE__
{
    return (configTest) ? __real_jsonConfigurationObject(configTest) : NULL;
}

// These signatures satisfy --wrap=doAndReplaceConfig in the Makefile
#ifdef __linux__
void __real_doAndReplaceConfig(void *data);
void __wrap_doAndReplaceConfig(void *data)
#endif // __linux__
#ifdef __APPLE__
void doAndReplaceConfig(void *data)
#endif // __APPLE__
{
    return;
}

typedef struct {
    char* metadata;
    char* scopeData;
    void* full;
    size_t fullLen;
} ipc_msg_t;

// Helper functions to create the message sended via IPC

static ipc_msg_t *
createIpcMessage(const char *metadata, const char *scopeData) {
    ipc_msg_t * msg = scope_malloc(sizeof(ipc_msg_t));
    msg->metadata = scope_strdup(metadata);
    msg->scopeData = scope_strdup(scopeData);
    size_t metaDataLen = scope_strlen(metadata) + 1;
    size_t scopeDataLen = scope_strlen(scopeData) + 1;
    msg->fullLen = metaDataLen + scopeDataLen;
    msg->full = scope_calloc(1, sizeof(char) * (msg->fullLen));
    scope_strcpy(msg->full, msg->metadata);
    scope_strcpy(msg->full + metaDataLen, msg->scopeData);
    return msg;
}

static void 
destroyIpcMessage(ipc_msg_t *ipcMsg) {
    scope_free(ipcMsg->metadata);
    scope_free(ipcMsg->scopeData);
    scope_free(ipcMsg->full);
    scope_free(ipcMsg);
}

static void
ipcInactiveDesc(void **state) {
    size_t mqSize = 0;
    long msgCount = -1;
    mqd_t mqdes = (mqd_t)-1;
    bool res = ipcIsActive(mqdes, &mqSize, &msgCount);
    assert_false(res);
    assert_int_equal(msgCount, -1);
}

static void
ipcOpenNonExistingConnection(void **state) {
    mqd_t mqDes = ipcOpenWriteConnection("/NonExistingConnection");
    assert_int_equal(mqDes, -1);
}

static void
ipcCommunicationTest(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    bool res;
    mqd_t mqWriteDes, mqReadDes;
    size_t mqWriteSize, mqReadSize;
    long msgCount = -1;
    struct mq_attr attr;
    void *buf;
    ssize_t dataLen;

    // Setup read-only IPC
    mqReadDes = scope_mq_open(ipcConnName, O_RDONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadDes, -1);
    res = ipcIsActive(mqReadDes, &mqReadSize, &msgCount);
    assert_true(res);
    assert_int_equal(msgCount, 0);

    // Read-only IPC verify that is impossible to send msg to IPC
    scope_errno = 0;
    status = scope_mq_send(mqReadDes, "test", sizeof("test"), 0);
    assert_int_equal(scope_errno, EBADF);
    assert_int_equal(status, -1);

    // Setup write-only IPC
    mqWriteDes = ipcOpenWriteConnection(ipcConnName);
    assert_int_not_equal(mqWriteDes, -1);
    res = ipcIsActive(mqWriteDes, &mqWriteSize, &msgCount);
    assert_true(res);
    assert_int_equal(msgCount, 0);

    // Write-only IPC verify that it is possible to send msg to IPC
    status = scope_mq_send(mqWriteDes, "test", sizeof("test"), 0);
    assert_int_not_equal(status, -1);

    status = scope_mq_getattr(mqWriteDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    // Write-only IPC verify that is impossible to read msg from IPC
    scope_errno = 0;
    dataLen = scope_mq_receive(mqWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_equal(scope_errno, EBADF);
    assert_int_equal(dataLen, -1);

    scope_free(buf);

    status = scope_mq_getattr(mqWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    // Read-only IPC verify that it is possible to read msg from IPC
    dataLen = scope_mq_receive(mqReadDes, buf, attr.mq_msgsize, 0);
    assert_int_equal(dataLen, sizeof("test"));

    scope_free(buf);

    status = scope_mq_getattr(mqWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 0);
    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 0);

    // Closing IPC(s)
    status = ipcCloseConnection(mqWriteDes);
    assert_int_equal(status, 0);

    status = ipcCloseConnection(mqReadDes);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestEmptyQueue(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    struct mq_attr attr;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    // Empty Message queue
    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_RECEIVE_ERROR);
    assert_int_equal(uniqueId, -1);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestNotJson(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    struct mq_attr attr;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Not JSON message on queue
    char msg[] = "test";
    status = scope_mq_send(mqReadWriteDes, msg, sizeof(msg), 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_JSON_ERROR);
    assert_int_equal(uniqueId, -1);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestMissingReqField(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    struct mq_attr attr;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Empty JSON message on queue
    char msg[] = "{}";
    status = scope_mq_send(mqReadWriteDes, msg, sizeof(msg), 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_REQ_ERROR);
    assert_int_equal(uniqueId, -1);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestMissingUniqueField(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    struct mq_attr attr;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    ipc_msg_t *msg = createIpcMessage("{\"req\":0}", "{\"req\":1}");
    status = scope_mq_send(mqReadWriteDes, msg->full, msg->fullLen, 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_UNIQ_ERROR);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestMissingRemainField(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    struct mq_attr attr;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    ipc_msg_t *msg = createIpcMessage("{\"req\":0,\"uniq\":1234}", "{\"req\":1}");
    status = scope_mq_send(mqReadWriteDes, msg->full, msg->fullLen, 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_REMAIN_ERROR);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestKnown(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    char *scopeReq;
    struct mq_attr attr;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Get Scope Status message on queue
    ipc_msg_t *msg = createIpcMessage("{\"req\":0,\"uniq\":1234,\"remain\":128}", "{\"req\":1}");
    status = scope_mq_send(mqReadWriteDes, msg->full, msg->fullLen, 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_non_null(scopeReq);
    assert_int_equal(parseStatus, REQ_PARSE_OK);
    assert_string_equal(scopeReq, msg->scopeData);
    assert_int_equal(uniqueId, 1234);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestUnknown(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    req_parse_status_t parseStatus = REQ_PARSE_GENERIC_ERROR;
    int uniqueId = -1;
    struct mq_attr attr;
    char *scopeReq;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Send unknown message on queue
    ipc_msg_t *msg = createIpcMessage("{\"req\":0,\"uniq\":4567,\"remain\":128}", "{\"req\":99999}");
    status = scope_mq_send(mqReadWriteDes, msg->full, msg->fullLen, 0);
    assert_int_equal(status, 0);


    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    scopeReq = ipcRequestHandler(mqReadWriteDes, attr.mq_msgsize, &parseStatus, &uniqueId);
    assert_int_equal(parseStatus, REQ_PARSE_OK);
    assert_string_equal(scopeReq, msg->scopeData);
    assert_int_equal(uniqueId, 4567);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerResponseQueueTooSmallForResponse(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_resp_result_t res;
    int uniqueId = 9999;

    // Limit the parameter so small that it is too small to handle any response
    const long maxMsgSize = 3;
    struct mq_attr attr = {.mq_flags = 0, 
                           .mq_maxmsg = 10,
                           .mq_msgsize = maxMsgSize,
                           .mq_curmsgs = 0};


    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, &attr);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);


    ipc_msg_t *msg = createIpcMessage("{\"req\":0,\"uniq\":4567,\"remain\":128}", "{\"req\":1}");

    // Get Scope Status Scope message
    res = ipcSendResponseWithScopeData(mqReadWriteDes, attr.mq_msgsize, msg->scopeData, uniqueId);
    assert_int_equal(res, RESP_UNSUFFICENT_MSGBUF_ERROR);
    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 0);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerResponseFailToSend(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadDes;
    ipc_resp_result_t res;
    struct mq_attr attr;
    int uniqueId = 9999;

    mqReadDes = scope_mq_open(ipcConnName, O_RDONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadDes, -1);

    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);

    ipc_msg_t *msg = createIpcMessage("", "{\"req\":1}");

    // Test that message can not be sended because of lack write permissons
    res = ipcSendResponseWithScopeData(mqReadDes, attr.mq_msgsize, msg->scopeData, uniqueId);

    assert_int_equal(res, RESP_SEND_OTHER);
    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 0);

    destroyIpcMessage(msg);

    status = scope_mq_close(mqReadDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerResponseSendError(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_resp_result_t res;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;
    int uniqueId = 7777;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    res = ipcSendResponseOnly(mqReadWriteDes, attr.mq_msgsize, REQ_PARSE_SCOPE_REQ_ERROR, uniqueId);
    assert_int_equal(res, RESP_RESULT_OK);
    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);

    // Handle Single Mq Response (client side) no data is expected
    cJSON *item;
    cJSON *mqResp = cJSON_Parse(buf);
    assert_non_null(mqResp);
    item = cJSON_GetObjectItemCaseSensitive(mqResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, 400);
    item = cJSON_GetObjectItemCaseSensitive(mqResp, "uniq");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, uniqueId);

    cJSON_Delete(mqResp);

    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerScopeResponseValid(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_resp_result_t res;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;
    int uniqueId = 5656;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    ipc_msg_t *msg = createIpcMessage("", "{\"req\":0}");

    res = ipcSendResponseWithScopeData(mqReadWriteDes, attr.mq_msgsize, msg->scopeData, uniqueId);
    assert_int_equal(res, RESP_RESULT_OK);
    destroyIpcMessage(msg);
    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);

    // Handle Single Mq Response (client side)
    cJSON *item;
    cJSON *mqResp = cJSON_Parse(buf);
    assert_non_null(mqResp);
    char *mqRespBytes = cJSON_PrintUnformatted(mqResp);
    size_t mqRespLenWithNul = scope_strlen(mqRespBytes) + 1;

    item = cJSON_GetObjectItemCaseSensitive(mqResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, 200);
    item = cJSON_GetObjectItemCaseSensitive(mqResp, "uniq");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, uniqueId);

    // Skip Nul byte as begin 
    cJSON *scopeResp = cJSON_Parse(buf + mqRespLenWithNul);
    assert_non_null(scopeResp);
    item = cJSON_GetObjectItemCaseSensitive(scopeResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    //Response OK
    assert_int_equal(item->valueint, 200);
    item = cJSON_GetObjectItemCaseSensitive(scopeResp, "scoped");
    assert_non_null(item);
    assert_true(cJSON_IsBool(item));
    assert_true(cJSON_IsFalse(item));

    cJSON_Delete(scopeResp);
    cJSON_Delete(mqResp);

    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerScopeResponseUnknown(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_resp_result_t res;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;
    int uniqueId = 5678;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    ipc_msg_t *msg = createIpcMessage("", "{\"req\":9999}");

    res = ipcSendResponseWithScopeData(mqReadWriteDes, attr.mq_msgsize, msg->scopeData, uniqueId);
    assert_int_equal(res, RESP_RESULT_OK);

    destroyIpcMessage(msg);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);

    // Handle Single Mq Response (client side)
    cJSON *item;
    cJSON *mqResp = cJSON_Parse(buf);
    assert_non_null(mqResp);
    char *mqRespBytes = cJSON_PrintUnformatted(mqResp);
    size_t mqRespLenWithNul = scope_strlen(mqRespBytes) + 1;

    item = cJSON_GetObjectItemCaseSensitive(mqResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, 200);
    item = cJSON_GetObjectItemCaseSensitive(mqResp, "uniq");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, uniqueId);

    cJSON *scopeResp = cJSON_Parse(buf + mqRespLenWithNul);
    assert_non_null(scopeResp);
    item = cJSON_GetObjectItemCaseSensitive(scopeResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    //Method not implemented
    assert_int_equal(item->valueint, 501);
    cJSON_Delete(scopeResp);
    cJSON_Delete(mqResp);

    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerScopeResponseGetCfgSingleMsg(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_resp_result_t res;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;
    int uniqueId = 5678;
    config_t *cfgRecv = NULL;

    configTest = cfgCreateDefault();
    assert_non_null(configTest);
    assert_int_equal(cfgLogLevel(configTest), CFG_LOG_WARN);
    // Change default value of configuration
    cfgLogLevelSet(configTest, CFG_LOG_TRACE);
    assert_int_equal(cfgLogLevel(configTest), CFG_LOG_TRACE);

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    ipc_msg_t *msg = createIpcMessage("{\"req\":0,\"uniq\":1234,\"remain\":128}", "{\"req\":1}");
    res = ipcSendResponseWithScopeData(mqReadWriteDes, attr.mq_msgsize, msg->scopeData, uniqueId);
    assert_int_equal(res, RESP_RESULT_OK);

    destroyIpcMessage(msg);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);
    assert_int_equal(attr.mq_curmsgs, 1);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);

    // Handle Single Mq Response (client side)
    cJSON *item;
    cJSON *itemCurrentCfg;
    cJSON *mqResp = cJSON_Parse(buf);
    assert_non_null(mqResp);
    char *mqRespBytes = cJSON_PrintUnformatted(mqResp);
    size_t mqRespLenWithNul = scope_strlen(mqRespBytes) + 1;

    item = cJSON_GetObjectItemCaseSensitive(mqResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, 200);
    item = cJSON_GetObjectItemCaseSensitive(mqResp, "uniq");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, uniqueId);

    cJSON *scopeResp = cJSON_Parse(buf + mqRespLenWithNul);
    assert_non_null(scopeResp);
    item = cJSON_GetObjectItemCaseSensitive(scopeResp, "status");
    assert_non_null(item);
    assert_true(cJSON_IsNumber(item));
    assert_int_equal(item->valueint, 200);

    item = cJSON_GetObjectItemCaseSensitive(scopeResp, "cfg");
    assert_non_null(item);
    assert_true(cJSON_IsObject(item));

    itemCurrentCfg = cJSON_GetObjectItemCaseSensitive(item, "current");
    assert_non_null(itemCurrentCfg);
    assert_true(cJSON_IsObject(itemCurrentCfg));
    char *scopeCfgBytes = cJSON_PrintUnformatted(itemCurrentCfg);

    cfgRecv = cfgFromString(scopeCfgBytes);
    assert_int_equal(cfgLogLevel(cfgRecv), CFG_LOG_TRACE);

    cfgDestroy(&cfgRecv);

    cJSON_Delete(scopeResp);
    cJSON_Delete(mqResp);
    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);

    cfgDestroy(&configTest);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ipcInactiveDesc),
        cmocka_unit_test(ipcOpenNonExistingConnection),
        cmocka_unit_test(ipcCommunicationTest),
        cmocka_unit_test(ipcHandlerRequestEmptyQueue),
        cmocka_unit_test(ipcHandlerRequestNotJson),
        cmocka_unit_test(ipcHandlerRequestMissingReqField),
        cmocka_unit_test(ipcHandlerRequestMissingUniqueField),
        cmocka_unit_test(ipcHandlerRequestMissingRemainField),
        cmocka_unit_test(ipcHandlerRequestKnown),
        cmocka_unit_test(ipcHandlerRequestUnknown),
        cmocka_unit_test(ipcHandlerResponseQueueTooSmallForResponse),
        cmocka_unit_test(ipcHandlerResponseFailToSend),
        cmocka_unit_test(ipcHandlerResponseSendError),
        cmocka_unit_test(ipcHandlerScopeResponseValid),
        cmocka_unit_test(ipcHandlerScopeResponseUnknown),
        cmocka_unit_test(ipcHandlerScopeResponseGetCfgSingleMsg),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
