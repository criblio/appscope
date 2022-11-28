#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ipc.h"
#include "test.h"
#include "runtimecfg.h"

rtconfig g_cfg = {0};

static void
ipcInactiveDesc(void **state) {
    size_t mqSize;
    mqd_t mqdes = (mqd_t)-1;
    bool res = ipcIsActive(mqdes, &mqSize);
    assert_false(res);
}

static void
ipcInfoMsgCountNonExisting(void **state) {
    long msgCount = ipcInfoMsgCount((mqd_t)-1);
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
    long msgCount;
    struct mq_attr attr;
    void *buf;
    ssize_t dataLen;

    // Setup read-only IPC
    mqReadDes = ipcCreateNonBlockReadConnection(ipcConnName);
    assert_int_not_equal(mqReadDes, -1);
    res = ipcIsActive(mqReadDes, &mqReadSize);
    assert_true(res);
    msgCount = ipcInfoMsgCount(mqReadDes);
    assert_int_equal(msgCount, 0);

    // Read-only IPC verify that is impossible to send msg to IPC
    scope_errno = 0;
    status = scope_mq_send(mqReadDes, "test", sizeof("test"), 0);
    assert_int_equal(scope_errno, EBADF);
    assert_int_equal(status, -1);

    // Setup write-only IPC
    mqWriteDes = ipcOpenWriteConnection(ipcConnName);
    assert_int_not_equal(mqWriteDes, -1);
    res = ipcIsActive(mqWriteDes, &mqWriteSize);
    assert_true(res);

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

    msgCount = ipcInfoMsgCount(mqWriteDes);
    assert_int_equal(msgCount, 1);
    msgCount = ipcInfoMsgCount(mqReadDes);
    assert_int_equal(msgCount, 1);

    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    // Read-only IPC verify that it is possible to read msg from IPC
    dataLen = scope_mq_receive(mqReadDes, buf, attr.mq_msgsize, 0);
    assert_int_equal(dataLen, sizeof("test"));

    scope_free(buf);

    msgCount = ipcInfoMsgCount(mqWriteDes);
    assert_int_equal(msgCount, 0);
    msgCount = ipcInfoMsgCount(mqReadDes);
    assert_int_equal(msgCount, 0);

    // Teardown IPC(s)
    status = ipcCloseConnection(mqWriteDes);
    assert_int_equal(status, 0);

    status = ipcCloseConnection(mqReadDes);
    assert_int_equal(status, 0);
    status = ipcDestroyConnection(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestEmpty(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_cmd_t cmd;
    struct mq_attr attr;
    bool res;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    // Empty Message queue
    res = ipcRequestMsgHandler(mqReadWriteDes, attr.mq_msgsize, &cmd);
    assert_false(res);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerRequestValid(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    ipc_cmd_t cmd;
    struct mq_attr attr;
    bool res;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Put valid message on queue (cmdGetScopeStatus)
    status = scope_mq_send(mqReadWriteDes, "getScopeStatus", scope_strlen("getScopeStatus"), 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    res = ipcRequestMsgHandler(mqReadWriteDes, attr.mq_msgsize, &cmd);
    assert_true(res);
    assert_int_equal(cmd, IPC_CMD_GET_SCOPE_STATUS);

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
    ipc_cmd_t cmd;
    struct mq_attr attr;
    bool res;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    // Put dummy message on queue
    status = scope_mq_send(mqReadWriteDes, "loremIpsum", scope_strlen("loremIpsum"), 0);
    assert_int_equal(status, 0);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    res = ipcRequestMsgHandler(mqReadWriteDes, attr.mq_msgsize, &cmd);
    assert_true(res);
    assert_int_equal(cmd, IPC_CMD_UNKNOWN);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerResponseFail(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadDes;
    bool res;
    struct mq_attr attr;
    long msgCount;

    mqReadDes = scope_mq_open(ipcConnName, O_RDONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadDes, -1);

    status = scope_mq_getattr(mqReadDes, &attr);
    assert_int_equal(status, 0);

    res = ipcResponseMsgHandler(mqReadDes, attr.mq_msgsize, IPC_CMD_GET_SCOPE_STATUS);
    assert_false(res);
    msgCount = ipcInfoMsgCount(mqReadDes);
    assert_int_equal(msgCount, 0);

    status = scope_mq_close(mqReadDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}


static void
ipcHandlerResponseValid(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    bool res;
    long msgCount;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    res = ipcResponseMsgHandler(mqReadWriteDes, attr.mq_msgsize, IPC_CMD_GET_SCOPE_STATUS);
    assert_true(res);
    msgCount = ipcInfoMsgCount(mqReadWriteDes);
    assert_int_equal(msgCount, 1);


    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);

    assert_int_equal(dataLen, sizeof("false") - 1);

    //below should reflect cmdGetScopeStatus
    status = scope_memcmp(buf, "false", dataLen);
    assert_int_equal(status, 0);

    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

static void
ipcHandlerResponseUnknown(void **state) {
    const char *ipcConnName = "/testConnection";
    int status;
    mqd_t mqReadWriteDes;
    bool res;
    long msgCount;
    void *buf;
    struct mq_attr attr;
    ssize_t dataLen;

    mqReadWriteDes = scope_mq_open(ipcConnName, O_RDWR | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, NULL);
    assert_int_not_equal(mqReadWriteDes, -1);

    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    res = ipcResponseMsgHandler(mqReadWriteDes, attr.mq_msgsize, IPC_CMD_UNKNOWN);
    assert_true(res);
    msgCount = ipcInfoMsgCount(mqReadWriteDes);
    assert_int_equal(msgCount, 1);


    status = scope_mq_getattr(mqReadWriteDes, &attr);
    assert_int_equal(status, 0);

    buf = scope_malloc(attr.mq_msgsize);
    assert_non_null(buf);

    dataLen = scope_mq_receive(mqReadWriteDes, buf, attr.mq_msgsize, 0);
    assert_int_not_equal(dataLen, -1);
    assert_int_equal(dataLen, sizeof("Unknown") - 1);

    //below should reflect cmdUnknown
    status = scope_memcmp(buf, "Unknown", dataLen);
    assert_int_equal(status, 0);

    scope_free(buf);

    status = scope_mq_close(mqReadWriteDes);
    assert_int_equal(status, 0);
    status = scope_mq_unlink(ipcConnName);
    assert_int_equal(status, 0);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ipcInactiveDesc),
        cmocka_unit_test(ipcInfoMsgCountNonExisting),
        cmocka_unit_test(ipcOpenNonExistingConnection),
        cmocka_unit_test(ipcCommunicationTest),
        cmocka_unit_test(ipcHandlerRequestEmpty),
        cmocka_unit_test(ipcHandlerRequestValid),
        cmocka_unit_test(ipcHandlerRequestUnknown),
        cmocka_unit_test(ipcHandlerResponseFail),
        cmocka_unit_test(ipcHandlerResponseValid),
        cmocka_unit_test(ipcHandlerResponseUnknown),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
