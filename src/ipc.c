#define _GNU_SOURCE

#include "ipc.h"
#include "runtimecfg.h"

/* Inter-process communication module based on the message-queue
 *
 * Message-queue system limits which are related to `ipcOpenConnection`
 * are defined in following files:
 * 
 * "/proc/sys/fs/mqueue/msg_max"
 * - describes maximum number of messsages in a queue (QUEUE_MSG_MAX)
 *
 * "/proc/sys/fs/mqueue/msgsize_max"
 * - describes maximum message size in a queue (QUEUE_MSG_SIZE)
 * 
 * "/proc/sys/fs/mqueue/queues_max"
 * - describes system-wide limit on the number of message queues that can be created
 * 
 * See details in: https://man7.org/linux/man-pages/man7/mq_overview.7.html
 */

#define QUEUE_MSG_MAX  10
#define QUEUE_MSG_SIZE 8192

#define INPUT_CMD_LEN(x) (sizeof(x)-1)
#define CMD_TABLE_SIZE(x) (sizeof(x)/sizeof(x[0]))

/* Output message function definition
 * Fills up the input buffer return the size of message
 */
typedef size_t (*out_msg_func_t)(char *, size_t);

static size_t
cmdGetScopeStatus(char *buf, size_t len) {
    // Excluding the terminating null byte
    const char *status = (g_cfg.funcs_attached) ? "true" : "false";
    return scope_snprintf(buf, len, "%s", status);
}

static size_t
cmdUnknown(char *buf, size_t len) {
    // Excluding the terminating null byte
    return scope_snprintf(buf, len, "Unknown");
}

typedef struct {
    const char *name;       // command name string [in]
    size_t nameLen;         // command name string length [in]
    ipc_cmd_t cmd;          // command id [out]
} input_cmd_table_t;

typedef struct {
    ipc_cmd_t cmd;          // command id [in]
    out_msg_func_t func;    // output func [out]
} output_cmd_table_t;

static int
ipcSend(mqd_t mqdes, const char *data, size_t len) {
    return scope_mq_send(mqdes, data, len, 0);
}

static ssize_t
ipcRecv(mqd_t mqdes, char *buf, size_t len) {
    return scope_mq_receive(mqdes, buf, len, 0);
}

mqd_t
ipcCreateNonBlockReadConnection(const char *name) {
    struct mq_attr attr = {.mq_flags = 0, 
                               .mq_maxmsg = QUEUE_MSG_MAX,
                               .mq_msgsize = QUEUE_MSG_SIZE,
                               .mq_curmsgs = 0};
    return scope_mq_open(name, O_RDONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK, 0666, &attr);
}

mqd_t
ipcOpenWriteConnection(const char *name) {
    return scope_mq_open(name, O_WRONLY | O_NONBLOCK);
}

int
ipcCloseConnection(mqd_t mqdes) {
    return scope_mq_close(mqdes);
}

int
ipcDestroyConnection(const char *name) {
    return scope_mq_unlink(name); 
}

long
ipcInfoMsgCount(mqd_t mqdes) {
    struct mq_attr attr;
    int res = scope_mq_getattr(mqdes, &attr);
    if (res == 0) {
        return attr.mq_curmsgs;
    }
    return -1;
}

bool
ipcIsActive(mqd_t mqdes, size_t *msgSize) {
    struct mq_attr attr;
    if (mqdes == (mqd_t) -1) {
        return FALSE;
    }
    int res = scope_mq_getattr(mqdes, &attr);
    if (res == 0) {
        *msgSize = attr.mq_msgsize;
        return TRUE;
    }
    return FALSE;
}

bool
ipcRequestMsgHandler(mqd_t mqDes, size_t mqSize, ipc_cmd_t *cmdRes) {
    char *buf = scope_malloc(mqSize);
    if (!buf) {
        return FALSE;
    }

    ssize_t recvLen = ipcRecv(mqDes, buf, mqSize);
    if (recvLen == -1) {
        scope_free(buf);
        return FALSE;
    }

    input_cmd_table_t inputCmdTable[] = {
        {"getScopeStatus",  INPUT_CMD_LEN("getScopeStatus"), IPC_CMD_GET_SCOPE_STATUS}
    };

    for (int i = 0; i < CMD_TABLE_SIZE(inputCmdTable); ++i) {
        if ((recvLen == inputCmdTable[i].nameLen) && !scope_memcmp(inputCmdTable[i].name, buf, recvLen)) {
            *cmdRes = inputCmdTable[i].cmd;
            scope_free(buf);
            return TRUE;
        }
    }
    *cmdRes = IPC_CMD_UNKNOWN;
    scope_free(buf);
    return TRUE;
}

bool
ipcResponseMsgHandler(mqd_t mqDes, size_t mqSize, ipc_cmd_t cmd) {
    bool res = FALSE;
    size_t len;
    if ((unsigned)(cmd) > IPC_CMD_UNKNOWN) {
        return res;
    }

    char *buf = scope_malloc(mqSize);
    if (!buf) {
        return res;
    }

    output_cmd_table_t outputCmdTable[] = {
        {IPC_CMD_GET_SCOPE_STATUS,  cmdGetScopeStatus},
        {IPC_CMD_UNKNOWN,           cmdUnknown}
    };

    for (int i = 0; i < CMD_TABLE_SIZE(outputCmdTable); ++i) {
        if (cmd == outputCmdTable[i].cmd) {
            len = outputCmdTable[i].func(buf, mqSize);
            break;
        }
    }

    int sendRes = ipcSend(mqDes, buf, len);
    if (sendRes != -1) {
        res = TRUE;
    }
    scope_free(buf);

    return res;
}
