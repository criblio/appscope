/*
 * ipcclient - IPC client
 *
 * A simple program to test communication between scoped process using 
 * gcc -g test/manual/ipcclient.c -lrt -o ipcclient
 * Connect to the scoped process
 * ./ipcclient -p <scoped_PID>
 * Connect to the scoped process and switch IPC namespace
 * ./ipcclient -p <scoped_PID> -i
 */

#define _GNU_SOURCE

#include <ctype.h>
#include <errno.h>
#include <linux/limits.h>
#include <mqueue.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

typedef unsigned int bool;
#define TRUE 1
#define FALSE 0

#define MAX_MESSAGES 10
#define MSG_BUFFER_SIZE 8192

static mqd_t readMqDesc;
static mqd_t writeMqDesc;
static char readerMqName[4096] = {0};
static char writerMqName[4096] = {0};

typedef struct {
    void* full;
    size_t fullLen;
} ipc_msg_t;


// Helper function to create message in format supported by IPC
static ipc_msg_t *
createIpcMessage(const char *metadata, const char *scopeData) {
    ipc_msg_t * msg = malloc(sizeof(ipc_msg_t));
    size_t metaDataLen = strlen(metadata) + 1;
    size_t scopeDataLen = strlen(scopeData) + 1;
    msg->fullLen = metaDataLen + scopeDataLen;
    msg->full = calloc(1, sizeof(char) * (msg->fullLen));
    strcpy(msg->full, metadata);
    strcpy(msg->full + metaDataLen, scopeData);
    return msg;
}

static void 
destroyIpcMessage(ipc_msg_t *ipcMsg) {
    free(ipcMsg->full);
    free(ipcMsg);
}

static ssize_t
msgBufGetNulIndx(const char* msgBuf, size_t msgLen) {
    for(size_t i = 0; i < msgLen; ++i) {
        if (!msgBuf[i]) {
            return i;
        }
    }
    return msgLen;
}

static bool
getLastPidNamesSpace(int pid, int *lastNsPid, bool *nestedNs) {
    char path[PATH_MAX] = {0};
    char buffer[4096];
    int tempNsPid = 0;
    int nsDepth = 0;

    if (snprintf(path, sizeof(path), "/proc/%d/status", pid) < 0) {
        perror("getLastPidNamesSpace: snprintf failed");
        return FALSE;
    }

    FILE *fstream = fopen(path, "r");
    if (fstream == NULL) {
        perror("getLastPidNamesSpace: fopen failed");
        return FALSE;
    }

    while (fgets(buffer, sizeof(buffer), fstream)) {
        if (strstr(buffer, "NSpid:")) {
            const char delimiters[] = ": \t";
            char *entry, *last;

            entry = strtok_r(buffer, delimiters, &last);
            // Skip NsPid string
            entry = strtok_r(NULL, delimiters, &last);
            // Iterate over NsPids values
            while (entry != NULL) {
                tempNsPid = atoi(entry);
                entry = strtok_r(NULL, delimiters, &last);
                nsDepth++;
            }
            break;
        }
    }

    if (nsDepth > 1) {
        *nestedNs = TRUE;
        *lastNsPid = tempNsPid;
    } else {
        *nestedNs = FALSE;
        *lastNsPid = pid;
    }

    fclose(fstream);

    return TRUE;
}

static bool
switchIPCNamespace(int pid) {
    char nsPath[PATH_MAX] = {0};
    int nsFd;
    if (snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/ipc", pid) < 0) {
        perror("setNamespace: snprintf failed");
        return FALSE;
    }

    if ((nsFd = open(nsPath, O_RDONLY)) == -1) {
        perror("setNamespace: open failed");
        return FALSE;
    }

    if (setns(nsFd, 0) != 0) {
        perror("setNamespace: setns failed");
        close(nsFd);
        return FALSE;
    }

    close(nsFd);

    return TRUE;
}

static void
cleanupMqDesc(void) {
    mq_close(writeMqDesc);
    mq_unlink(writerMqName);
    mq_close(readMqDesc);
    mq_unlink(readerMqName);
}

static int
printUsageAndExit(const char *cmd) {
    printf("Usage: %s -p <pid_scope_process> [-i]\n", cmd);
    printf("p - PID of scoped process\n");
    printf("i - switch IPC namespace during the communication\n");
    return EXIT_FAILURE;
}

static void
printMsgInfo(void) {
    printf("\nChoose message, type choice to stdin\n");
    printf("quit - stop sending\n");
    printf("cmds - get information about supported cmd\n");
    printf("status - get information about scope status\n");
    printf("core - trigger core dump\n");
}

static ipc_msg_t *
msgPrepareInfo(const char *inputBuf) {
    if(strcmp("quit\n", inputBuf) == 0) {
        printf("Quit message %s\n", inputBuf);
        return NULL;
    } else if(strcmp("cmds\n", inputBuf) == 0) {
        printf("cmd message\n");
        return createIpcMessage("{\"req\":0,\"uniq\":1234,\"remain\":128}", "{\"req\":0}");
    } else if(strcmp("status\n", inputBuf) == 0) {
        printf("status message\n");
        return createIpcMessage("{\"req\":0,\"uniq\":1234,\"remain\":128}", "{\"req\":1}");
    } else if(strcmp("core\n", inputBuf) == 0) {
        printf("status message\n");
        return createIpcMessage("{\"req\":0,\"uniq\":1234,\"remain\":128}", "{\"req\":5}");
    }

    printf("Unknown message %s\n", inputBuf);
    return NULL;
}

static char*
msgParse(const char *buf, size_t bufSize) {
    ssize_t indx = msgBufGetNulIndx(buf, bufSize);
    return strdup(buf + indx + 1);
}

int main(int argc, char **argv) {
    struct mq_attr attr = {.mq_flags = 0, 
                               .mq_maxmsg = MAX_MESSAGES,
                               .mq_msgsize = MSG_BUFFER_SIZE,
                               .mq_curmsgs = 0};
    mode_t oldMask;
    int res = EXIT_FAILURE;
    bool ipcSwitch = FALSE;
    bool nestedNs = FALSE;
    int pid = -1;
    int nsPid = -1;
    int c;

    while ((c = getopt(argc, argv, "ip:")) != -1) {
        switch (c) {
            case 'i':
                ipcSwitch = TRUE;
                break;
            case 'p':
                pid = atoi(optarg);
                break;
            case '?':
                if (optopt == 'p')
                fprintf(stderr, "Option -%d requires an argument.\n", optopt);
                else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                return res;
            default:
                return res;
        }
    }

    if (pid == -1) {
        return printUsageAndExit(argv[0]);
    }
    if (getLastPidNamesSpace(pid, &nsPid, &nestedNs) == FALSE) {
        return printUsageAndExit(argv[0]);
    }

    if (ipcSwitch) {
        if (!switchIPCNamespace(pid)) {
            return res;
        }
        snprintf(writerMqName, sizeof(writerMqName), "/ScopeIPCIn.%d", nsPid);
        snprintf(readerMqName, sizeof(readerMqName), "/ScopeIPCOut.%d", nsPid);
    } else {
        snprintf(writerMqName, sizeof(writerMqName), "/ScopeIPCIn.%d", pid);
        snprintf(readerMqName, sizeof(readerMqName), "/ScopeIPCOut.%d", pid);
    }

    atexit(cleanupMqDesc);

    // Ugly hack disable umask to handle run as a root
    oldMask = umask(0);
    writeMqDesc = mq_open(writerMqName, O_WRONLY | O_CREAT, 0666, &attr);
    if (writeMqDesc == (mqd_t)-1) {
        perror("!mq_open writeMqDesc failed");
        return res;
    }
    umask(oldMask);
    
    oldMask = umask(0);
    readMqDesc = mq_open(readerMqName, O_RDONLY | O_CREAT, 0666, &attr);
    if (readMqDesc == (mqd_t)-1) {
        perror("!mq_open readMqDesc failed");
        return res;
    }
    umask(oldMask);

    char InputBuf[MSG_BUFFER_SIZE] = {0};
    char RxBuf[MSG_BUFFER_SIZE] = {0};
    printMsgInfo();
    while (fgets(InputBuf, MSG_BUFFER_SIZE, stdin) != NULL) {

        size_t outputMsgSize;
        ipc_msg_t *msg = msgPrepareInfo(InputBuf);
        if (!msg) {
            break;
        }

        // Send message to scoped process
        if (mq_send(writeMqDesc, msg->full, msg->fullLen, 0) == -1) {
            perror("!mq_send writeMqDesc failed");
            goto end_iteration;
        }

        ssize_t recvLen = mq_receive(readMqDesc, RxBuf, MSG_BUFFER_SIZE, NULL);
        // Read response
        if (recvLen == -1) {
            perror("!mq_receive readMqDesc failed");
            goto end_iteration;
        }

        char *parsedMsg = msgParse(RxBuf, recvLen);

        if (ipcSwitch) {
            printf("Response from pid process %d: %s", pid, parsedMsg);
        } else {
            printf("Response from pid [%d parent process] [%d inside container] : %s", pid, nsPid, parsedMsg);
        }

        end_iteration:
            free(parsedMsg);
            destroyIpcMessage(msg);
            printMsgInfo();
            memset(InputBuf, 0, MSG_BUFFER_SIZE);
            memset(RxBuf, 0, MSG_BUFFER_SIZE);
    }
    return EXIT_SUCCESS;
}
