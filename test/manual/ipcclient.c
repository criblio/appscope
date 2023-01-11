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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_MESSAGES 10
#define MSG_BUFFER_SIZE 8192

static mqd_t readMqDesc;
static mqd_t writeMqDesc;
static char readerMqName[4096] = {0};
static char writerMqName[4096] = {0};

static bool
getLastPidNamesSpace(int pid, int *lastNsPid, bool *nestedNs) {
    char path[PATH_MAX] = {0};
    char buffer[4096];
    int tempNsPid = 0;
    int nsDepth = 0;

    if (snprintf(path, sizeof(path), "/proc/%d/status", pid) < 0) {
        perror("getLastPidNamesSpace: snprintf failed");
        return false;
    }

    FILE *fstream = fopen(path, "r");
    if (fstream == NULL) {
        perror("getLastPidNamesSpace: fopen failed");
        return false;
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
        *nestedNs = true;
        *lastNsPid = tempNsPid;
    } else {
        *nestedNs = false;
        *lastNsPid = pid;
    }

    fclose(fstream);

    return true;
}

static bool
switchIPCNamespace(int pid) {
    char nsPath[PATH_MAX] = {0};
    int nsFd;
    if (snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/ipc", pid) < 0) {
        perror("setNamespace: snprintf failed");
        return false;
    }

    if ((nsFd = open(nsPath, O_RDONLY)) == -1) {
        perror("setNamespace: open failed");
        return false;
    }

    if (setns(nsFd, 0) != 0) {
        perror("setNamespace: setns failed");
        close(nsFd);
        return false;
    }

    close(nsFd);

    return true;
}

static void
cleanupReadDesc(void) {
    mq_close(writeMqDesc);
    mq_unlink(writerMqName);
    mq_close(readMqDesc);
    mq_unlink(readerMqName);
}

static int
printUsageAndExit(const char *cmd) {
    printf("Usage: %s -p <pid_scope_process> [-i] [-r]\n", cmd);
    return EXIT_FAILURE;
}

static int
printRlimitAndExit(void) {
    struct rlimit rlim;
    if (!getrlimit(RLIMIT_MSGQUEUE, &rlim)) {
        printf("Rlimit RLIMIT_MSGQUEUE: current %zu max %zu", rlim.rlim_cur, rlim.rlim_max);
        return EXIT_SUCCESS;
    }
    printf("getrlimit failed errno %d", errno);
    return EXIT_FAILURE;
}

int main(int argc, char **argv) {
    struct mq_attr attr = {.mq_flags = 0, 
                               .mq_maxmsg = MAX_MESSAGES,
                               .mq_msgsize = MSG_BUFFER_SIZE,
                               .mq_curmsgs = 0};
    mode_t oldMask;
    int res = EXIT_FAILURE;
    bool ipcSwitch = false;
    bool nestedNs = false;
    int pid = -1;
    int nsPid = -1;
    int c;

    while ((c = getopt(argc, argv, "rip:")) != -1) {
        switch (c) {
            case 'r':
                return printRlimitAndExit();
                break;
            case 'i':
                ipcSwitch = true;
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
    if (getLastPidNamesSpace(pid, &nsPid, &nestedNs) == false) {
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

    atexit(cleanupReadDesc);

    writeMqDesc = mq_open(writerMqName, O_WRONLY);
    if (writeMqDesc == (mqd_t)-1) {
        perror("!mq_open writeMqDesc failed");
        return res;
    }

    char Txbuf[MSG_BUFFER_SIZE] = {0};
    char RxBuf[MSG_BUFFER_SIZE] = {0};

    printf("\nPass example message to stdin [type 'quit' to stop]\n");
    while (fgets(Txbuf, MSG_BUFFER_SIZE, stdin) != NULL) {
        if(strcmp("quit\n", Txbuf) == 0) {
            break;
        }

        // Send message to scoped process
        if (mq_send(writeMqDesc, Txbuf, strlen(Txbuf)-1, 0) == -1) {
            perror("!mq_send writeMqDesc failed");
            goto end_iteration;
        }


        // Read response
        if (mq_receive(readMqDesc, RxBuf, MSG_BUFFER_SIZE, NULL) == -1) {
            perror("!mq_receive readMqDesc failed");
            goto end_iteration;
        }

        if (ipcSwitch) {
            printf("Response from pid process %d : %s", pid, RxBuf);
        } else {
            printf("Response from pid [%d parent process] [%d inside container] : %s", pid, nsPid, RxBuf);
        }
        end_iteration:
            printf("\nPass example message to stdin [type 'quit' to stop]\n");
            memset(Txbuf, 0, MSG_BUFFER_SIZE);
            memset(RxBuf, 0, MSG_BUFFER_SIZE);
    }

    if (mq_close(writeMqDesc) == -1) {
        perror("!mq_close writeMqDesc failed");
    }

    return EXIT_SUCCESS;
}
