#define _GNU_SOURCE
#include "snapshot.h"
#include "scopestdlib.h"
#include "log.h"
#include "utils.h"
#include "coredump.h"
#include "runtimecfg.h"

#define UNW_LOCAL_ONLY
#include "libunwind.h"

#include <stdlib.h>

#define SYMBOL_BT_NAME_LEN (256)

/*
 * Prefix for snapshot directory
 * TODO: this can be configurable
 */
#define SNAPSHOT_DIR_PREFIX "/tmp/appscope/"
#define SNAPSHOT_DIR_LEN  C_STRLEN(SNAPSHOT_DIR_PREFIX)

extern proc_id_t g_proc;
extern rtconfig g_cfg;

/*
 * Helper functions for snapshot writing
 */

/*
 * snapshotWriteConstStr - write the string with known length
 */
#define snapshotWriteConstStr(fd, s) sigSafeWrite(fd, s, C_STRLEN(s))

/*
 * snapshotWriteStr - write the string with unknown length
 */
#define snapshotWriteStr(fd, s) sigSafeWrite(fd, s, (scope_strlen(s)))

/*
 * snapshotWriteNumber - convert specific number and writes it
 */
#define snapshotWriteNumberDec(fd, val) sigSafeWriteNumber(fd, val, 10)
#define snapshotWriteNumberHex(fd, val) sigSafeWriteNumber(fd, val, 16)


// TODO: Below probably need support some argument in case of optional action g_cfg ?
typedef bool               (*actionEnabledFunc)(void);
typedef bool               (*actionExecute)(const char *, siginfo_t *);

struct snapshotAction {
    actionEnabledFunc enabled;
    actionExecute execute;
};

/*
 * Always enable specific snapshot action
 */
static bool
snapActionAlwaysEnabled(void) {
    return TRUE;
}

/*
 * Snapshot info
 */
static bool
snapInfo(const char *dirPath, siginfo_t *unused) {
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "info");
    int fd = scope_open(filePath, O_CREAT | S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (!fd) {
        return FALSE;
    }
    snapshotWriteConstStr(fd, "Scope Version: ");
    snapshotWriteConstStr(fd, SCOPE_VER);
    snapshotWriteConstStr(fd, "\nUnix Time: ");
    snapshotWriteNumberDec(fd, scope_time(NULL));
    snapshotWriteConstStr(fd, " sec\nPID: ");
    snapshotWriteNumberDec(fd, g_proc.pid);
    snapshotWriteConstStr(fd ,"\nProcess name: ");
    snapshotWriteStr(fd, g_proc.procname);
    snapshotWriteConstStr(fd, "\n");

    scope_close(fd);
    return (scope_chmod(filePath, 0755) == 0) ? TRUE : FALSE;
}

/*
 * Snapshot configuration
 */
static bool
snapConfig(const char *dirPath, siginfo_t *unused) {
    if (!g_cfg.cfgStr) {
        // should be unreachable
        return FALSE;
    }
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "cfg");
    int fd = scope_open(filePath, O_CREAT | S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (!fd) {
        return FALSE;
    }

    snapshotWriteStr(fd, g_cfg.cfgStr);
    scope_close(fd);
    return (scope_chmod(filePath, 0755) == 0) ? TRUE : FALSE;
}

/*
 * Checks if backtrace snapshot action is enabled
 */
static bool
snapActionBacktraceEnabled(void) {
    // TODO: handle configuration here
    return TRUE;
}

/*
 * Snapshot backtrace
 */
static bool
snapBacktrace(const char *dirPath, siginfo_t *info) {
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "backtrace");
    int fd = scope_open(filePath, O_CREAT | S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (!fd) {
        return FALSE;
    }

    snapshotWriteConstStr(fd, "backtrace info: signal ");
    snapshotWriteNumberDec(fd, info->si_signo);
    snapshotWriteConstStr(fd, " errno ");
    snapshotWriteNumberDec(fd, info->si_errno);
    snapshotWriteConstStr(fd, " fault address 0x");
    snapshotWriteNumberHex(fd, (long)(info->si_addr));
    snapshotWriteConstStr(fd, ", reason of fault:\n");
    if (info->si_signo == SIGSEGV) {
        switch (info->si_code) {
            case SEGV_MAPERR:
                snapshotWriteConstStr(fd, "Address not mapped to object\n");
                break;
            case SEGV_ACCERR:
                snapshotWriteConstStr(fd, "Invalid permissions for mapped object\n");
                break;
            case SEGV_BNDERR:
                snapshotWriteConstStr(fd, "Failed address bound checks\n");
                break;
            case SEGV_PKUERR:
                snapshotWriteConstStr(fd, "Access was denied by memory protection keys\n");
                break;
            default: 
                snapshotWriteConstStr(fd, "Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGBUS) {
        switch (info->si_code) {
            case BUS_ADRALN:
                snapshotWriteConstStr(fd, "Invalid address alignment\n");
                break;
            case BUS_ADRERR:
                snapshotWriteConstStr(fd, "Nonexistent physical address\n");
                break;
            case BUS_OBJERR:
                snapshotWriteConstStr(fd, "Object-specific hardware error\n");
                break;
            case BUS_MCEERR_AR:
                snapshotWriteConstStr(fd, "Hardware memory error consumed on a machine check\n");
                break;
            case BUS_MCEERR_AO:
                snapshotWriteConstStr(fd, "Hardware memory error detected in process but not consumed\n");
                break;
            default: 
                snapshotWriteConstStr(fd, "Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGILL) {
        switch (info->si_code) {
            case ILL_ILLOPC:
                snapshotWriteConstStr(fd, "Illegal opcode\n");
                break;
            case ILL_ILLOPN:
                snapshotWriteConstStr(fd, "Illegal operand\n");
                break;
            case ILL_ILLADR:
                snapshotWriteConstStr(fd, "Illegal addressing mode\n");
                break;
            case ILL_ILLTRP:
                snapshotWriteConstStr(fd, "Illegal trap\n");
                break;
            case ILL_PRVOPC:
                snapshotWriteConstStr(fd, "Privileged opcode\n");
                break;
            case ILL_PRVREG:
                snapshotWriteConstStr(fd, "Privileged register\n");
                break;
            case ILL_COPROC:
                snapshotWriteConstStr(fd, "Coprocessor error\n");
                break;
            case ILL_BADSTK:
                snapshotWriteConstStr(fd, "Internal stack error\n");
                break;
            default: 
                snapshotWriteConstStr(fd, "Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGFPE) {
        switch (info->si_code) {
            case FPE_INTDIV:
                snapshotWriteConstStr(fd, "Integer divide by zero\n");
                break;
            case FPE_INTOVF:
                snapshotWriteConstStr(fd, "Integer overflow\n");
                break;
            case FPE_FLTDIV:
                snapshotWriteConstStr(fd, "Floating point divide by zero\n");
                break;
            case FPE_FLTOVF:
                snapshotWriteConstStr(fd, "Floating point overflow\n");
                break;
            case FPE_FLTUND:
                snapshotWriteConstStr(fd, "Floating point underflow\n");
                break;
            case FPE_FLTRES:
                snapshotWriteConstStr(fd, "Floating point inexact result\n");
                break;
            case FPE_FLTINV:
                snapshotWriteConstStr(fd, "Floating point invalid operation\n");
                break;
            default: 
                snapshotWriteConstStr(fd, "Unknown Error\n");
                break;
        }
    }

    // Analyze and print stacktrace logic 
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip;

    unw_scope_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    int frame_count = 0;
    snapshotWriteConstStr(fd, "--- backtrace\n");
    while(unw_step(&cursor) > 0) {
        char symbol[SYMBOL_BT_NAME_LEN] = {0};
        unw_word_t offset;

        int ret = unw_get_reg(&cursor, UNW_REG_IP, &ip);
        if (ret) {
            continue;
        }

        ret = unw_get_proc_name(&cursor, symbol, SYMBOL_BT_NAME_LEN, &offset);
        snapshotWriteConstStr(fd, "#");
        snapshotWriteNumberDec(fd, frame_count);
        snapshotWriteConstStr(fd, " 0x");
        snapshotWriteNumberHex(fd, ip);
        if (!ret) {
            snapshotWriteConstStr(fd, " ");
            snapshotWriteStr(fd, symbol);
            snapshotWriteConstStr(fd, " + ");
            snapshotWriteNumberDec(fd, offset);
        } else {
            snapshotWriteConstStr(fd, " ?");
        }
        snapshotWriteConstStr(fd, "\n");

        frame_count++;
    }

    scope_close(fd);
    return (scope_chmod(filePath, 0755) == 0) ? TRUE : FALSE;
}

/*
 * Checks if coredump snapshot action is enabled
 */
static bool
snapActionCoredumpEnabled(void) {
    // TODO: handle configuration here
    return TRUE;
}

/*
 * Snapshot coredump
 */
static bool
snapCoreDump(const char *dirPath, siginfo_t *unused) {
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "core");
    return coreDumpGenerate(filePath);
}

/*
 * Array of all snapshot actions
 */
static const
struct snapshotAction allSnapshotActions[] = {
    {.enabled = snapActionAlwaysEnabled,    .execute = snapInfo},
    {.enabled = snapActionAlwaysEnabled,    .execute = snapConfig},
    {.enabled = snapActionBacktraceEnabled, .execute = snapBacktrace},
    {.enabled = snapActionCoredumpEnabled,  .execute = snapCoreDump},
};

/*
 * Signal handler responsible for snapshot
 */
void
snapshotSignalHandler(int sig, siginfo_t *info, void *secret) {
    char snapPidDirPath[PATH_MAX] = {0};
    int currentOffset = SNAPSHOT_DIR_LEN;
    // Start with create a prefix path
    scope_memcpy(snapPidDirPath, SNAPSHOT_DIR_PREFIX, SNAPSHOT_DIR_LEN);

    char pidBuf[32] = {0};
    int msgLen = 0;

    // Append PID to path
    sigSafeUtoa(scope_getpid(), pidBuf, 10, &msgLen);
    currentOffset += msgLen + 1;
    if (currentOffset> PATH_MAX) {
        // This should never happened
        return;
    }
    scope_memcpy(snapPidDirPath + SNAPSHOT_DIR_LEN, pidBuf, msgLen);
    scope_memcpy(snapPidDirPath + SNAPSHOT_DIR_LEN + msgLen, "/", 1);
    // Create the base PID directory
    sigSafeMkdirRecursive(snapPidDirPath);

    // Perform all snapshot actions which are enabled
    for (int index = 0; index < ARRAY_SIZE(allSnapshotActions); ++index) {
        struct snapshotAction act = allSnapshotActions[index];
        if (act.enabled() == TRUE) {
            act.execute(snapPidDirPath, info);
        }
    }

    raise(SIGSTOP);
}
