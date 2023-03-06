#define _GNU_SOURCE
#include "snapshot.h"
#include "scopestdlib.h"
#include "log.h"
#include "utils.h"
#include "coredump.h"
#include "runtimecfg.h"
#include "fn.h"
#include "dbg.h"

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
#define snapshotWriteConstStr(fd, s) scope_write(fd, s, C_STRLEN(s))

/*
 * snapshotWriteStr - write the string with unknown length
 */
#define snapshotWriteStr(fd, s) scope_write(fd, s, (scope_strlen(s)))

/*
 * snapshotWriteNumber - convert specific number and writes it
 */
#define snapshotWriteNumberDec(fd, val) sigSafeWriteNumber(fd, val, 10)
#define snapshotWriteNumberHex(fd, val) sigSafeWriteNumber(fd, val, 16)


typedef bool               (*actionEnabledFunc)(void);
typedef bool               (*actionExecute)(const char *, const char *, siginfo_t *);

struct snapshotAction {
    actionEnabledFunc enabled;
    actionExecute execute;
};

// Current snapshot option
static unsigned snapshotOpt = 0;

/*
 * We use the bit information about enabled option
 * Do not change the order
 */
#define SNP_OPT_COREDUMP     0
#define SNP_OPT_BACKTRACE    1

/*
 * Always enable specific snapshot action
 */
static bool
snapActionAlwaysEnabled(void) {
    return TRUE;
}


/*
 * Application signal action
 */
static struct sigaction appSigSegvAction;
static struct sigaction appSigBusAction;
static struct sigaction appSigIllAction;
static struct sigaction appSigFpeAction;

/*
 * Save application signal handler
 * Returns TRUE in case of sig is a signal supported by snapshot
 */
bool
snapshotBackupAppSignalHandler(int sig, const struct sigaction *act) {
    if (sig == SIGSEGV) {
        appSigSegvAction = *act;
        return TRUE;
    } else if (sig == SIGBUS) {
        appSigBusAction = *act;
        return TRUE;
    } else if (sig == SIGILL) {
        appSigIllAction = *act;
        return TRUE;
    } else if (sig == SIGFPE) {
        appSigFpeAction = *act;
        return TRUE;
    }
    return FALSE;
}

bool
snapshotRetrieveAppSignalHandler(int sig, struct sigaction *const act) {
    if (sig == SIGSEGV) {
        *act = appSigSegvAction;
        return TRUE;
    } else if (sig == SIGBUS) {
        *act = appSigBusAction;
        return TRUE;
    } else if (sig == SIGILL) {
        *act = appSigIllAction;
        return TRUE;
    } else if (sig == SIGFPE) {
        *act = appSigFpeAction;
        return TRUE;
    }
    return FALSE;
}

static bool
handlerWasSaved(struct sigaction *act)
{
    if (act->sa_flags & SA_SIGINFO) {
        return act->sa_sigaction != NULL;
    }
    return act->sa_handler != NULL;
}

static void
callSavedHandler(struct sigaction *handler, int sig, siginfo_t *info, void *secret)
{
    if (!handlerWasSaved(handler)) return;

    // from the man page on sigaction:
    //   If SA_SIGINFO is specified in sa_flags, then sa_sigaction
    //     (instead of sa_handler) specifies the signal-handling function
    //     for signum.  This function receives three arguments, as described
    //     below.
    if (handler->sa_flags & SA_SIGINFO) {
        handler->sa_sigaction(sig, info, secret);
    } else {
        handler->sa_handler(sig);
    }
}

/*
 * Call original application handler saved by AppScope
 */
static void inline
appSignalHandler(int sig, siginfo_t *info, void *secret) {
    if (sig == SIGSEGV && handlerWasSaved(&appSigSegvAction)) {
        callSavedHandler(&appSigSegvAction, sig, info, secret);
    } else if (sig == SIGBUS && handlerWasSaved(&appSigBusAction)) {
        callSavedHandler(&appSigBusAction, sig, info, secret);
    } else if (sig == SIGILL && handlerWasSaved(&appSigIllAction)) {
        callSavedHandler(&appSigIllAction, sig, info, secret);
    } else if (sig == SIGFPE && handlerWasSaved(&appSigFpeAction)) {
        callSavedHandler(&appSigFpeAction, sig, info, secret);
    } else {
        // If there was no application handler just abort
        abort();
    }
}

/*
 * Snapshot info
 */
static bool
snapInfo(const char *dirPath, const char *epochStr, siginfo_t *unused) {
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "info_");
    scope_strcat(filePath, epochStr);
    int fd = scope_open(filePath, O_CREAT | S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (!fd) {
        return FALSE;
    }
    snapshotWriteConstStr(fd, "Scope Version: ");
    snapshotWriteConstStr(fd, SCOPE_VER);
    snapshotWriteConstStr(fd, "\nUnix Time: ");
    snapshotWriteConstStr(fd, epochStr);
    snapshotWriteConstStr(fd, " sec\nPID: ");
    snapshotWriteNumberDec(fd, g_proc.pid);
    snapshotWriteConstStr(fd ,"\nProcess name: ");
    snapshotWriteStr(fd, g_proc.procname);

    scope_close(fd);
    return (scope_chmod(filePath, 0755) == 0) ? TRUE : FALSE;
}

/*
 * Snapshot configuration
 */
static bool
snapConfig(const char *dirPath, const char *epochStr, siginfo_t *unused) {
    if (!g_cfg.cfgStr) {
        // should be unreachable
        return FALSE;
    }
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "cfg_");
    scope_strcat(filePath, epochStr);
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
    return SCOPE_BIT_CHECK(snapshotOpt, SNP_OPT_BACKTRACE);
}

/*
 * Snapshot backtrace
 */
static bool
snapBacktrace(const char *dirPath, const char *epochStr, siginfo_t *info) {
    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "backtrace_");
    scope_strcat(filePath, epochStr);
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

    // Analyze and print backtrace logic 
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
    return SCOPE_BIT_CHECK(snapshotOpt, SNP_OPT_COREDUMP);
}

/*
 * Snapshot coredump
 */
static bool
snapCoreDump(const char *dirPath, const char *epochStr, siginfo_t *unused) {
    // The current implementation of coredump does not support Go
    if (g_isgo) return TRUE;

    char filePath[PATH_MAX] = {0};
    scope_strcpy(filePath, dirPath);
    scope_strcat(filePath, "core_");
    scope_strcat(filePath, epochStr);
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
 * Enable/Disable core dump snapshot option
 */
void
snapshotSetCoredump(bool val) {
    SCOPE_BIT_SET_VAR(snapshotOpt, SNP_OPT_COREDUMP, val);
}

/*
 * Enable/Disable backtrace snapshot option
 */
void
snapshotSetBacktrace(bool val) {
    SCOPE_BIT_SET_VAR(snapshotOpt, SNP_OPT_BACKTRACE, val);
}

/*
 * Check if snapshot feature is enabled
 */
bool
snapshotIsEnabled(void) {
    return snapshotOpt ? TRUE : FALSE;
}
/*
 * Signal handler responsible for snapshot
 */
void
snapshotSignalHandler(int sig, siginfo_t *info, void *secret) {
    char snapPidDirPath[PATH_MAX] = {0};
    int currentOffset = SNAPSHOT_DIR_LEN;

    // Define that we are in a signal handler
    g_issighandler = TRUE;

    // Start with create a prefix path
    scope_memcpy(snapPidDirPath, SNAPSHOT_DIR_PREFIX, SNAPSHOT_DIR_LEN);

    char pidBuf[32] = {0};
    int msgLen = 0;

    // Append PID to path
    sigSafeUtoa(scope_getpid(), pidBuf, 10, &msgLen);
    currentOffset += msgLen + 1;
    if (currentOffset > PATH_MAX) {
        DBG(NULL);
        g_issighandler = FALSE;
        return;
    }
    scope_memcpy(snapPidDirPath + SNAPSHOT_DIR_LEN, pidBuf, msgLen);
    scope_memcpy(snapPidDirPath + SNAPSHOT_DIR_LEN + msgLen, "/", 1);
    // Create the base PID directory
    sigSafeMkdirRecursive(snapPidDirPath);

    // Snapshot Timestamp
    char timeBuf[1024] = {0};
    msgLen = 0;
    // Convert epoch to string
    sigSafeUtoa(scope_time(NULL), timeBuf, 10, &msgLen);

    // Perform all snapshot actions which are enabled
    for (int index = 0; index < ARRAY_SIZE(allSnapshotActions); ++index) {
        struct snapshotAction act = allSnapshotActions[index];
        if (act.enabled() == TRUE) {
            act.execute(snapPidDirPath, timeBuf, info);
        }
    }

    /*
    * This sleep is a fragile way to give the scope daemon time to read
    * the snapshot output files before this process exits (think: containerized
    * process). At the time this was written, after the daemon sees the signal
    * it just waits one second before grabbing these files. The coredump
    * is the last file we create, and while the coredump file creation
    * is *normally* less than a second it can take 60s (only seen on java7).
    * In short this is easy but needs improvement.
    */
    sleep(2);

    // We don't need to run the app's signal handler if we're dealing with a Go app
    // since we are hooked just before die ; and after any application-level signal handling
    if (!g_isgo) {
        appSignalHandler(sig, info, secret);
    }

    g_issighandler = FALSE;
}
