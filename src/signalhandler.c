#define _GNU_SOURCE
#include "signalhandler.h"
#include "scopestdlib.h"
#include "log.h"
#include "utils.h"
#include "scopecoredump.h"

#define UNW_LOCAL_ONLY
#include "libunwind.h"

#include <stdlib.h>

#define SYMBOL_BT_NAME_LEN (256)

extern log_t *g_log;
extern proc_id_t g_proc;
/*
 * Logs the specific message with error level in signal safe way
 */
static void
scopeLogSigSafe(const char *msg, size_t msgLen) {
    if (!g_log) {
        return;
    }
    logSigSafeSendWithLen(g_log, msg, msgLen, CFG_LOG_ERROR);
}

/*
 * Helper functions for signal safe logging
 *
 * scopeLogErrorSigSafeCStr - logs the const string
 * scopeLogErrorSigSafeStr - logs the string with unknown length
 *
 */
#define scopeLogErrorSigSafeCStr(s) scopeLogSigSafe(s, C_STRLEN(s))
#define scopeLogErrorSigSafeStr(s) scopeLogSigSafe(s, (scope_strlen(s)))

/*
 * Converts the specific value using base for conversion to msg and
 * logging it with error level in signal safe way
 */
static void
scopeLogSigSafeNumber(long val, int base) {
    if (!g_log) {
        return;
    }

    char buf[32] = {0};
    int msgLen = 0;
    sigSafeUtoa(val, buf, base, &msgLen);
    logSigSafeSendWithLen(g_log, buf, msgLen, CFG_LOG_ERROR);
}

/*
 * Formats the current backtrace into human-readable description
 * and logging it with error level in signal safe way 
 */
static void
scopeLogBacktrace(void) {
    unw_cursor_t cursor;
    unw_context_t uc;
    unw_word_t ip;

    unw_scope_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    int frame_count = 0;
    scopeLogErrorSigSafeCStr("--- scopeLogBacktrace\n");
    while(unw_step(&cursor) > 0) {
        char symbol[SYMBOL_BT_NAME_LEN];
        unw_word_t offset;

        int ret = unw_get_reg(&cursor, UNW_REG_IP, &ip);
        if (ret) {
            continue;
        }

        ret = unw_get_proc_name(&cursor, symbol, SYMBOL_BT_NAME_LEN, &offset);
        scopeLogErrorSigSafeCStr("#");
        scopeLogSigSafeNumber(frame_count, 10);
        scopeLogErrorSigSafeCStr(" 0x");
        scopeLogSigSafeNumber(ip, 16);
        if (!ret) {
            scopeLogErrorSigSafeCStr(" ");
            scopeLogErrorSigSafeStr(symbol);
            scopeLogErrorSigSafeCStr(" + ");
            scopeLogSigSafeNumber(offset, 10);
        } else {
            scopeLogErrorSigSafeCStr(" ?");
        }
        scopeLogErrorSigSafeCStr("\n");

        frame_count++;
    }
}

static inline void
logBacktraceInfo(siginfo_t *info) {
    scopeLogErrorSigSafeCStr("Scope Version: ");
    scopeLogErrorSigSafeCStr(SCOPE_VER);
    scopeLogErrorSigSafeCStr("\n");
    scopeLogErrorSigSafeCStr("Unix Time: ");
    scopeLogSigSafeNumber(scope_time(NULL), 10);
    scopeLogErrorSigSafeCStr(" sec\n");
    scopeLogErrorSigSafeCStr("PID: ");
    scopeLogSigSafeNumber(g_proc.pid, 10);
    scopeLogErrorSigSafeCStr("\n");
    scopeLogErrorSigSafeCStr("Process name: ");
    scopeLogErrorSigSafeStr(g_proc.procname);
    scopeLogErrorSigSafeCStr("\n");
    scopeLogErrorSigSafeCStr("!scopeSignalHandlerBacktrace signal ");
    scopeLogSigSafeNumber(info->si_signo, 10);
    scopeLogErrorSigSafeCStr(" errno ");
    scopeLogSigSafeNumber(info->si_errno, 10);
    scopeLogErrorSigSafeCStr(" fault address 0x");
    scopeLogSigSafeNumber((long)(info->si_addr), 16);
    scopeLogErrorSigSafeCStr(", reason of fault:\n");
    int sig_code = info->si_code;
    if (info->si_signo == SIGSEGV) {
        switch (sig_code) {
            case SEGV_MAPERR:
                scopeLogErrorSigSafeCStr("Address not mapped to object\n");
                break;
            case SEGV_ACCERR:
                scopeLogErrorSigSafeCStr("Invalid permissions for mapped object\n");
                break;
            case SEGV_BNDERR:
                scopeLogErrorSigSafeCStr("Failed address bound checks\n");
                break;
            case SEGV_PKUERR:
                scopeLogErrorSigSafeCStr("Access was denied by memory protection keys\n");
                break;
            default: 
                scopeLogErrorSigSafeCStr("Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGBUS) {
        switch (sig_code) {
            case BUS_ADRALN:
                scopeLogErrorSigSafeCStr("Invalid address alignment\n");
                break;
            case BUS_ADRERR:
                scopeLogErrorSigSafeCStr("Nonexistent physical address\n");
                break;
            case BUS_OBJERR:
                scopeLogErrorSigSafeCStr("Object-specific hardware error\n");
                break;
            case BUS_MCEERR_AR:
                scopeLogErrorSigSafeCStr("Hardware memory error consumed on a machine check\n");
                break;
            case BUS_MCEERR_AO:
                scopeLogErrorSigSafeCStr("Hardware memory error detected in process but not consumed\n");
                break;
            default: 
                scopeLogErrorSigSafeCStr("Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGILL) {
        switch (sig_code) {
            case ILL_ILLOPC:
                scopeLogErrorSigSafeCStr("Illegal opcode\n");
                break;
            case ILL_ILLOPN:
                scopeLogErrorSigSafeCStr("Illegal operand\n");
                break;
            case ILL_ILLADR:
                scopeLogErrorSigSafeCStr("Illegal addressing mode\n");
                break;
            case ILL_ILLTRP:
                scopeLogErrorSigSafeCStr("Illegal trap\n");
                break;
            case ILL_PRVOPC:
                scopeLogErrorSigSafeCStr("Privileged opcode\n");
                break;
            case ILL_PRVREG:
                scopeLogErrorSigSafeCStr("Privileged register\n");
                break;
            case ILL_COPROC:
                scopeLogErrorSigSafeCStr("Coprocessor error\n");
                break;
            case ILL_BADSTK:
                scopeLogErrorSigSafeCStr("Internal stack error\n");
                break;
            default: 
                scopeLogErrorSigSafeCStr("Unknown Error\n");
                break;
        }
    } else if (info->si_signo == SIGFPE) {
        switch (sig_code) {
            case FPE_INTDIV:
                scopeLogErrorSigSafeCStr("Integer divide by zero\n");
                break;
            case FPE_INTOVF:
                scopeLogErrorSigSafeCStr("Integer overflow\n");
                break;
            case FPE_FLTDIV:
                scopeLogErrorSigSafeCStr("Floating point divide by zero\n");
                break;
            case FPE_FLTOVF:
                scopeLogErrorSigSafeCStr("Floating point overflow\n");
                break;
            case FPE_FLTUND:
                scopeLogErrorSigSafeCStr("Floating point underflow\n");
                break;
            case FPE_FLTRES:
                scopeLogErrorSigSafeCStr("Floating point inexact result\n");
                break;
            case FPE_FLTINV:
                scopeLogErrorSigSafeCStr("Floating point invalid operation\n");
                break;
            default: 
                scopeLogErrorSigSafeCStr("Unknown Error\n");
                break;
        }
    }
    scopeLogBacktrace();
}

/*
 * Signal handler which logs the backtrace information.
 */
void
scopeSignalHandlerBacktrace(int sig, siginfo_t *info, void *secret) {
    logBacktraceInfo(info);
    abort();
}

/*
 * Signal handler which generates core dump.
 */
void
scopeSignalHandlerCoreDump(int sig, siginfo_t *info, void *secret) {
    scopeCoreDumpGenerate(scope_getpid());
    abort();
}

/*
 * Signal handler which logs the backtrace information and generates core dump.
 */
void
scopeSignalHandlerFull(int sig, siginfo_t *info, void *secret) {
    logBacktraceInfo(info);
    scopeLogErrorSigSafeCStr("Core dump generation status: ");
    if (scopeCoreDumpGenerate(scope_getpid()) == TRUE) {
        scopeLogErrorSigSafeCStr("success\n");
    } else {
        scopeLogErrorSigSafeCStr("failure\n");
    }
    abort();
}
