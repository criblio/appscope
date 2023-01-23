#define _GNU_SOURCE
#include "signalhandler.h"
#include "scopestdlib.h"
#include "log.h"
#include "utils.h"

#define UNW_LOCAL_ONLY
#include "libunwind.h"

#include <stdlib.h>

#define SYMBOL_BT_NAME_LEN (256)

extern log_t *g_log;

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
 * scopeLogErrorSigSafeStr - logs the const string 
 *
 */
#define scopeLogErrorSigSafeCStr(s) scopeLogSigSafe(s, sizeof(s) - 1)
#define scopeLogErrorSigSafeStr(s) scopeLogSigSafe(s, (scope_strlen(s)))

/*
 * Converts the specifc value using base for conversion to msg and
 * logging it with error level in signal safe way
 */
static void
scopeLogSigSafeNumber(long val, int base) {
    if (!g_log) {
        return;
    }
    char *bufOut = NULL;
    char buf[32] = {0};
    int msgLen = 0;
    bufOut = sigSafeUtoa(val, buf, base, &msgLen);
    logSigSafeSendWithLen(g_log, bufOut, msgLen, CFG_LOG_ERROR);
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

    unw_getcontext(&uc);
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

/*
 * Signal handler for SIGSEGV and SIGBUS.
 * Logs the backtrace inforamation.
 */
void
scopeSignalHandlerBacktrace(int sig, siginfo_t *info, void *secret) {
    scopeLogErrorSigSafeCStr("!scopeSignalHandlerBacktrace signal ");
    scopeLogSigSafeNumber(info->si_signo, 10);
    scopeLogErrorSigSafeCStr(" errno ");
    scopeLogSigSafeNumber(info->si_errno, 10);
    scopeLogErrorSigSafeCStr(" fault address ");
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
    }
    scopeLogBacktrace();
    abort();
}
