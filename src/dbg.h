#ifndef __DBG_H__
#define __DBG_H__

#include "log.h"
#include "scopetypes.h"
#include <stdio.h>

typedef struct _dbg_t dbg_t;

extern dbg_t *g_dbg;

// Constructors Destructors
void dbgInit();
void dbgDestroy();

// Accessors
unsigned long long dbgCountAllLines();
unsigned long long dbgCountMatchingLines(const char *);
void dbgDumpAll(FILE *);

// Setters
void dbgAddLine(const char *key, const char *fmt, ...);

#define STRINGIFY(x)      #x
#define TOSTRING(x)       STRINGIFY(x)
#define DBG_FILE_AND_LINE __FILE__ ":" TOSTRING(__LINE__)

#define PRINTF_FORMAT(fmt_id, arg_id) __attribute__((format(printf, (fmt_id), (arg_id))))

//
//  The DBG macro is used to keep track of unexpected/undesirable
//  conditions as instrumented with DBG in the source code.  This is done
//  by storing the source file and line of every DBG macro that is executed.
//
//  At the most basic level, a count is incremented for each file/line.
//  In addition to this the time, errno, and optionally a string are stored
//  for the earliest and most recent time each file/line is hit.
//
//  Example uses:
//     DBG(NULL);                                    // No optional string
//     DBG("Should never get here");                 // Boring string
//     DBG("Hostname/port: %s:%d", hostname, port)   // Formatted string

#define DBG(...) dbgAddLine(DBG_FILE_AND_LINE, ##__VA_ARGS__)

//
//  Dynamic commands allow this information to be output from an actively
//  running process, with process ID <pid>.  It just runs dbgDumpAll(),
//  outputting the results to the file specified by SCOPE_CMD_DBG_PATH.
//  To do this with default configuration settings, run this command and
//  output should appear in /tmp/mydbg.txt within a SCOPE_SUMMARY_PERIOD:
//
//     echo "SCOPE_CMD_DBG_PATH=/tmp/mydbg.txt" >> /tmp/scope.<pid>
//

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
// logging utilities
//

extern log_t *g_log;
extern proc_id_t g_proc;
extern bool g_constructor_debug_enabled;

void scopeLog(cfg_log_level_t, const char *, ...) PRINTF_FORMAT(2, 3);
void scopeLogHex(cfg_log_level_t, const void *, size_t, const char *, ...) PRINTF_FORMAT(4, 5);

#define scopeLogError(...) scopeLog(CFG_LOG_ERROR, __VA_ARGS__)
#define scopeLogWarn(...)  scopeLog(CFG_LOG_WARN, __VA_ARGS__)
#define scopeLogInfo(...)  scopeLog(CFG_LOG_INFO, __VA_ARGS__)
#ifdef DEBUG
#define scopeLogDebug(...) scopeLog(CFG_LOG_DEBUG, __VA_ARGS__)
#define scopeLogTrace(...) scopeLog(CFG_LOG_TRACE, __VA_ARGS__)
#else
#define scopeLogDebug(...)
#define scopeLogTrace(...)
#endif

#define scopeLogHexError(...) scopeLogHex(CFG_LOG_ERROR, __VA_ARGS__)
#define scopeLogHexWarn(...)  scopeLogHex(CFG_LOG_WARN, __VA_ARGS__)
#define scopeLogHexInfo(...)  scopeLogHex(CFG_LOG_INFO, __VA_ARGS__)
#ifdef DEBUG
#define scopeLogHexDebug(...) scopeLogHex(CFG_LOG_DEBUG, __VA_ARGS__)
#define scopeLogHexTrace(...) scopeLogHex(CFG_LOG_TRACE, __VA_ARGS__)
#else
#define scopeLogHexDebug(...)
#define scopeLogHexTrace(...)
#endif

#endif // __DBG_H__
