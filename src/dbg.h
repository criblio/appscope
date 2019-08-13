#ifndef __DBG_H__
#define __DBG_H__

#include <stdio.h>

typedef struct _dbg_t dbg_t;

extern dbg_t* g_dbg;

// Constructors Destructors
void                 dbgInit();
void                 dbgDestroy();

// Accessors
unsigned long long   dbgCountAllLines();
unsigned long long   dbgCountMatchingLines(const char*);
void                 dbgDumpAll(FILE*);

// Setters
void                 dbgAddLine(const char* key, const char* fmt, ...);

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

//
//  The DBG macro is used to keep track of unexpected/undesireable
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

#define DBG(...) dbgAddLine(AT, ## __VA_ARGS__)

#endif // __DBG_H__
