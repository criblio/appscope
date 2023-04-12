#define _GNU_SOURCE

#include "coredump.h"
#include "scopestdlib.h"
#include "utils.h"

#ifdef DISABLE_COREDUMP
bool
coreDumpGenerate(const char *path) {
    return FALSE;
}
#else
#include "google/coredumper.h"

/*
 * Generates core dump in location specifed by path
 *
 * Developer note: `WriteCoreDump` internally used PTRACE_ATTACH.
 * If You try to observe behavior of `WriteCoreDump` function
 * using GDB You will receive different result than in normal run.
 * Ref: Schrödinger's cat
 * 
 * Return status of operation
 */
bool
coreDumpGenerate(const char *path) {
    return (WriteCoreDump(path) == 0);
}
#endif
