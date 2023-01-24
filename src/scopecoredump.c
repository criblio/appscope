#define _GNU_SOURCE

#include "scopecoredump.h"
#include "scopestdlib.h"
#include "utils.h"
#include "google/coredumper.h"

// Prefix for core dump file
#define CORE_PREFIX "/tmp/scope_core."

/*
 * Generates core dump in location based on pid
 * Return status of operation
 */
bool
scopeCoreDumpGenerate(pid_t pid) {
    char path[PATH_MAX] = {0};
    scope_memcpy(path, CORE_PREFIX, sizeof(CORE_PREFIX) - 1);
    char pidBuf[32] = {0};
    int msgLen = 0;
    sigSafeUtoa(pid, pidBuf, 10, &msgLen);
    scope_memcpy(path + sizeof(CORE_PREFIX) - 1, pidBuf, msgLen);
    return (WriteCoreDump(path) == 0);
}
