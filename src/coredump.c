#define _GNU_SOURCE

#include "coredump.h"
#include "scopestdlib.h"
#include "utils.h"
#include "google/coredumper.h"


/*
 * Generates core dump in location based on <pathPrefix><pid>
 * Return status of operation
 */
bool
coreDumpGenerate(const char *pathPrefix, size_t pathPrefixSize, pid_t pid) {
    char path[PATH_MAX] = {0};
    if (pathPrefixSize > PATH_MAX) {
        return FALSE;
    }
    scope_memcpy(path, pathPrefix, pathPrefixSize);
    char pidBuf[32] = {0};
    int msgLen = 0;
    sigSafeUtoa(pid, pidBuf, 10, &msgLen);
    if (pathPrefixSize + msgLen > PATH_MAX) {
        return FALSE;
    }
    scope_memcpy(path + pathPrefixSize, pidBuf, msgLen);

    return (WriteCoreDump(path) == 0);
}
