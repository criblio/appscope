#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "ns.h"
#include "scopestdlib.h"

/*
 * TODO: Refactor this hardcoded path
 * This can be consolidated with libdir.c but required
 * further cleaning like reverse return logic in libdirExists
 */
#define LDSCOPE_IN_CHILD_NS "/tmp/ldscope"
#define VALID_NS_DEPTH 2

/*
 * Load static loader (ldscope) into memory.
 *
 * Returns loader memory address in case of success, NULL otherwise.
 */
static char*
loadLdscopeMem(size_t *ldscopeSize)
{
    // Load ldscope into memory
    char *ldscopeMem = NULL;

    char ldscopePath[PATH_MAX] = {0};

    if (scope_readlink("/proc/self/exe", ldscopePath, sizeof(ldscopePath) - 1) == -1) {
        scope_perror("readlink(/proc/self/exe) failed");
        goto closeLdscopeFd;
    }

    int ldscopeInputFd = scope_open(ldscopePath, O_RDONLY);
    if (!ldscopeInputFd) {
        scope_perror("scope_open failed");
        goto closeLdscopeFd;
    }

    *ldscopeSize = scope_lseek(ldscopeInputFd, 0, SEEK_END);
    if (*ldscopeSize == (off_t)-1) {
        scope_perror("scope_lseek failed");
        goto closeLdscopeFd;
    }

    ldscopeMem = scope_mmap(NULL, *ldscopeSize, PROT_READ, MAP_PRIVATE, ldscopeInputFd, 0);
    if (ldscopeMem == MAP_FAILED) {
        scope_perror("scope_mmap failed");
        ldscopeMem = NULL;
        goto closeLdscopeFd;
    }

closeLdscopeFd:

    scope_close(ldscopeInputFd);

    return ldscopeMem;
}

/*
 * Extract static loader (ldscope) into child namespace.
 *
 * Returns TRUE in case of success, FALSE otherwise.
 */
static bool
extractLdscopeToChildNamespace(char* ldscopeMem, size_t ldscopeSize)
{
    bool status = FALSE;

    int ldscopeDestFd = scope_open(LDSCOPE_IN_CHILD_NS, O_RDWR | O_CREAT, 0771);
    if (!ldscopeDestFd) {
        scope_perror("scope_open failed");
        return status;
    }

    if (scope_ftruncate(ldscopeDestFd, ldscopeSize) != 0) {
        goto cleanupDestFd;
    }

    char* dest = scope_mmap(NULL, ldscopeSize, PROT_READ | PROT_WRITE, MAP_SHARED, ldscopeDestFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    scope_memcpy(dest, ldscopeMem, ldscopeSize);

    scope_munmap(dest, ldscopeSize);

    status = TRUE;

cleanupDestFd:

    scope_close(ldscopeDestFd);

    return TRUE;
}

static bool
join_namespace(pid_t hostPid)
{
    bool status = FALSE;
    size_t ldscopeSize = 0;

    char *ldscopeMem = loadLdscopeMem(&ldscopeSize);
    if (ldscopeMem == NULL) {
        return status;
    }

    /*
    * Reassociate current process to the "child namespace"
    * - PID namespace - allows to "child process" of the calling process 
    *   be created in separate namespace
    *   In other words the calling process will not change it's ownPID
    *   namespace
    * - mount namespace - allows to copy static loader into a "child namespace"
    */
    char *nsType[] = {"pid", "mnt"};

    for (int i = 0; i < (sizeof(nsType)/sizeof(nsType[0])); ++i) {
        char nsPath[PATH_MAX] = {0};
        if (scope_snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/%s", hostPid, nsType[i]) < 0) {
            scope_perror("scope_snprintf failed");
            goto cleanupLdscopeMem;
        }
        int nsFd = scope_open(nsPath, O_RDONLY);
        if (!nsFd) {
            scope_perror("scope_open failed");
            goto cleanupLdscopeMem;
        }

        if (scope_setns(nsFd, 0) != 0) {
            scope_perror("setns failed");
            goto cleanupLdscopeMem;
        }
    }

    status = extractLdscopeToChildNamespace(ldscopeMem, ldscopeSize);

cleanupLdscopeMem:

    scope_munmap(ldscopeMem, ldscopeSize);

    return status;
}

/*
 * Check for PID in the child namespace.
 *
 * Returns TRUE if specific process contains two namespaces FALSE otherwise.
 */
bool
nsIsPidInChildNs(pid_t pid, pid_t *nsPid)
{
    char path[PATH_MAX] = {0};
    char buffer[4096];
    bool status = FALSE;
    int lastNsPid = 0;
    int nsDepth = 0;

    if (scope_snprintf(path, sizeof(path), "/proc/%d/status", pid) < 0) return FALSE;

    FILE *fstream = scope_fopen(path, "r");

    if (fstream == NULL) {
        return FALSE;
    }

    while (scope_fgets(buffer, sizeof(buffer), fstream)) {
        if (scope_strstr(buffer, "NSpid:")) {
            const char delimiters[] = ": \t";
            char *entry, *last;

            entry = scope_strtok_r(buffer, delimiters, &last);
            // Skip NsPid string
            entry = scope_strtok_r(NULL, delimiters, &last);
            // Iterate over NsPids values
            while (entry != NULL) {
                lastNsPid = scope_atoi(entry);
                entry = scope_strtok_r(NULL, delimiters, &last);
                nsDepth++;
            }
            break;
        }
    }

    /*
    * TODO: we currently tested nesting depth 
    * equals VALID_NS_DEPTH, check more depth level
    */
    if (nsDepth == VALID_NS_DEPTH) {
        status = TRUE;
        *nsPid = lastNsPid;
    }

    scope_fclose(fstream);

    return status;
}
 
 /*
 * Perform fork and exec which cause that direct children
 * effectively will join a new PID namespace
 *
 * Returns status of operation
 */
int
nsForkAndExec(pid_t parentPid, pid_t nsPid)
{
    if (join_namespace(parentPid) == FALSE) {
        scope_fprintf(scope_stderr, "error: join_namespace failed\n");
        return EXIT_FAILURE; 
    }
    pid_t child = fork();
    if (child < 0) {
        scope_fprintf(scope_stderr, "error: fork() failed\n");
        return EXIT_FAILURE;
    } else if (child == 0) {
        // Child
        char *nsAttachPidStr = NULL;
        if (scope_asprintf(&nsAttachPidStr, "%d", nsPid) <= 0) {
            scope_perror("error: asprintf() failed\n");
            return EXIT_FAILURE;
        }
        int execArgc = 0;
        char **execArgv = scope_calloc(4, sizeof(char *));
        if (!execArgv) {
            scope_fprintf(scope_stderr, "error: calloc() failed\n");
            return EXIT_FAILURE;
        }

        execArgv[execArgc++] = LDSCOPE_IN_CHILD_NS;

        execArgv[execArgc++] = "-a";
        execArgv[execArgc++] = nsAttachPidStr;

        return execve(LDSCOPE_IN_CHILD_NS, execArgv, environ);
    }
    // Parent
    int status;
    scope_waitpid(child, &status, 0);
    if (WIFEXITED(status)) {
        int exitChildStatus = WEXITSTATUS(status);
        if (exitChildStatus == 0) {
            scope_fprintf(scope_stderr, "Attach to process %d in child process succeeded\n", parentPid);
        } else {
            scope_fprintf(scope_stderr, "Attach to process %d in child process failed\n", parentPid);
        }
        return exitChildStatus;
    }
    scope_fprintf(scope_stderr, "error: attach failed() failed\n");
    return EXIT_FAILURE;
}
