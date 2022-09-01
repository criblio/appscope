#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "ns.h"
#include "setup.h"
#include "scopestdlib.h"

#define LDSCOPE_IN_CHILD_NS "/tmp/ldscope"
#define VALID_NS_DEPTH 2


/*
 * Extract file from parent namespace into child namespace.
 *
 * Returns TRUE in case of success, FALSE otherwise.
 */
static bool
extractMemToChildNamespace(char* inputMem, size_t inputSize, const char *outFile, mode_t outPermFlag)
{
    bool status = FALSE;
    int outFd;

    if ((outFd = scope_open(outFile, O_RDWR | O_CREAT, outPermFlag)) == -1) {
        scope_perror("scope_open failed");
        return status;
    }

    if (scope_ftruncate(outFd, inputSize) != 0) {
        goto cleanupDestFd;
    }

    char* dest = scope_mmap(NULL, inputSize, PROT_READ | PROT_WRITE, MAP_SHARED, outFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    scope_memcpy(dest, inputMem, inputSize);

    scope_munmap(dest, inputSize);

    status = TRUE;

cleanupDestFd:

    scope_close(outFd);

    return status;
}

static bool
setNamespace(pid_t pid, const char* ns)
{
    bool res = FALSE;
    char nsPath[PATH_MAX] = {0};
    int nsFd;
    if (scope_snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/%s", pid, ns) < 0) {
        scope_perror("scope_snprintf failed");
        goto exit;
    }

    if ((nsFd = scope_open(nsPath, O_RDONLY)) == -1) {
        scope_perror("scope_open failed");
        goto exit;
    }

    if (scope_setns(nsFd, 0) != 0) {
        scope_perror("setns failed");
        goto exit;
    }

    res = TRUE;
exit:

    return res;
}

static bool
join_namespace(pid_t hostPid)
{
    bool status = FALSE;
    size_t ldscopeSize = 0;
    size_t cfgSize = 0;

    char path[PATH_MAX] = {0};

    if (scope_readlink("/proc/self/exe", path, sizeof(path) - 1) == -1) {
       return status;
    }

    char *ldscopeMem = setupLoadFileIntoMem(&ldscopeSize, path);
    if (ldscopeMem == NULL) {
        return status;
    }

    // Configuration is optional
    char *scopeCfgMem = setupLoadFileIntoMem(&cfgSize, getenv("SCOPE_CONF_PATH"));

    /*
    * Reassociate current process to the "child namespace"
    * - PID namespace - allows to "child process" of the calling process 
    *   be created in separate namespace
    *   In other words the calling process will not change it's ownPID
    *   namespace
    * - mount namespace - allows to copy file(s) into a "child namespace"
    */
    if (setNamespace(hostPid, "pid") == FALSE) {
        goto cleanupMem;
    }
    if (setNamespace(hostPid, "mnt") == FALSE) {
        goto cleanupMem;
    }

    status = extractMemToChildNamespace(ldscopeMem, ldscopeSize, LDSCOPE_IN_CHILD_NS, 0775);

    if (scopeCfgMem) {
        char scopeCfgPath[PATH_MAX] = {0};

        scope_snprintf(scopeCfgPath, sizeof(scopeCfgPath), "/tmp/scope_%d.yml", hostPid);
        status = extractMemToChildNamespace(scopeCfgMem, cfgSize, scopeCfgPath, 0664);
        // replace the SCOPE_CONF_PATH with namespace path
        setenv("SCOPE_CONF_PATH", scopeCfgPath, 1);
    }   

cleanupMem:

    scope_munmap(ldscopeMem, ldscopeSize);

    if (scopeCfgMem) {
        scope_munmap(scopeCfgMem, cfgSize);
    }

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
 * Setup the service for specified child process
 * Returns status of operation 0 in case of success, other values in case of failure
 */
int
nsService(pid_t pid, const char* serviceName) {

    if (setNamespace(pid, "mnt") == FALSE) {
        return -1;
    }

    return setupService(serviceName);
}

 
 /*
 * Configure the child mount namespace
 * - switch the mount namespace to child
 * - configure the setup
 * Returns status of operation 0 in case of success, other values in case of failure
 */
int
nsConfigure(pid_t pid, void* scopeCfgFilterMem, size_t filterFileSize)
{
    if (setNamespace(pid, "mnt") == FALSE) {
        scope_fprintf(scope_stderr, "setNamespace mnt failed\n");
        return EXIT_FAILURE;
    }

    if (setupConfigure(scopeCfgFilterMem, filterFileSize, TRUE) == FALSE) {
        scope_fprintf(scope_stderr, "setup child namespace failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
 
 /*
 * Perform fork and exec which cause that direct children
 * effectively will join a new PID namespace.
 *
 * Reassociating the PID namespace (setns CLONE_NEWPID) has somewhat different from 
 * other namespace types. Reassociating the calling thread with a PID namespace
 * changes only the PID namespace that subsequently created child processes of
 * the caller will be placed in. It does not change the PID namespace of the caller itself.
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
