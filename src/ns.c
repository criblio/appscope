#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "ns.h"
#include "loaderop.h"
#include "service.h"
#include "libdir.h"
#include "scopestdlib.h"

/*
 * TODO: Refactor this hardcoded path
 * This can be consolidated with libdir.c but required
 * further cleaning like reverse return logic in libdirExists
 */
#define LDSCOPE_IN_CHILD_NS "/tmp/ldscope"
#define LIBSCOPE_IN_CHILD_NS "/tmp/libscope.so"
#define PROFILE_SETUP "LD_PRELOAD=/tmp/libscope.so\n"
#define PROFILE_SETUP_LEN (sizeof(PROFILE_SETUP)-1)

#define VALID_NS_DEPTH 2

/*
 * Load File into memory.
 *
 * Returns memory address in case of success, NULL otherwise.
 */
static char*
loadFileIntoMem(size_t *size, char* path)
{
    // Load file into memory
    char *resMem = NULL;
    int fd;

    if (path == NULL) {
        return resMem;
    }

    if ((fd = scope_open(path, O_RDONLY)) == -1) {
        scope_perror("scope_open failed");
        goto closeFd;
    }

    *size = scope_lseek(fd, 0, SEEK_END);
    if (*size == (off_t)-1) {
        scope_perror("scope_lseek failed");
        goto closeFd;
    }

    resMem = scope_mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (resMem == MAP_FAILED) {
        scope_perror("scope_mmap failed");
        resMem = NULL;
        goto closeFd;
    }

closeFd:

    scope_close(fd);

    return resMem;
}

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

    char *ldscopeMem = loadFileIntoMem(&ldscopeSize, path);
    if (ldscopeMem == NULL) {
        return status;
    }

    // Configuration is optional
    char *scopeCfgMem = loadFileIntoMem(&cfgSize, getenv("SCOPE_CONF_PATH"));

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
 * Setup the /etc/profile scope startup script
 * Returns status of operation TRUE in case of success, FALSE otherwise
 */
static bool
setupProfile(void){
    int fd = scope_open("/etc/profile.d/scope.sh", O_CREAT | O_RDWR | O_TRUNC, 0644);

    if (fd < 0) {
        scope_perror("scope_fopen failed");
        return FALSE;
    }

    if (scope_write(fd, PROFILE_SETUP, PROFILE_SETUP_LEN) != PROFILE_SETUP_LEN) {
        scope_perror("scope_write failed");
        scope_close(fd);
        return FALSE;
    }

    if (scope_close(fd) != 0) {
        scope_perror("scope_fopen failed");
        return FALSE;
    }

    return TRUE;
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

    return serviceSetup(serviceName);
}

 
 /*
 * Configure the child mount namespace
 * - load into memory filter file content from the host
 * - switch the mount namespace to child
 * - setup /etc/profile file
 * - extract memory filter file /tmp/libscope.so 
 * - extract libscope.so to /tmp/libscope.so 
 * - patch the library
 * Returns status of operation 0 in case of success, other values in case of failure
 */
int
nsConfigure(pid_t pid)
{
    size_t filterFileSize = 0;
    char * scopeCfgFilterMem = NULL;
    int status = EXIT_FAILURE;

    scopeCfgFilterMem = loadFileIntoMem(&filterFileSize, getenv("SCOPE_FILTER_PATH"));
    if (scopeCfgFilterMem == NULL) {
        scope_fprintf(scope_stderr, "error: Filter file location (SCOPE_FILTER_PATH) was not defined\n");
        return status;
    }

    if (setNamespace(pid, "mnt") == FALSE) {
        scope_fprintf(scope_stderr, "setNamespace mnt failed\n");
        goto cleanupMem;
    }

    // Extract filter file 
    if (extractMemToChildNamespace(scopeCfgFilterMem, filterFileSize, "/tmp/scope_filter.yml", 0664) == FALSE) {
        scope_fprintf(scope_stderr, "extract filter to child namespace failed\n");
        goto cleanupMem;
    }

    // Setup /etc/profile
    if (setupProfile() == FALSE) {
        scope_fprintf(scope_stderr, "setupProfile failed\n");
        goto cleanupMem;
    }

    // Extract libscope.so
    if (libdirExtractLibraryTo(LIBSCOPE_IN_CHILD_NS)) {
        scope_fprintf(scope_stderr, "extract libscope.so failed\n");
        goto cleanupMem;
    }

    // Patch the library
    if (loaderOpPatchLibrary(LIBSCOPE_IN_CHILD_NS) == PATCH_FAILED) {
        scope_fprintf(scope_stderr, "patch libscope.so failed\n");
        goto cleanupMem;
    }

    status = EXIT_SUCCESS;

cleanupMem:

    scope_munmap(scopeCfgFilterMem, filterFileSize);

    return status;
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
