#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "ns.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "libver.h"
#include "libdir.h"
#include "setup.h"
#include "scopestdlib.h"

#define SCOPE_CRONTAB "* * * * * root /tmp/scope_att.sh\n"
#define SCOPE_START_SCRIPT "#! /bin/bash\nrm /etc/cron.d/scope_cron\n%s start -f < %s\nrm -- $0\n"
#define SCOPE_STOP_SCRIPT "#! /bin/bash\nrm /etc/cron.d/scope_cron\n%s stop -f\nrm -- $0\n"

// NS Action types
typedef enum {
    START = 0,
    STOP = 1,
} ns_action_t;

/*
 * Extract memory to specific output file.
 *
 * Returns TRUE in case of success, FALSE otherwise.
 */
static bool
extractMemToFile(char *inputMem, size_t inputSize, const char *outFile, mode_t outPermFlag, bool overwrite, uid_t nsUid, gid_t nsGid) {
    bool status = FALSE;
    int outFd;

    if (!scope_access(outFile, R_OK) && !overwrite) {
        return TRUE;
    }

    if ((outFd = nsFileOpenWithMode(outFile, O_RDWR | O_CREAT, outPermFlag, nsUid, nsGid, scope_geteuid(), scope_getegid())) == -1) {
        return status;
    }

    if (scope_ftruncate(outFd, inputSize) != 0) {
        goto cleanupDestFd;
    }

    char *dest = scope_mmap(NULL, inputSize, PROT_READ | PROT_WRITE, MAP_SHARED, outFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    scope_memcpy(dest, inputMem, inputSize);

    scope_munmap(dest, inputSize);

    status = TRUE;

cleanupDestFd:

    scope_fchmod(outFd, outPermFlag);

    scope_close(outFd);

    return status;
}

/*
 * Reassociate process identified with pid with a specific namespace described by ns.
 *
 * Returns TRUE if operation was success, FALSE otherwise.
 */
static bool
setNamespace(pid_t pid, const char *ns) {
    char nsPath[PATH_MAX] = {0};
    int nsFd;
    if (scope_snprintf(nsPath, sizeof(nsPath), "/proc/%d/ns/%s", pid, ns) < 0) {
        scope_perror("setNamespace: scope_snprintf failed");
        return FALSE;
    }

    if ((nsFd = scope_open(nsPath, O_RDONLY)) == -1) {
        scope_perror("setNamespace: scope_open failed");
        return FALSE;
    }

    if (scope_setns(nsFd, 0) != 0) {
        scope_perror("setNamespace: setns failed");
        scope_close(nsFd);
        return FALSE;
    }

    scope_close(nsFd);

    return TRUE;
}

/*
 * Joins the namespaces:
 * - child PID (optionally)
 * - mount namespace (mandatory).
 *
 * Returns TRUE if operation was success, FALSE otherwise.
 */
static bool
joinChildNamespace(pid_t hostPid, bool joinPidNs) {
    bool status = FALSE;
    size_t ldscopeSize = 0;
    size_t cfgSize = 0;
    mkdir_status_t dirRes = MKDIR_STATUS_ERR_OTHER;

    char path[PATH_MAX] = {0};

    uid_t nsUid = nsInfoTranslateUid(hostPid);
    gid_t nsGid = nsInfoTranslateGid(hostPid);

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
    *   In other words the calling process will not change it's own PID
    *   namespace
    * - mount namespace - allows to copy file(s) into a "child namespace"
    */
    if (joinPidNs && setNamespace(hostPid, "pid") == FALSE) {
        goto cleanupMem;
    }

    if (setNamespace(hostPid, "mnt") == FALSE) {
        goto cleanupMem;
    }

    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);

    /* For official version try to use /usr/lib/appscope */
    if (isDevVersion == FALSE) {
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "/usr/lib/appscope/%s/", loaderVersion);
        dirRes = libdirCreateDirIfMissing(path, 0755, nsUid, nsGid);
        if (dirRes <= MKDIR_STATUS_EXISTS) {
            scope_strncat(path, "ldscope", sizeof("ldscope"));
            status = extractMemToFile(ldscopeMem, ldscopeSize, path, 0775, isDevVersion, nsUid, nsGid);
        }
    }

    /* For dev version or if extract for official version try to use /tmp/appscope path */
    if (status == FALSE) {
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "/tmp/appscope/%s/", loaderVersion);
        dirRes = libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
        if (dirRes <= MKDIR_STATUS_EXISTS) {
            scope_strncat(path, "ldscope", sizeof("ldscope"));
            status = extractMemToFile(ldscopeMem, ldscopeSize, path, 0775, isDevVersion, nsUid, nsGid);
        }
    }

    /* Cleanup if extraction of ldscope fails */
    if (status == FALSE) {
        goto cleanupMem;
    }

    if (scopeCfgMem) {
        char scopeCfgPath[PATH_MAX] = {0};

        // extract scope.yml configuration
        scope_snprintf(scopeCfgPath, sizeof(scopeCfgPath), "/tmp/scope%d.yml", hostPid);
        status = extractMemToFile(scopeCfgMem, cfgSize, scopeCfgPath, 0664, TRUE, nsUid, nsGid);
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
 * Setup the service for specified child process
 * Returns status of operation SERVICE_STATUS_SUCCESS in case of success, other values in case of failure
 */
service_status_t
nsService(pid_t hostPid, const char *serviceName) {

    uid_t nsUid = nsInfoTranslateUid(hostPid);
    gid_t nsGid = nsInfoTranslateGid(hostPid);

    if (setNamespace(hostPid, "mnt") == FALSE) {
        return SERVICE_STATUS_ERROR_OTHER;
    }

    return setupService(serviceName, nsUid, nsGid);
}

/*
 * Remove scope from all services in a given namespace
 * Returns status of operation SERVICE_STATUS_SUCCESS in case of success, other values in case of failure
 */
service_status_t
nsUnservice(pid_t hostPid) {

    uid_t nsUid = nsInfoTranslateUid(hostPid);
    gid_t nsGid = nsInfoTranslateGid(hostPid);

    if (setNamespace(hostPid, "mnt") == FALSE) {
        return SERVICE_STATUS_ERROR_OTHER;
    }

    return setupUnservice(nsUid, nsGid);
}
 
 /*
 * Configure the child mount namespace
 * - switch the mount namespace to child
 * - configure the setup
 * Returns status of operation 0 in case of success, other values in case of failure
 */
int
nsConfigure(pid_t pid, void *scopeCfgFilterMem, size_t filterFileSize) {
    uid_t nsUid = nsInfoTranslateUid(pid);
    gid_t nsGid = nsInfoTranslateGid(pid);

    if (setNamespace(pid, "mnt") == FALSE) {
        scope_fprintf(scope_stderr, "setNamespace mnt failed\n");
        return EXIT_FAILURE;
    }

    if (setupConfigure(scopeCfgFilterMem, filterFileSize, nsUid, nsGid)) {
        scope_fprintf(scope_stderr, "setup child namespace failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

 /*
 * Unconfigure the child mount namespace
 * - switch the mount namespace to child
 * - unconfigure the setup
 * Returns status of operation 0 in case of success, other values in case of failure
 */
int
nsUnconfigure(pid_t pid) {
    uid_t nsUid = nsInfoTranslateUid(pid);
    gid_t nsGid = nsInfoTranslateGid(pid);

    if (setNamespace(pid, "mnt") == FALSE) {
        scope_fprintf(scope_stderr, "setNamespace mnt failed\n");
        return EXIT_FAILURE;
    }

    if (setupUnconfigure(nsUid, nsGid)) {
        scope_fprintf(scope_stderr, "setup child namespace failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

 /*
 * Check if libscope.so is loaded in specified PID
 * Returns TRUE if library is loaded, FALSE otherwise.
 */
static bool
isLibScopeLoaded(pid_t pid)
{
    char mapsPath[PATH_MAX] = {0};
    char buffer[9076];
    FILE *fd;
    bool status = FALSE;

    if (scope_snprintf(mapsPath, sizeof(mapsPath), "/proc/%d/maps", pid) < 0) {
        return status;
    }

    if ((fd = scope_fopen(mapsPath, "r")) == NULL) {
        return status;
    }

    while (scope_fgets(buffer, sizeof(buffer), fd)) {
        if (scope_strstr(buffer, "libscope.so")) {
            status = TRUE;
            break;
        }
    }

    scope_fclose(fd);
    return status;
}

 
 /*
 * Perform fork and exec which cause that direct children
 * effectively will join a new PID namespace(optionally) and mount namespace.
 *
 * Note:
 * Reassociating the PID namespace (setns CLONE_NEWPID) has somewhat different from 
 * other namespace types. Reassociating the calling thread with a PID namespace
 * changes only the PID namespace that subsequently created child processes of
 * the caller will be placed in. It does not change the PID namespace of the caller itself.
 *
 * Returns status of operation
 */
int
nsForkAndExec(pid_t parentPid, pid_t nsPid, char attachType)
{
    char *opStatus = "Detach";
    char *childOp = "-d";
    bool libLoaded = isLibScopeLoaded(parentPid);

    if (attachType == 'a') {
        childOp = "-a";
        opStatus = (libLoaded == FALSE) ? "Attach" : "Reattach";
    } else if (libLoaded == FALSE) {
        scope_fprintf(scope_stderr, "error: PID: %d has never been attached\n", parentPid);
        return EXIT_FAILURE; 
    }
    /*
    * TODO In case of Reattach/Detach - when libLoaded = TRUE
    * We only need the mount namespace to /dev/shm but currently ldscopedyn
    * also check the pid namespace
    */

    if (joinChildNamespace(parentPid, parentPid != nsPid) == FALSE) {
        scope_fprintf(scope_stderr, "error: joinChildNamespace failed\n");
        return EXIT_FAILURE; 
    }

    pid_t child = fork();
    if (child < 0) {
        scope_fprintf(scope_stderr, "error: fork() failed\n");
        return EXIT_FAILURE;
    } else if (child == 0) {
        char loaderInChildPath[PATH_MAX] = {0};

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

        const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
        bool isDevVersion = libverIsNormVersionDev(loaderVersion);

        scope_snprintf(loaderInChildPath, PATH_MAX, "/usr/lib/appscope/%s/ldscope", loaderVersion);
        if (scope_access(loaderInChildPath, R_OK) || isDevVersion) {
            scope_memset(loaderInChildPath, 0, PATH_MAX);
            scope_snprintf(loaderInChildPath, PATH_MAX, "/tmp/appscope/%s/ldscope", loaderVersion);
            if (scope_access(loaderInChildPath, R_OK)) {
                scope_fprintf(scope_stderr, "error: access ldscope failed\n");
                return EXIT_FAILURE;
            }
        }

        execArgv[execArgc++] = loaderInChildPath;
        execArgv[execArgc++] = childOp;
        execArgv[execArgc++] = nsAttachPidStr;

        return execve(loaderInChildPath, execArgv, environ);
    }
    // Parent
    int status;
    scope_waitpid(child, &status, 0);
    if (WIFEXITED(status)) {
        int exitChildStatus = WEXITSTATUS(status);
        if (exitChildStatus == 0) {
            scope_fprintf(scope_stderr, "%s to process %d in child process succeeded\n", opStatus, parentPid);
        } else {
            scope_fprintf(scope_stderr, "%s to process %d in child process failed\n", opStatus, parentPid);
        }
        return exitChildStatus;
    }
    scope_fprintf(scope_stderr, "error: %s failed() failed\n", opStatus);
    return EXIT_FAILURE;
}

/* Create the cron file
 *
 * When the start command is executed within a container we can't
 * set ns to that of a host process. Therefore, start a process in the
 * host context using crond. This process will run a script which will
 * run the start command in the context of the host. It should run once and
 * then clean up after itself.
 *
 * This should be called after the mnt namespace has been switched.
 */
static bool
createCron(const char *hostPrefixPath, const char *script) {
    int outFd;
    char buf[1024] = {0};
    char path[PATH_MAX] = {0};

    // Check access to cron.d directory
    if (scope_snprintf(path, sizeof(path), "%s/etc/cron.d", hostPrefixPath) < 0) {
        scope_perror("createCron: /etc/cron.d error: snprintf() failed\n");
        return FALSE;
    }

    if (scope_access(path, R_OK)) {
        scope_fprintf(scope_stderr, "createCron: error %s does not exist\n", path);
        return FALSE;
    }

    // Create the script to be executed by cron
    scope_memset(path, 0, PATH_MAX);
    if (scope_snprintf(path, sizeof(path), "%s/tmp/scope_att.sh", hostPrefixPath) < 0) {
        scope_perror("createCron: /tmp/scope_att.sh error: snprintf() failed\n");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    if ((outFd = scope_open(path, O_RDWR | O_CREAT, 0775)) == -1) {
        scope_perror("createCron: script path: scope_open failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    // Write cron action - scope start
    if (scope_snprintf(buf, sizeof(buf), script) < 0) {
        scope_perror("createCron: script: error: snprintf() failed\n");
        scope_close(outFd);
        return FALSE;
    }

    if (scope_write(outFd, buf, scope_strlen(buf)) == -1) {
        scope_perror("createCron: script: scope_write failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        scope_close(outFd);
        return FALSE;
    }

    if (scope_close(outFd) == -1) {
        scope_perror("createCron: script: scope_close failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    // Create the cron entry
    scope_memset(path, 0, PATH_MAX);
    if (scope_snprintf(path, sizeof(path), "%s/etc/cron.d/scope_cron", hostPrefixPath) < 0) {
        scope_perror("createCron: /etc/cron.d/scope_cron error: snprintf() failed\n");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    if ((outFd = scope_open(path, O_RDWR | O_CREAT, 0775)) == -1) {
        scope_perror("createCron: cron: scope_open failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    // crond will detect this file entry and run on its' next cycle
    if (scope_write(outFd, SCOPE_CRONTAB, sizeof(SCOPE_CRONTAB) - 1) == -1) {
        scope_perror("createCron: cron: scope_write failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        scope_close(outFd);
        return FALSE;
    }

    if (scope_close(outFd) == -1) {
        scope_perror("createCron: cron: scope_close failed");
        scope_fprintf(scope_stderr, "path: %s\n", path);
        return FALSE;
    }

    return TRUE;
}

 /*
 * Check if switching mount namespace is required.
 *
 * Returns status of operation, TRUE if switching is required, FALSE if not.
 */
static bool
switchMntNsRequired(const char *hostFsPrefix) {
    const char* const hostDir[] = {
        "/etc/cron.d/",
        "/usr/lib/",
        "/tmp/",
    };

    for (int i = 0; i < sizeof(hostDir)/sizeof(char*); ++i) {
        char path[PATH_MAX] = {0};
        if (scope_snprintf(path, sizeof(path), "%s%s", hostFsPrefix, hostDir[i]) < 0) {
            scope_perror("switchMntNsRequired: scope_snprintf failed");
            return TRUE;
        }
        if (scope_access(path, W_OK)) {
            return TRUE;
        }
    }
    return FALSE;
}

 /*
 * Performs switching to host mount namespace operation.
 *
 * Returns status of operation, TRUE in case of success, FALSE in case of failure.
 */
static bool
setHostMntNs(const char *hostFsPrefix) {
    int nsFd;
    char nsPath[PATH_MAX] = {0};

    if (scope_snprintf(nsPath, sizeof(nsPath), "%s/proc/1/ns/mnt", hostFsPrefix) < 0) {
        scope_perror("setHostMntNs: scope_snprintf CRIBL_EDGE_FS_ROOT failed");
        return FALSE;
    }

    if ((nsFd = scope_open(nsPath, O_RDONLY)) == -1) {
        scope_perror("setHostMntNs: scope_open failed: host fs is not mounted");
        return FALSE;
    }

    if (scope_setns(nsFd, CLONE_NEWNS) != 0) {
        scope_perror("setHostMntNs: setns failed");
        return FALSE;
    }

    scope_close(nsFd);

    return TRUE;
}

/*
 * Joins the host mount namespace to perform a Start or Stop action.
 * Required conditions:
 * - scope_filter must exist (if action is Start)
 * - scope must exist
 * TODO: unify it with joinChildNamespace
 * Returns TRUE if operation was success, FALSE otherwise.
 */
static bool
joinHostNamespace(ns_action_t action) {
    bool status = FALSE;
    size_t ldscopeSize = 0;
    size_t cfgSize = 0;
    size_t scopeSize = 0;
    char path[PATH_MAX] = {0};
    char hostPrefixPath[PATH_MAX] = {0};
    char hostFilterPath[PATH_MAX] = {0};
    char hostScopePath[PATH_MAX] = {0};
    char *scopeFilterCfgMem = NULL;
    char *scopeMem = NULL;
    char script[1024];

    if (scope_readlink("/proc/self/exe", path, sizeof(path) - 1) == -1) {
        return status;
    }

    // Load "ldscope" into memory
    char *ldscopeMem = setupLoadFileIntoMem(&ldscopeSize, path);
    if (ldscopeMem == NULL) {
        return status;
    }

    // Load "scope" into memory
    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);
    scope_memset(path, 0, PATH_MAX);
    scope_snprintf(path, PATH_MAX, "/usr/lib/appscope/%s/scope", loaderVersion);
    if ((scope_access(path, R_OK)) || (isDevVersion)) {
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "/tmp/appscope/%s/scope", loaderVersion);
        if (scope_access(path, R_OK)) {
            goto cleanupMem;
        }
    }
    scopeMem = setupLoadFileIntoMem(&scopeSize, path);
    if (scopeMem == NULL) {
        goto cleanupMem;
    }

    if (action == START) {
        // Load "filter file" into memory
        // First try to use env variable
        char *envFilterVal = getenv("SCOPE_FILTER");
        if (envFilterVal) {
            /*
            * If filter env was defined and wasn't disable 
            * the filter handling, try path interpretation
            */
            size_t envFilterLen = scope_strlen(envFilterVal);
            if (scope_strncmp(envFilterVal, "false", envFilterLen) && (!scope_access(envFilterVal, R_OK))) {
                scopeFilterCfgMem = setupLoadFileIntoMem(&cfgSize, envFilterVal);
            }
        } else {
            /*
            * Try to use defaults
            */
            if (!scope_access(SCOPE_FILTER_USR_PATH, R_OK)) {
                scopeFilterCfgMem = setupLoadFileIntoMem(&cfgSize, SCOPE_FILTER_USR_PATH);
            } else if (!scope_access(SCOPE_FILTER_TMP_PATH, R_OK)) {
                scopeFilterCfgMem = setupLoadFileIntoMem(&cfgSize, SCOPE_FILTER_TMP_PATH);
            }
        }
        if (scopeFilterCfgMem == NULL) {
            goto cleanupMem;
        }
    }

    // Get root fs path
    char *envFsRootVal = getenv("CRIBL_EDGE_FS_ROOT");
    if (envFsRootVal) {
        scope_snprintf(hostPrefixPath, PATH_MAX, "%s", envFsRootVal);
    } else {
        scope_strncpy(hostPrefixPath, "/hostfs", sizeof("/hostfs"));
    }

    // Verify access to host filesystem inside the container
    if (scope_access(hostPrefixPath, R_OK)) {
        goto cleanupMem;
    }

    /*
     * Reassociate current process to the host mount namespace
     * - allows to copy file(s) into the host fs
    */
    if (switchMntNsRequired(hostPrefixPath) == TRUE) {
        if (setHostMntNs(hostPrefixPath) == FALSE) {
            goto cleanupMem;
        }
        // if we switch mount namespace we do not use hostfs prefix
        scope_strcpy(hostPrefixPath, "");
    }

    // At this point we are using the host fs
    
    uid_t eUid = scope_geteuid();
    gid_t eGid = scope_getegid();

    /*
     * Ensure that we have the correct dest dir
     */
    scope_memset(path, 0, PATH_MAX);
    scope_snprintf(path, PATH_MAX, "%s/usr/lib/appscope/%s/", hostPrefixPath ,loaderVersion);
    mkdir_status_t res = libdirCreateDirIfMissing(path, 0755, eUid, eGid);
    if ((res > MKDIR_STATUS_EXISTS) || (isDevVersion)) {
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "%s/tmp/appscope/%s/", hostPrefixPath, loaderVersion);
        mkdir_status_t res = libdirCreateDirIfMissing(path, 0777, eUid, eGid);
        if (res > MKDIR_STATUS_EXISTS) {
            goto cleanupMem;
        }
        scope_snprintf(hostScopePath, PATH_MAX, "/tmp/appscope/%s/scope", loaderVersion);
    } else {
        scope_snprintf(hostScopePath, PATH_MAX, "/usr/lib/appscope/%s/scope", loaderVersion);
    }

    /*
     * Create a "ldscope" on the host
     */
    scope_strncat(path, "ldscope", sizeof("ldscope"));
    if ((status = extractMemToFile(ldscopeMem, ldscopeSize, path, 0775, isDevVersion, eUid, eGid)) == FALSE) {
        goto cleanupMem;
    }

    /*
     * Create a "scope" on the host
     */
    scope_memset(path, 0, PATH_MAX);
    scope_snprintf(path, PATH_MAX, "%s%s", hostPrefixPath, hostScopePath);
    if (extractMemToFile(scopeMem, scopeSize, path, 0775, isDevVersion, eUid, eGid) == FALSE) {
        goto cleanupMem;
    }

    if (action == START) {
        /*
         * Create a "filter file" on the host
         */
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "%s/usr/lib/appscope/scope_filter", hostPrefixPath);
        if ((status == extractMemToFile(scopeFilterCfgMem, cfgSize, path, 0664, TRUE, eUid, eGid)) == FALSE) {
            scope_memset(path, 0, PATH_MAX);
            scope_snprintf(path, PATH_MAX, "%s/tmp/appscope/scope_filter", hostPrefixPath);
            if ((status == extractMemToFile(scopeFilterCfgMem, cfgSize, hostFilterPath, 0664, TRUE, eUid, eGid)) == FALSE) {
                goto cleanupMem;
            }
            scope_strcpy(hostFilterPath, "/tmp/appscope/scope_filter");
        } else {
            scope_strcpy(hostFilterPath, "/usr/lib/appscope/scope_filter");
        }

        /*
         * Create a "cron script" on the host
         */
        scope_snprintf(script, sizeof(script), SCOPE_START_SCRIPT, hostScopePath, hostFilterPath);
        status = createCron(hostPrefixPath, script);
    } else {
        /*
         * Create a "cron script" on the host
         */
        scope_snprintf(script, sizeof(script), SCOPE_STOP_SCRIPT, hostScopePath);
        status = createCron(hostPrefixPath, script);
    }

cleanupMem:
    scope_munmap(ldscopeMem, ldscopeSize);

    if (scopeFilterCfgMem) {
        scope_munmap(scopeFilterCfgMem, cfgSize);
    }

    if (scopeMem) {
        scope_munmap(scopeMem, scopeSize);
    }

    return status;
}

 /*
 * Verify if current running process runs in the container.
 * Returns TRUE if process runs in the container FALSE otherwise
 */
static bool
isRunningInContainer(void) {
    struct stat st = {0};
    return (scope_stat("/proc/2/comm", &st) != 0) ? TRUE : FALSE;
}

 /*
 * Perform ldscope host start operation - this operation begins from container namespace.
 *
 * - switch namespace to host
 * - create cron entry with filter file
 *
 * Returns exit code of operation
 */
int
nsHostStart(void) {
    if (isRunningInContainer() == FALSE) {
        scope_fprintf(scope_stderr, "error: nsHostStart failed process is running on host\n");
        return EXIT_FAILURE;
    }
    scope_fprintf(scope_stdout, "Executing from a container, run the start command from the host\n");

    if (joinHostNamespace(START) == FALSE) {
        scope_fprintf(scope_stderr, "error: joinHostNamespace failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

 /*
 * Perform ldscope host stop operation - this operation begins from container namespace.
 *
 * - switch namespace to host
 * - create cron entry with filter file
 *
 * Returns exit code of operation
 */
int
nsHostStop(void) {
     if (isRunningInContainer() == FALSE) {
        scope_fprintf(scope_stderr, "error: nsHostStop failed process is running on host\n");
        return EXIT_FAILURE;
    }
    scope_fprintf(scope_stdout, "Executing from a container, run the stop command from the host\n");

    if (joinHostNamespace(STOP) == FALSE) {
        scope_fprintf(scope_stderr, "error: joinHostNamespace failed\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
