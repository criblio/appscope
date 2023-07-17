#define _GNU_SOURCE

#include <fcntl.h>
#include <libgen.h>
#include <sched.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "attach.h"
#include "libver.h"
#include "libdir.h"
#include "ns.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "setup.h"
#include "scopetypes.h"

#define SCOPE_CRONTAB "* * * * * root /tmp/att%d.sh\n"
#define SCOPE_START_SCRIPT "#! /bin/bash\nrm /etc/cron.d/cron\n%s start -f < %s\nrm -- $0\n"
#define SCOPE_STOP_SCRIPT "#! /bin/bash\nrm /etc/cron.d/cron\n%s stop -f\nrm -- $0\n"
#define SCOPE_ATTACH_SCRIPT "#! /bin/bash\nrm /etc/cron.d/scope%d\n%s --ldattach %d\nrm -- $0\n"

// NS Action types
typedef enum {
    START = 0,
    STOP = 1,
} ns_action_t;

/*
 * Ensure that cron configuration is present in specified root path
 */
static bool
isCronConfigPresent(const char *rootPath) {
    const char* const cronFiles[] = {
        "/etc/cron.d",
        "/etc/crontab",
    };

    for (int i = 0; i < ARRAY_SIZE(cronFiles); ++i) {
        char path[PATH_MAX] = {0};
        if (snprintf(path, sizeof(path), "%s%s", rootPath, cronFiles[i]) < 0) {
            perror("isCronConfigPresent: error: snprintf() failed\n");
            return FALSE;
        }

        if (access(path, R_OK)) {
            return FALSE;
        }
    }

    return TRUE;
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
createCron(const char *hostPrefixPath, const char *script, pid_t pid) {
    int outFd;
    char buf[1024] = {0};
    char cronjob[1024] = {0};
    char path[PATH_MAX] = {0};


    // Create the /tmp/att.sh script to be executed by cron
    // We use a script so it can delete the cron file after it's run
    if (snprintf(path, sizeof(path), "%s/tmp/att%d.sh", hostPrefixPath, pid) < 0) {
        perror("createCron: /tmp/att.sh error: snprintf() failed\n");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }
    if ((outFd = open(path, O_RDWR | O_CREAT, 0775)) == -1) {
        perror("createCron: script path: open failed");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }
    // Write script contents
    if (snprintf(buf, sizeof(buf), script) < 0) {
        perror("createCron: script: error: snprintf() failed\n");
        close(outFd);
        return FALSE;
    }
    if (write(outFd, buf, strlen(buf)) == -1) {
        perror("createCron: script: write failed");
        fprintf(stderr, "path: %s\n", path);
        close(outFd);
        return FALSE;
    }
    if (close(outFd) == -1) {
        perror("createCron: script: close failed");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }

    // Create and write to the /etc/cron.d/<cronfile> entry
    memset(path, 0, PATH_MAX);
    if (snprintf(path, sizeof(path), "%s/etc/cron.d/scope%d", hostPrefixPath, pid) < 0) {
        perror("createCron: /etc/cron.d/cron error: snprintf() failed\n");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }
    if ((outFd = open(path, O_RDWR | O_CREAT, 0775)) == -1) {
        perror("createCron: cron: open failed");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }
    // Write cronfile contents
    if (snprintf(cronjob, sizeof(cronjob), SCOPE_CRONTAB, pid) < 0) {
        perror("createCron: sprintf() failed\n");
        return FALSE;
    }
    if (write(outFd, cronjob, strlen(cronjob)) == -1) {
        perror("createCron: cron: write failed");
        fprintf(stderr, "path: %s\n", path);
        close(outFd);
        return FALSE;
    }
    if (close(outFd) == -1) {
        perror("createCron: cron: close failed");
        fprintf(stderr, "path: %s\n", path);
        return FALSE;
    }

    return TRUE;
}

/*
 * Extract memory to specific output file.
 *
 * Returns TRUE in case of success, FALSE otherwise.
 */
static bool
extractMemToFile(char *inputMem, size_t inputSize, const char *outFile, mode_t outPermFlag, bool overwrite, uid_t nsUid, gid_t nsGid) {
    bool status = FALSE;
    int outFd;

    if (!access(outFile, R_OK) && !overwrite) {
        return TRUE;
    }

    if ((outFd = nsFileOpenWithMode(outFile, O_RDWR | O_CREAT, outPermFlag, nsUid, nsGid, geteuid(), getegid())) == -1) {
        return status;
    }

    if (ftruncate(outFd, inputSize) != 0) {
        goto cleanupDestFd;
    }

    char *dest = mmap(NULL, inputSize, PROT_READ | PROT_WRITE, MAP_SHARED, outFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    memcpy(dest, inputMem, inputSize);

    munmap(dest, inputSize);

    status = TRUE;

cleanupDestFd:
    fchmod(outFd, outPermFlag);

    close(outFd);

    return status;
}

/*
 * Reassociate process identified with pid with a specific namespace described by ns.
 * Returns TRUE if operation was success, FALSE otherwise.
 */
bool
nsSetNsRootDir(const char *rootdir, pid_t pid, const char *ns) {
    char nsPath[PATH_MAX] = {0};
    int nsFd;
    if (snprintf(nsPath, sizeof(nsPath), "%s/proc/%d/ns/%s", rootdir, pid, ns) < 0) {
        perror("nsSetNsRootDir: snprintf failed");
        return FALSE;
    }

    if ((nsFd = open(nsPath, O_RDONLY)) == -1) {
        perror("nsSetNsRootDir: open failed");
        return FALSE;
    }

    if (setns(nsFd, 0) != 0) {
        perror("nsSetNsRootDir: setns failed");
        close(nsFd);
        return FALSE;
    }

    close(nsFd);

    return TRUE;
}

#if 0
/*
 * Get the namespace file descriptor for the namespace you are currently in
 */
static int
getSelfNamespace(const char *ns) {
    char nsPath[PATH_MAX] = {0};
    int nsFd = -1;
    if (snprintf(nsPath, sizeof(nsPath), "/proc/self/ns/%s", ns) < 0) {
        perror("getSelfNamespace: snprintf failed");
        return nsFd;
    }

    if ((nsFd = open(nsPath, O_RDONLY)) == -1) {
        perror("getSelfNamespace: open failed");
        return nsFd;
    }

    return nsFd;
}
#endif

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
    size_t scopeSize = 0;
    size_t cfgSize = 0;
    mkdir_status_t dirRes = MKDIR_STATUS_ERR_OTHER;

    char path[PATH_MAX] = {0};

    uid_t nsUid = nsInfoTranslateUidRootDir("", hostPid);
    gid_t nsGid = nsInfoTranslateGidRootDir("", hostPid);

    if (readlink("/proc/self/exe", path, sizeof(path) - 1) == -1) {
        return status;
    }

    char *scopeMem = setupLoadFileIntoMem(&scopeSize, path);
    if (scopeMem == NULL) {
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
    * No need to provide a rootdir here. We are in the host and looking at a child pid.
    */
    if (joinPidNs && nsSetNsRootDir("", hostPid, "pid") == FALSE) {
        goto cleanupMem;
    }

    if (nsSetNsRootDir("", hostPid, "mnt") == FALSE) {
        goto cleanupMem;
    }

    char *workdirPath = getenv("SCOPE_HOST_WORKDIR_PATH");
    if (workdirPath) {
        memset(path, 0, PATH_MAX);
        libdirCreateDirIfMissing(workdirPath, 0777, nsUid, nsGid);
        snprintf(path, PATH_MAX, "%s/%s", workdirPath, "payloads");
        libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
        memset(path, 0, PATH_MAX);
        snprintf(path, PATH_MAX, "%s/%s", workdirPath, "cmd");
        libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
    }

    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);

    /* For official version try to use /usr/lib/appscope */
    // TODO this seems wrong? write to /usr/lib only if root (not if dev==true)
    if (isDevVersion == FALSE) {
        memset(path, 0, PATH_MAX);
        snprintf(path, PATH_MAX, "/usr/lib/appscope/%s/", loaderVersion);
        dirRes = libdirCreateDirIfMissing(path, 0755, nsUid, nsGid);
        if (dirRes <= MKDIR_STATUS_EXISTS) {
            strncat(path, "scope", sizeof(path) - 1);
            status = extractMemToFile(scopeMem, scopeSize, path, 0775, isDevVersion, nsUid, nsGid);
        }
    }

    /* For dev version or if extract for official version try to use /tmp/appscope path */
    // TODO this seems wrong? write to /tmp only if nonroot (not if dev==false)
    if (status == FALSE) {
        memset(path, 0, PATH_MAX);
        snprintf(path, PATH_MAX, "/tmp/appscope/%s/", loaderVersion);
        dirRes = libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
        if (dirRes <= MKDIR_STATUS_EXISTS) {
            strncat(path, "scope", sizeof(path) - 1);
            status = extractMemToFile(scopeMem, scopeSize, path, 0775, isDevVersion, nsUid, nsGid);
        }
    }

    /* Cleanup if extraction of scope fails */
    if (status == FALSE) {
        goto cleanupMem;
    }

    if (scopeCfgMem) {
        char scopeCfgPath[PATH_MAX] = {0};

        // extract scope.yml configuration
        if (workdirPath) {
            snprintf(scopeCfgPath, sizeof(scopeCfgPath), "%s/scope.yml", workdirPath);
        } else {
            snprintf(scopeCfgPath, sizeof(scopeCfgPath), "/tmp/scope%d.yml", hostPid);
        }
        status = extractMemToFile(scopeCfgMem, cfgSize, scopeCfgPath, 0664, TRUE, nsUid, nsGid);
        // replace the SCOPE_CONF_PATH with namespace path
        setenv("SCOPE_CONF_PATH", scopeCfgPath, 1);
    }   

cleanupMem:
    munmap(scopeMem, scopeSize);

    if (scopeCfgMem) {
        munmap(scopeCfgMem, cfgSize);
    }

    return status;
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

    if (snprintf(mapsPath, sizeof(mapsPath), "/proc/%d/maps", pid) < 0) {
        return status;
    }

    if ((fd = fopen(mapsPath, "r")) == NULL) {
        return status;
    }

    while (fgets(buffer, sizeof(buffer), fd)) {
        if (strstr(buffer, "libscope.so")) {
            status = TRUE;
            break;
        }
    }

    fclose(fd);
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
nsForkAndExec(pid_t parentPid, pid_t nsPid, bool ldattach)
{
    char *opStatus = "Detach";
    char *childOp = "-d";
    bool libLoaded = isLibScopeLoaded(parentPid);

    if (ldattach) {
        childOp = "-a";
        opStatus = (libLoaded == FALSE) ? "Attach" : "Reattach";
    } else if (libLoaded == FALSE) {
        fprintf(stderr, "error: PID: %d has never been attached\n", parentPid);
        return EXIT_FAILURE; 
    }
    /*
    * TODO In case of Reattach/Detach - when libLoaded = TRUE
    * We only need the mount namespace to /dev/shm but currently scope
    * also check the pid namespace
    */

    if (joinChildNamespace(parentPid, parentPid != nsPid) == FALSE) {
        fprintf(stderr, "error: joinChildNamespace failed\n");
        return EXIT_FAILURE; 
    }

    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "error: fork() failed\n");
        return EXIT_FAILURE;
    } else if (child == 0) {
        char loaderInChildPath[PATH_MAX] = {0};

        // Child
        char *nsAttachPidStr = NULL;
        if (asprintf(&nsAttachPidStr, "%d", nsPid) <= 0) {
            perror("error: asprintf() failed\n");
            return EXIT_FAILURE;
        }
        int execArgc = 0;
        char **execArgv = calloc(4, sizeof(char *));
        if (!execArgv) {
            fprintf(stderr, "error: calloc() failed\n");
            return EXIT_FAILURE;
        }

        const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
        bool isDevVersion = libverIsNormVersionDev(loaderVersion);

        snprintf(loaderInChildPath, PATH_MAX, "/usr/lib/appscope/%s/scope", loaderVersion);
        if (access(loaderInChildPath, R_OK) || isDevVersion) {
            memset(loaderInChildPath, 0, PATH_MAX);
            snprintf(loaderInChildPath, PATH_MAX, "/tmp/appscope/%s/scope", loaderVersion);
            if (access(loaderInChildPath, R_OK)) {
                fprintf(stderr, "error: access scope failed\n");
                free(execArgv);
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
    waitpid(child, &status, 0);
    if (WIFEXITED(status)) {
        int exitChildStatus = WEXITSTATUS(status);
        if (exitChildStatus == 0) {
            fprintf(stderr, "%s to process %d in child process succeeded\n", opStatus, parentPid);
        } else {
            fprintf(stderr, "%s to process %d in child process failed\n", opStatus, parentPid);
        }
        return exitChildStatus;
    }
    fprintf(stderr, "error: %s failed() failed\n", opStatus);
    return EXIT_FAILURE;
}

// TODO migrate to cmdAttach
// Change to the target mount namespace
// Extract scope into that namespace
// Extract a cron script into that namespace to run `scope --ldattach [pid]`
// Optionally copy a config into the target mnt ns
int
nsAttach(pid_t pid, const char *rootDir)
{
    pid_t hostPid = pid;
    int ret = EXIT_SUCCESS;
    size_t cfgSize = 0;
    char script[1024];
    char *scopePath = NULL;
    char scopeCfgPath[PATH_MAX] = {0};
    char scopeCmd[PATH_MAX] = {0};
    char path[PATH_MAX] = {0};
    unsigned char *file = NULL;
    size_t file_len;

    snprintf(path, PATH_MAX, "%s", rootDir);

    /*
    * Check if cron configuration exists in specified root directory
    */
    if (isCronConfigPresent(path) == FALSE) {
        /*
        * Fallthrough scenario
        * Try to modify the specified root directory to point host directory
        * E.g. In case rootDir is equal /hostfs/proc/<pid>/root/ and it points
        * to another container
        */

        /*
        * TODO: make this more robust based on full path with "host pid"
        */
        char *pch = strstr(path, "/proc/");
        if (!pch) {
            fprintf(stderr, "error: nsAttach: failed to find cron configuration in %s\n", path);
            return EXIT_FAILURE;
        }
        if (sscanf(pch, "/proc/%d/root", &hostPid) != 1) {
            fprintf(stderr, "error: nsAttach: cannot find host pid in %s\n", pch);
            return EXIT_FAILURE;
        }

        // Try to access the hostfs path
        path[pch - path] = 0;

        if (isCronConfigPresent(path) == FALSE) {
            fprintf(stderr, "error: nsAttach: failed to find cron configuration in %s\n", path);
            return EXIT_FAILURE;
        }
    }


    // Configuration is optionally loaded into memory while in the origin namespace
    char *scopeCfgMem = setupLoadFileIntoMem(&cfgSize, getenv("SCOPE_CONF_PATH"));

    // Extract library from scope binary into memory while in the origin namespace
    if ((file_len = getAsset(STATIC_LOADER_FILE, &file)) == -1) {
        fprintf(stderr, "nsAttach getAsset failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    // Switch to rootdir mnt namespace
    uid_t nsUid = nsInfoTranslateUidRootDir(path, 1);
    gid_t nsGid = nsInfoTranslateGidRootDir(path, 1);

    // Use pid 1 to locate ns fd
    if (nsSetNsRootDir(rootDir, 1, "mnt") == FALSE) {
        fprintf(stderr, "nsAttach nsSetNsRootDir mnt failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    // Extract the scope loader
    if (libdirExtract(file, file_len, nsUid, nsGid, STATIC_LOADER_FILE)) {
        fprintf(stderr, "nsAttach extract failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    // Create script and cron job to perform the attach 

    scopePath = (char *)libdirGetPath(STATIC_LOADER_FILE);
    if (!scopePath) {
        fprintf(stderr, "error: nsAttach: failed to get loader path\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    if (!strncpy(scopeCmd, scopePath, sizeof(scopeCmd))) {
        perror("error: nsAttach: strncpy failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    // Set up the working directory in the target ns, to match the origin ns working directory
    char *workdirPath = getenv("SCOPE_HOST_WORKDIR_PATH");
    if (workdirPath) {
        memset(path, 0, PATH_MAX);
        libdirCreateDirIfMissing(workdirPath, 0777, nsUid, nsGid);
        memset(path, 0, PATH_MAX);
        snprintf(path, PATH_MAX, "%s/%s", workdirPath, "payloads");
        libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
        memset(path, 0, PATH_MAX);
        snprintf(path, PATH_MAX, "%s/%s", workdirPath, "cmd");
        libdirCreateDirIfMissing(path, 0777, nsUid, nsGid);
    } else {
        workdirPath = "/tmp";
    }

    // If a config was loaded into memory, extract it into the target ns and update
    // the scope command to include the config env var
    if (scopeCfgMem) {
        if (snprintf(scopeCfgPath, sizeof(scopeCfgPath), "%s/scope%d.yml", workdirPath, hostPid) < 0) {
            perror("error: nsAttach: snprintf() failed\n");
            ret = EXIT_FAILURE;
            goto out;
        }
        if (!extractMemToFile(scopeCfgMem, cfgSize, scopeCfgPath, 0664, TRUE, nsUid, nsGid)) {
            fprintf(stderr, "error: nsAttach: failed to extract config to target ns\n");
            ret = EXIT_FAILURE;
            goto out;
        }
        if (snprintf(scopeCmd, sizeof(scopeCmd), "SCOPE_CONF_PATH=%s %s", scopeCfgPath, scopePath) < 0) {
            perror("error: nsAttach: snprintf() failed\n");
            ret = EXIT_FAILURE;
            goto out;
        }
    }

    if (snprintf(script, sizeof(script), SCOPE_ATTACH_SCRIPT, hostPid, scopeCmd, hostPid) < 0) {
        perror("error: nsAttach: sprintf() failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

    if (createCron("", script, hostPid) == FALSE) {
        perror("error: nsAttach: createCronFile() failed\n");
        ret = EXIT_FAILURE;
        goto out;
    }

out:
    if (file) munmap(file, file_len);
    if (scopeCfgMem) munmap(scopeCfgMem, cfgSize);

    return ret;
}

