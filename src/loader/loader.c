#define _GNU_SOURCE

#include <dlfcn.h>
#include <dirent.h>
#include <errno.h>
#include <elf.h>
#include <fcntl.h>
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#include "inject.h"
#include "loader.h"
#include "libdir.h"
#include "loaderutils.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "ns.h"
#include "patch.h"
#include "setup.h"
 
int g_log_level = CFG_LOG_WARN;

/* 
 * This avoids a segfault when code using shm_open() is compiled statically.
 * For some reason, compiling the code statically causes the __shm_directory()
 * function calls in librt.a to not reach the implementation in libpthread.a.
 * Implementing the function ourselves fixes this issue.
 *
 * See https://stackoverflow.com/a/47914897
 */
#ifndef  SHM_MOUNT
#define  SHM_MOUNT "/dev/shm/"
#endif
static const char  shm_mount[] = SHM_MOUNT;
const char *__shm_directory(size_t *len)
{
    if (len)
        *len = strlen(shm_mount);
    return shm_mount;
}

static int
attach(pid_t pid, char *scopeLibPath)
{
    char *exe_path = NULL;
    elf_buf_t *ebuf;

    if (geteuid()) {
        printf("error: --ldattach requires root\n");
        return EXIT_FAILURE;
    }

    if (getExePath(pid, &exe_path) == -1) {
        fprintf(stderr, "error: can't get path to executable for pid %d\n", pid);
        return EXIT_FAILURE;
    }

    if ((ebuf = getElf(exe_path)) == NULL) {
        free(exe_path);
        fprintf(stderr, "error: can't read the executable %s\n", exe_path);
        return EXIT_FAILURE;
    }

    if (is_static(ebuf->buf) == TRUE) {
        fprintf(stderr, "error: can't attach to the static executable: %s\nNote that the executable can be 'scoped' using the command 'scope run -- %s'\n", exe_path, exe_path);
        free(exe_path);
        freeElf(ebuf->buf, ebuf->len);
        return EXIT_FAILURE;
    }

    free(exe_path);
    freeElf(ebuf->buf, ebuf->len);

    printf("Attaching to process %d\n", pid);
    int ret = injectScope(pid, scopeLibPath);

    // done
    return ret;
}

/*
 * Given the ability to separate lib load and interposition
 * we enable a detach capability as well as 3 types
 * of attach cases. 4 commands in all.
 *
 * Load and attach:
 * libscope is not loaded. This case is
 * handled in function attach().
 *
 * First attach:
 * libscope is loaded and we are not interposing
 * functions, not scoped. This is the first time
 * libscope will have been attached.
 *
 * Reattach:
 * libscope is loaded, the process has been attached
 * in one form at least once previously.
 *
 * Detach:
 * libscope is loaded and we are interposing functions.
 * Remove all interpositions and stop scoping.
 */
static int
attachCmd(pid_t pid, bool attach)
{
    int fd, sfd, mfd;
    bool first_attach = FALSE;
    export_sm_t *exaddr;
    char buf[PATH_MAX] = {0};

    /*
     * The SM segment is used in the first attach case where
     * we've never been attached before. The segment is
     * populated with state including the address of the
     * reattach command in the lib. The segment is read
     * only and the size of the segment can't be modified.
     *
     * We use the presence of the segment to identify the
     * state of the lib. The segment is deleted when a
     * first attach is performed.
     */
    sfd = findFd(pid, SM_NAME);   // e.g. memfd:anon
    if (sfd > 0) {
        first_attach = TRUE;
    }

    /*
     * On command detach if the SM segment exists, we have never been
     * attached. Return and do not create the command file as it
     * will never be deleted until attached and then causes an
     * unintended detach.
     */
    if ((attach == FALSE) && (first_attach == TRUE)) {
        printf("Already detached from pid %d\n", pid);
        return EXIT_SUCCESS;
    }

    /*
     * Before executing any command, create and populate the dyn config file.
     * It is used for all cases:
     *   First attach: no attach command, include env vars, reload command
     *   Reattach: attach command = true, include env vars, reload command
     *   Detach: attach command = false, no env vars, no reload command
     */
    snprintf(buf, sizeof(buf), "%s/%s.%d",
                   DYN_CONFIG_CLI_DIR, DYN_CONFIG_CLI_PREFIX, pid);

    /*
     * Unlink a possible existing file before creating a new one
     * due to a fact that open will fail if the file is
     * sealed (still processed on library side).
     * File sealing is supported on tmpfs - /dev/shm (DYN_CONFIG_CLI_DIR).
     */
    unlink(buf);


    uid_t nsUid = nsInfoTranslateUid(pid);
    gid_t nsGid = nsInfoTranslateGid(pid);

    fd = nsFileOpen(buf, O_WRONLY|O_CREAT, nsUid, nsGid, geteuid(), getegid());
    if (fd == -1) {
        return EXIT_FAILURE;
    }

    if (fchmod(fd, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH) == -1) {
        perror("fchmod() failed");
        return EXIT_FAILURE;
    }

    /*
     * Ensuring that the process being operated on can remove
     * the dyn config file being created here.
     * In the case where a root user executes the command,
     * we need to change ownership of dyn config file.
     */

    if (geteuid() == 0) {
        uid_t euid = -1;
        gid_t egid = -1;

        if (getProcUidGid(pid, &euid, &egid) == -1) {
            fprintf(stderr, "error: osGetProcUidGid() failed\n");
            close(fd);
            return EXIT_FAILURE;
        }

        if (fchown(fd, euid, egid) == -1) {
            perror("fchown() failed");
            close(fd);
            return EXIT_FAILURE;
        }
    }

    if (first_attach == FALSE) {
        const char *cmd = (attach == TRUE) ? "SCOPE_CMD_ATTACH=true\n" : "SCOPE_CMD_ATTACH=false\n";
        if (write(fd, cmd, strlen(cmd)) <= 0) {
            perror("write() failed");
            close(fd);
            return EXIT_FAILURE;
        }
    }

    if (attach == TRUE) {
        int i;

        if (first_attach == TRUE) {
            printf("First attach to pid %d\n", pid);
        } else {
            printf("Reattaching to pid %d\n", pid);
        }

        /*
        * Reload the configuration during first attach & reattach if we want to apply
        * config from a environment variable
        * Handle SCOPE_CONF_RELOAD in first order because of "processReloadConfig" logic
        * is done in two steps:
        * - first - create a configuration based on path (default one or custom one)
        * - second - process env variables existing in the process (cfgProcessEnvironment)
        * We append rest of the SCOPE_ variables after since in this way we ovewrite the ones
        * which was set by cfgProcessEnvironment in second step.
        * TODO: Handle the file and env variables
        */
        char *scopeConfReload = getenv("SCOPE_CONF_RELOAD");
        if (!scopeConfReload) {
            dprintf(fd, "SCOPE_CONF_RELOAD=TRUE\n");
        } else {
            dprintf(fd, "SCOPE_CONF_RELOAD=%s\n", scopeConfReload);
        }

        for (i = 0; environ[i]; ++i) {
            size_t envLen = strlen(environ[i]);
            if ((envLen > 6) &&
                (strncmp(environ[i], "SCOPE_", 6) == 0) &&
                (!strstr(environ[i], "SCOPE_CONF_RELOAD"))) {
                    dprintf(fd, "%s\n", environ[i]);
            }
        }
    } else {
        printf("Detaching from pid %d\n", pid);
    }

    close(fd);

    int rc = EXIT_SUCCESS;

    if (first_attach == TRUE) {
        memset(buf, 0, PATH_MAX);
        // the only command we do in this case is first attach
        snprintf(buf, sizeof(buf), "/proc/%d/fd/%d", pid, sfd);
        if ((mfd = open(buf, O_RDONLY)) == -1) {
            close(sfd);
            perror("open");
            return EXIT_FAILURE;
        }

        if ((exaddr = mmap(NULL, sizeof(export_sm_t), PROT_READ,
                                 MAP_SHARED, mfd, 0)) == MAP_FAILED) {
            close(sfd);
            close(mfd);
            perror("mmap");
            return EXIT_FAILURE;
        }

        if (injectFirstAttach(pid, exaddr->cmdAttachAddr) == EXIT_FAILURE) {
            fprintf(stderr, "error: %s: you must have administrator privileges to run this command\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }

        //fprintf(stderr, "%s: %s 0x%lx\n", __FUNCTION__, buf, exaddr->cmdAttachAddr);

        close(sfd);
        close(mfd);

        if (munmap(exaddr, sizeof(export_sm_t))) {
            fprintf(stderr, "error: %s: unmapping in the the reattach command failed\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }
    }
        return rc;
}

int
cmdService(char *serviceName, pid_t nspid)
{
    uid_t eUid = geteuid();
    gid_t eGid = getegid();

    if (!serviceName) {
        return EXIT_FAILURE;
    }

    if (nspid == -1) {
        // Service on Host
        return setupService(serviceName, eUid, eGid);
    } else {
        // Service on Container
        pid_t nsContainerPid = 0;
        if ((nsInfoGetPidNs(nspid, &nsContainerPid) == TRUE) ||
            (nsInfoIsPidInSameMntNs(nspid) == FALSE)) {
            return nsService(nspid, serviceName);
        }
    }

    return EXIT_FAILURE;
}

int
cmdUnservice(pid_t nspid)
{
    if (nspid == -1) {
        // Service on Host
        return setupUnservice();
    } else {
        // Service on Container
        pid_t nsContainerPid = 0;
        if ((nsInfoGetPidNs(nspid, &nsContainerPid) == TRUE) ||
            (nsInfoIsPidInSameMntNs(nspid) == FALSE)) {
            return nsUnservice(nspid);
        }
    }

    return EXIT_FAILURE;
}

int
cmdConfigure(char *configFilterPath, pid_t nspid)
{
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    int status = EXIT_FAILURE;

    if (!configFilterPath) {
        return EXIT_FAILURE;
    }

    size_t configFilterSize = 0;
    void *confgFilterMem = setupLoadFileIntoMem(&configFilterSize, configFilterPath);
    if (confgFilterMem == NULL) {
        fprintf(stderr, "error: Load filter file into memory %s\n", configFilterPath);
        return EXIT_FAILURE;
    }

    if (nspid == -1) {
        // Configure on Host
        status = setupConfigure(confgFilterMem, configFilterSize, eUid, eGid);
    } else {
        // Configure on Container
        pid_t nsContainerPid = 0;
        if ((nsInfoGetPidNs(nspid, &nsContainerPid) == TRUE) ||
            (nsInfoIsPidInSameMntNs(nspid) == FALSE)) {
            status = nsConfigure(nspid, confgFilterMem, configFilterSize);
        }
    }

    if (confgFilterMem) {
        munmap(confgFilterMem, configFilterSize);
    }

    return status;
}

int
cmdUnconfigure(pid_t nspid)
{
    int status = EXIT_FAILURE;

    if (nspid == -1) {
        // Configure on Host
        status = setupUnconfigure();
    } else {
        // Configure on Container
        pid_t nsContainerPid = 0;
        if ((nsInfoGetPidNs(nspid, &nsContainerPid) == TRUE) ||
            (nsInfoIsPidInSameMntNs(nspid) == FALSE)) {
            status = nsUnconfigure(nspid);
        }
    }

    return status;
}

// Handle getfile command
int
cmdGetFile(char *paths, pid_t nspid)
{
    char *src_path;
    char *dest_path;
    pid_t nsContainerPid = 0;

    if ((src_path = strtok(paths, ",")) == NULL) {
        fprintf(stderr, "error: no source file path\n");
        return EXIT_FAILURE;
    }
    if ((dest_path = strtok(NULL, ",")) == NULL) {
        fprintf(stderr, "error: no destination file path\n");
        return EXIT_FAILURE;
    }

    if (nspid == -1) {
        fprintf(stderr, "error: getfile requires a namespace pid\n");
        return EXIT_FAILURE;
    }

    if ((nsInfoGetPidNs(nspid, &nsContainerPid) == FALSE) ||
        (nsInfoIsPidInSameMntNs(nspid) == TRUE)) {
        fprintf(stderr, "error: invalid namespace\n");
        return EXIT_FAILURE;
    }

    return nsGetFile(src_path, dest_path, nspid);
}

/*
 * Handle attach/detach commands.
 */
int
cmdAttach(bool ldattach, pid_t pid)
{
    int res = EXIT_FAILURE;
    char *scopeLibPath;
    char path[PATH_MAX] = {0};
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    uid_t nsUid = eUid;
    uid_t nsGid = eGid;
    elf_buf_t *scope_ebuf = NULL;
    elf_buf_t *ebuf = NULL;

    nsUid = nsInfoTranslateUid(pid);
    nsGid = nsInfoTranslateGid(pid);

    scope_ebuf = getElf("/proc/self/exe");
    if (!scope_ebuf) {
        perror("setenv");
        goto out;
    }

    // Extract and patch libscope from scope static. Don't attempt to extract from scope dynamic
    if (is_static(scope_ebuf->buf)) {
        if (libdirExtract(nsUid, nsGid)) {
            fprintf(stderr, "error: failed to extract library\n");
            goto out;
        }

        scopeLibPath = (char *)libdirGetPath();

        if (patchLibrary(scopeLibPath) == PATCH_FAILED) {
            fprintf(stderr, "error: failed to patch library\n");
            goto out;
        }
    } else {
        scopeLibPath = (char *)libdirGetPath();
    }

    if (access(scopeLibPath, R_OK|X_OK)) {
        fprintf(stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        goto out;
    }

    if (setenv("SCOPE_LIB_PATH", scopeLibPath, 1)) {
        perror("setenv(SCOPE_LIB_PATH) failed");
        goto out;
    }

    // Set SCOPE_PID_ENV
    setPidEnv(getpid());

    /*
     * Get the pid as int
     * Validate that the process exists
     * Perform namespace switch if required
     * Is the library currently loaded in the target process
     * Attach using ptrace or a dynamic command, depending on lib presence
     * Return at end of the operation
     */
    pid_t nsAttachPid = 0;

    snprintf(path, sizeof(path), "/proc/%d", pid);
    if (access(path, F_OK)) {
        printf("error: --ldattach, --lddetach PID not a current process: %d\n", pid);
        goto out;
    }

    uint64_t rc = findLibrary("libscope.so", pid, FALSE, NULL, 0);

    /*
    * If the expected process exists in different PID namespace (container)
    * we do a following switch context sequence:
    * - load static loader file into memory
    * - [optionally] save the configuration file pointed by SCOPE_CONF_PATH into memory
    * - switch the namespace from parent
    * - save previously loaded static loader into new namespace
    * - [optionally] save previously loaded configuration file into new namespace
    * - fork & execute static loader attach one more time with updated PID
    */
    if (nsInfoGetPidNs(pid, &nsAttachPid) == TRUE) {
        // must be root to switch namespace
        if (eUid) {
            printf("error: --ldattach requires root\n");
            goto out;
        }

        res = nsForkAndExec(pid, nsAttachPid, ldattach);
        goto out;
    /*
    * Process can exists in same PID namespace but in different mnt namespace
    * we do a simillar sequecne like above but without switching PID namespace
    * and updating PID.
    */
    } else if (nsInfoIsPidInSameMntNs(pid) == FALSE) {
        // must be root to switch namespace
        if (eUid) {
            printf("error: --ldattach requires root\n");
            goto out;
        }
        res =  nsForkAndExec(pid, pid, ldattach);
        goto out;
    }

    if (ldattach) {
        //uint64_t rc;
        char path[PATH_MAX];

        // create /dev/shm/${PID}.env when attaching, for the library to load
        snprintf(path, sizeof(path), "/attach_%d.env", pid);
        int fd = nsFileShmOpen(path, O_RDWR|O_CREAT, S_IRUSR|S_IRGRP|S_IROTH, nsUid, nsGid, eUid, eGid);
        if (fd == -1) {
            fprintf(stderr, "nsFileShmOpen failed\n");
            goto out;
        }

        // add the env vars we want in the library
        dprintf(fd, "SCOPE_LIB_PATH=%s\n", libdirGetPath());

        int i;
        for (i = 0; environ[i]; i++) {
            if (strlen(environ[i]) > 6 && strncmp(environ[i], "SCOPE_", 6) == 0) {
                dprintf(fd, "%s\n", environ[i]);
            }
        }

        // done
        close(fd);

        // rc from findLibrary
        if (rc == -1) {
            fprintf(stderr, "error: can't get path to executable for pid %d\n", pid);
            res = EXIT_FAILURE;
        } else if (rc == 0) {
            // proc exists, libscope does not exist, a load & attach
            res = attach(pid, scopeLibPath);
        } else {
            // libscope exists, a first time attach or a reattach
            res = attachCmd(pid, TRUE);
        }

        // remove the env var file
        snprintf(path, sizeof(path), "/attach_%d.env", pid);
        shm_unlink(path);
        goto out;
    } else {
        // pid & libscope need to exist before moving forward
        if (rc == -1) {
            fprintf(stderr, "error: pid %d does not exist\n", pid);
            goto out;
        } else if (rc == 0) {
            // proc exists, libscope does not exist.
            fprintf(stderr, "error: pid %d has never been attached\n", pid);
            goto out;
        }
        res =  attachCmd(pid, FALSE);
        goto out;
    }
out:
    if (ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
    }

    if (scope_ebuf) {
        freeElf(scope_ebuf->buf, scope_ebuf->len);
        free(scope_ebuf);
    }

    exit(res);
}

// Handle the run command
int
cmdRun(bool ldattach, bool lddetach, pid_t pid, pid_t nspid, int argc, char **argv)
{
    char *scopeLibPath;
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    uid_t nsUid = eUid;
    uid_t nsGid = eGid;
    elf_buf_t *scope_ebuf = NULL;
    elf_buf_t *ebuf = NULL;

    if (nspid != -1) {
        nsUid = nsInfoTranslateUid(nspid);
        nsGid = nsInfoTranslateGid(nspid);
    }

    scope_ebuf = getElf("/proc/self/exe");
    if (!scope_ebuf) {
        perror("setenv");
        goto out;
    }

    // Extract and patch libscope from scope static. Don't attempt to extract from scope dynamic
    if (is_static(scope_ebuf->buf)) {
        if (libdirExtract(nsUid, nsGid)) {
            fprintf(stderr, "error: failed to extract library\n");
            goto out;
        }

        scopeLibPath = (char *)libdirGetPath();

        if (patchLibrary(scopeLibPath) == PATCH_FAILED) {
            fprintf(stderr, "error: failed to patch library\n");
            goto out;
        }
    } else {
        scopeLibPath = (char *)libdirGetPath();
    }

    if (access(scopeLibPath, R_OK|X_OK)) {
        fprintf(stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        goto out;
    }

    if (setenv("SCOPE_LIB_PATH", scopeLibPath, 1)) {
        perror("setenv(SCOPE_LIB_PATH) failed");
        goto out;
    }

    // Set SCOPE_PID_ENV
    setPidEnv(getpid());


    /*
     * What kind of app are we trying to scope?
     */

    char *inferior_command = NULL;

    inferior_command = getpath(argv[0]); 
    if (!inferior_command) {
        fprintf(stderr,"scope could not find or execute command `%s`.  Exiting.\n", argv[0]);
        goto out;
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto out;
        }
    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto out;
        }
    }


    /*
     * If the app we want to scope is Dynamic
     * Just exec it with LD_PRELOAD and we're done
     */

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", scopeLibPath, 0) == -1) {
            perror("setenv");
            goto out;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto out;
        }

        goto out;
    }


    /*
     * If the app we want to scope is Static
     * There are determinations to be made
     */

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto out;
    }

    // Add a comment here
    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        goto out;
    }

    program_invocation_short_name = basename(argv[0]);

    // If it's not a Go app, we don't support it
    // so just exec it without scope
    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        goto out;
    }

    // If scope itself is static, we need to call scope dynamic
    // because we need to use dlopen
    if (is_static(scope_ebuf->buf)) {

        // Get scopedyn from the scope executable
        unsigned char *start; 
        size_t len;
        if ((len = getAsset(LOADER_FILE, &start)) == -1) {
            fprintf(stderr, "error: failed to retrieve loader\n");
            goto out;
        }

        // Patch the scopedyn executable on the heap (for musl support)
        if (patchLoader(start, nsUid, nsGid) == PATCH_FAILED) {
            fprintf(stderr, "error: failed to patch loader\n");
            goto out;
        }

#if 0   // Write scopedyn to /tmp for debugging
        int fd_dyn; 
        if ((fd_dyn = open("/tmp/scopedyn", O_CREAT | O_RDWR | O_TRUNC)) == -1) {
            perror("cmdRun:open");
            goto out;
        }
        int rc = write(fd_dyn, start, len);
        if (rc < len) {
            perror("cmdRun:write");
            goto out;
        }
        close(fd_dyn);
#endif
        // Write scopedyn to shared memory
        char path_to_fd[PATH_MAX];
        int fd = memfd_create("", 0);
        if (fd == -1) {
            perror("memfd_create");
            goto out;
        }
        ssize_t written = write(fd, start, len);
        if (written != g_scopedynsz) {
            fprintf(stderr, "error: failed to write loader to shm\n");
            goto out;
        }

        // Exec scopedyn from shared memory
        // Append "scopedyn" to argv first
        int execArgc = 0;
        char *execArgv[argc + 1];
        execArgv[execArgc++] = "scopedyn";
        for (int i = 0; i < argc; i++) {
            execArgv[execArgc++] = argv[i];
        }
        execArgv[execArgc++] = NULL;
        sprintf(path_to_fd, "/proc/self/fd/%i", fd);
        execve(path_to_fd, &execArgv[0], environ);
        perror("execve");

    // If scope itself is dynamic, dlopen libscope and sys_exec the app
    // and we're done
    } else {
        // we should never be here. we dont run scope dynamic.
    }
 
out:
    /*
     * Cleanup and exec the user app (where possible)
     * If there are errors, the app will run without scope
     */
    if (ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
    }

    if (scope_ebuf) {
        freeElf(scope_ebuf->buf, scope_ebuf->len);
        free(scope_ebuf);
    }

    if (inferior_command) {
        execve(inferior_command, &argv[0], environ);
        perror("execve");
    }

    exit(EXIT_FAILURE);
}

// Handle the install command
int
cmdInstall(char *rootdir)
{
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    uid_t nsUid = eUid;
    uid_t nsGid = eGid;

    // If rootdir is provided, extract the library into a separate namespace and return
    if (rootdir) {
        // Use pid 1 to locate ns fd
        if (nsInstall(rootdir, 1)) {
            fprintf(stderr, "error: failed to extract library\n");
            return EXIT_FAILURE;
        }
    // Install the library locally
    } else {
        if (setupInstall(nsUid, nsGid)) {
            fprintf(stderr, "error: failed to extract library\n");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
