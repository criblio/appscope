#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <elf.h>
#include <libgen.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/utsname.h>
#include <dlfcn.h>

#include "../scopestdlib.h"
#include "../scopetypes.h"
#include "libdir.h"
#include "loaderop.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "ns.h"
#include "setup.h"
#include "loaderutils.h"
#include "inject.h"
#include "loader.h"
 
// maybe set this from a cmd line switch?
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
 * libscope is not loaded.  This case is
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

// Handle run, attach, detach commands.
// TODO: Split this up? Remove the argc/argv/env?
int
cmdRun(bool ldattach, bool lddetach, pid_t pid, pid_t nspid, int argc, char **argv, char **env)
{
    char *scopeLibPath;
    char path[PATH_MAX] = {0};
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    uid_t nsUid = eUid;
    uid_t nsGid = eGid;

    // Extract the library
    if (nspid != -1) {
        nsUid = nsInfoTranslateUid(nspid);
        nsGid = nsInfoTranslateGid(nspid);
    }

    if (libdirExtract(LIBRARY_FILE, nsUid, nsGid)) {
        fprintf(stderr, "error: failed to extract library\n");
        return EXIT_FAILURE;
    }

    // Set SCOPE_LIB_PATH
    scopeLibPath = (char *)libdirGetPath(LIBRARY_FILE);

    if (access(scopeLibPath, R_OK|X_OK)) {
        fprintf(stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        return EXIT_FAILURE;
    }

    if (setenv("SCOPE_LIB_PATH", scopeLibPath, 1)) {
        perror("setenv(SCOPE_LIB_PATH) failed");
        return EXIT_FAILURE;
    }

    // Set SCOPE_PID_ENV
    setPidEnv(getpid());

    /*
     * Attach & Detach
     *
     * Get the pid as int
     * Validate that the process exists
     * Perform namespace switch if required
     * Is the library currently loaded in the target process
     * Attach using ptrace or a dynamic command, depending on lib presence
     * Return at end of the operation
     */
    if (ldattach || lddetach) {
        pid_t nsAttachPid = 0;

        snprintf(path, sizeof(path), "/proc/%d", pid);
        if (access(path, F_OK)) {
            printf("error: --ldattach, --lddetach PID not a current process: %d\n", pid);
            return EXIT_FAILURE;
        }

        uint64_t rc = findLibrary("libscope.so", pid, FALSE);

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
                return EXIT_FAILURE;
            }
            return nsForkAndExec(pid, nsAttachPid, ldattach);
        /*
        * Process can exists in same PID namespace but in different mnt namespace
        * we do a simillar sequecne like above but without switching PID namespace
        * and updating PID.
        */
        } else if (nsInfoIsPidInSameMntNs(pid) == FALSE) {
            // must be root to switch namespace
            if (eUid) {
                printf("error: --ldattach requires root\n");
                return EXIT_FAILURE;
            }
            return nsForkAndExec(pid, pid, ldattach);
        }

        if (ldattach) {
            int ret;
            uint64_t rc;
            char path[PATH_MAX];

            // create /dev/shm/${PID}.env when attaching, for the library to load
            snprintf(path, sizeof(path), "/attach_%d.env", pid);
            int fd = nsFileShmOpen(path, O_RDWR|O_CREAT, S_IRUSR|S_IRGRP|S_IROTH, nsUid, nsGid, eUid, eGid);
            if (fd == -1) {
                fprintf(stderr, "nsFileShmOpen failed\n");
                return EXIT_FAILURE;
            }

            // add the env vars we want in the library
            dprintf(fd, "SCOPE_LIB_PATH=%s\n", libdirGetPath(LIBRARY_FILE));

            int i;
            for (i = 0; environ[i]; i++) {
                if (strlen(environ[i]) > 6 && strncmp(environ[i], "SCOPE_", 6) == 0) {
                    dprintf(fd, "%s\n", environ[i]);
                }
            }

            // done
            close(fd);

            rc = findLibrary("libscope.so", pid, FALSE);
            if (rc == -1) {
                fprintf(stderr, "error: can't get path to executable for pid %d\n", pid);
                ret = EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist, a load & attach
                ret = attach(pid, scopeLibPath);
            } else {
                // libscope exists, a first time attach or a reattach
                ret = attachCmd(pid, TRUE);
            }

            // remove the env var file
            snprintf(path, sizeof(path), "/attach_%d.env", pid);
            shm_unlink(path);
            return ret;
        } else if (lddetach) {
            // pid & libscope need to exist before moving forward
            if (rc == -1) {
                fprintf(stderr, "error: pid %d does not exist\n", pid);
                return EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist.
                fprintf(stderr, "error: pid %d has never been attached\n", pid);
                return EXIT_FAILURE;
            }
            return attachCmd(pid, FALSE);
        } else {
            fprintf(stderr, "error: attach or detach with invalid option\n");
            return EXIT_FAILURE;
        }
    }

    // Execute and scope the app
    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    void *handle = NULL;
    char *inferior_command = NULL;

    inferior_command = getpath(argv[0]); 
    //printf("%s:%d %s\n", __FUNCTION__, __LINE__, inferior_command);
    if (!inferior_command) {
        fprintf(stderr,"scope could not find or execute command `%s`.  Exiting.\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }
    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    // Dynamic executable path
    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", scopeLibPath, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }

        //printf("%s:%d %d: %s %s %s %s\n", __FUNCTION__, __LINE__,
        //       argc, argv[0], argv[1], argv[2], argv[3]);
        execve(inferior_command, &argv[0], environ);
        perror("execve");
        goto err;
    }

    // Static executable path
    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        execve(inferior_command, argv, environ);
    }

    program_invocation_short_name = basename(argv[0]);

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(inferior_command, &argv[0], environ);
    }


// is scope static or dynamic?

    elf_buf_t *scope_ebuf;
    scope_ebuf = getElf("/proc/self/exe");
    if (!scope_ebuf) {
            perror("setenv");
            goto err;
    }

    if (is_static(scope_ebuf->buf)) {
        // if scope is static, exec scopedyn
        
        char *execArgv[argc+2];
        int execArgc = 2;
        for (int i = 0; i < argc; i++) {
            execArgv[execArgc++] = argv[i++];
        }
        execArgv[execArgc++] = NULL;
        char *execname = "scopedyn";
        char *execz = "-z"; //todo remove
        execArgv[0] = execname;
        execArgv[1] = execz;
        execve("/home/sean/src/appscope-dev/bin/linux/x86_64/scopedyn", &execArgv[0], environ);
        perror("execve");

    } else {
        // otherwise if scope is dynamic do this
        if ((handle = dlopen(scopeLibPath, RTLD_LAZY)) == NULL) {
            goto err;
        }

        sys_exec = dlsym(handle, "sys_exec");
        if (!sys_exec) {
            goto err;
        }

        sys_exec(ebuf, inferior_command, argc, &argv[0], env);
    }

    /*
     * We should not return from sys_exec unless there was an error loading the static exec.
     * In this case, just start the exec without being scoped.
     * Was wondering if we should free the mapped elf image.
     * But, since we exec on failure to load, it doesn't matter.
     */
    execve(inferior_command, &argv[0], environ);



err:
    if (ebuf) free(ebuf);
    if (scope_ebuf) freeElf(scope_ebuf->buf, scope_ebuf->len);
    exit(EXIT_FAILURE);
}
