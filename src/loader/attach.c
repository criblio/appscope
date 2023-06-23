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

#include "attach.h"
#include "inject.h"
#include "libdir.h"
#include "loaderutils.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "ns.h"
#include "patch.h"
#include "setup.h"

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

/*
 * createDynConfigFile creates dynamic configuration file used to
 * followin operations:
 * - first attach
 * - detach
 * - reattach
 * Returns the file descriptor of the dynamic configuration file
 */
static int
createDynConfigFile(pid_t pid) {
    char buf[PATH_MAX] = {0};

    /*
     * Before executing any command, create and populate the dyn config file.
     * It is used for all cases:
     *   First attach: no attach command, include env vars, reload command
     *   Reattach: attach command = true, include env vars, reload command
     *   Detach: attach command = false, no env vars, no reload command
     */
    if (nsInfoIsPidInSameMntNs(pid) == FALSE) {
        pid_t nsPid = 0;
        nsInfoGetPidNs(pid, &nsPid);
        snprintf(buf, sizeof(buf), "/proc/%d/root/%s/%s.%d", pid, 
                DYN_CONFIG_CLI_DIR, DYN_CONFIG_CLI_PREFIX, nsPid);
    } else {
        snprintf(buf, sizeof(buf), "%s/%s.%d",
                DYN_CONFIG_CLI_DIR, DYN_CONFIG_CLI_PREFIX, pid);
    }
    
    /*
     * Unlink a possible existing file before creating a new one
     * due to a fact that open will fail if the file is
     * sealed (still processed on library side).
     * File sealing is supported on tmpfs - /dev/shm (DYN_CONFIG_CLI_DIR).
     */

    unlink(buf);

    uid_t nsUid = nsInfoTranslateUidRootDir("", pid);
    gid_t nsGid = nsInfoTranslateGidRootDir("", pid);

    return nsFileOpen(buf, O_WRONLY|O_CREAT, nsUid, nsGid, geteuid(), getegid());
}

int
load_and_attach(pid_t pid, char *scopeLibPath)
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
        fprintf(stderr, "error: can't read the executable %s\n", exe_path);
        free(exe_path);
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
 * handled in function load_and_attach().
 *
 * First attach:
 * libscope is loaded and we are not interposing
 * functions, not scoped. This is the first time
 * libscope will have been attached.
 *
 * Reattach:
 * libscope is loaded, the process has been attached
 * in one form at least once previously.
 */
int
attach(pid_t pid)
{
    int fd, sfd, mfd, i;
    bool first_attach = FALSE;
    export_sm_t *exaddr;

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

    fd = createDynConfigFile(pid);
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
        const char *cmd = "SCOPE_CMD_ATTACH=true\n";
        if (write(fd, cmd, strlen(cmd)) <= 0) {
            perror("write() failed");
            close(fd);
            return EXIT_FAILURE;
        }
    }

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

    close(fd);

    int rc = EXIT_SUCCESS;

    if (first_attach == TRUE) {
        char buf[PATH_MAX] = {0};
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
detach(pid_t pid)
{
    int fd, sfd;
    bool first_attach = FALSE;
    uint64_t rc;

    // Check process exists and that the libscope library exists in the process
    rc = findLibrary("libscope.so", pid, FALSE, NULL, 0);
    if (rc == -1) {
        printf("error: PID not a current process: %d\n", pid);
        return EXIT_FAILURE;
    } else if (rc == 0) {
        fprintf(stderr, "error: libscope does not exist in this process %d\n", pid);
        return EXIT_FAILURE;
    }

    // Perform detach by writing a dynamic config
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
    if (first_attach == TRUE) {
        printf("Already detached from pid %d\n", pid);
        return EXIT_SUCCESS; // TODO is this a success?
    }

    fd = createDynConfigFile(pid);
    if (fd == -1) {
        perror("open() failed");
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

    const char *cmd = "SCOPE_CMD_ATTACH=false\n";
    if (write(fd, cmd, strlen(cmd)) <= 0) {
        perror("write() failed");
        close(fd);
        return EXIT_FAILURE;
    }

    printf("Detaching from pid %d\n", pid);

    close(fd);

    return EXIT_SUCCESS;
}
