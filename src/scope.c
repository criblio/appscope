/*
 * Load and run static executables
 *
 * objcopy -I binary -O elf64-x86-64 -B i386 ./lib/linux/libscope.so ./lib/linux/libscope.o
 * gcc -Wall -g src/scope.c -ldl -lrt -o scope ./lib/linux/libscope.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <limits.h>
#include <errno.h>
#include <getopt.h>

#include "scopestdlib.h"
#include "fn.h"
#include "dbg.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include "utils.h"
#include "inject.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "os.h"

static void
showUsage(char *prog)
{
    scope_printf(
      "\n"
      "Cribl AppScope Dynamic Loader %s\n"
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n"
      "usage: %s [OPTIONS] --lib LIBRARY [--] EXECUTABLE [ARGS...]\n"
      "       %s [OPTIONS] --attach PID\n"
      "       %s [OPTIONS] --detach PID\n"
      "\n"
      "options:\n"
      "  -u, -h, --usage, --help  display this info\n"
      "  -a, --attach PID         attach to the specified process ID\n"
      "  -d, --detach PID         detach from the specified process ID\n"
      "\n"
      "Unless you are an AppScope developer, you are likely in the wrong place.\n"
      "See `scope` or `ldscope` instead.\n"
      "\n"
      "User docs are at https://appscope.dev/docs/. The project is hosted at\n"
      "https://github.com/criblio/appscope. Please direct feature requests and\n"
      "defect reports there.\n"
      "\n",
      SCOPE_VER, prog, prog, prog
    );
}

static int
attach(pid_t pid, char *scopeLibPath)
{
    char *exe_path = NULL;
    elf_buf_t *ebuf;

    if (scope_getuid()) {
        scope_printf("error: --attach requires root\n");
        return EXIT_FAILURE;
    }

    if (osGetExePath(pid, &exe_path) == -1) {
        scope_fprintf(scope_stderr, "error: can't get path to executable for pid %d\n", pid);
        return EXIT_FAILURE;
    }

    if ((ebuf = getElf(exe_path)) == NULL) {
        scope_free(exe_path);
        scope_fprintf(scope_stderr, "error: can't read the executable %s\n", exe_path);
        return EXIT_FAILURE;
    }

    if (is_static(ebuf->buf) == TRUE) {
        scope_fprintf(scope_stderr, "error: can't attach to the static executable: %s\nNote that the executable can be 'scoped' using the command 'scope run -- %s'\n", exe_path, exe_path);
        scope_free(exe_path);
        freeElf(ebuf->buf, ebuf->len);
        return EXIT_FAILURE;
    }

    scope_free(exe_path);
    freeElf(ebuf->buf, ebuf->len);

    scope_printf("Attaching to process %d\n", pid);
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
    sfd = osFindFd(pid, SM_NAME);   // e.g. memfd:scope_anon
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
        scope_printf("Already detached from pid %d\n", pid);
        return EXIT_SUCCESS;
    }

    /*
     * Before executing any command, create and populate the dyn config file.
     * It is used for all cases:
     *   First attach: no attach command, include env vars, reload command
     *   Reattach: attach command = true, include env vars, reload command
     *   Detach: attach command = false, no env vars, no reload command
     */
    scope_snprintf(buf, sizeof(buf), "%s/%s.%d",
                   DYN_CONFIG_CLI_DIR, DYN_CONFIG_CLI_PREFIX, pid);

    /*
     * Unlink a possible existing file before creating a new one
     * due to a fact that scope_open will fail if the file is
     * sealed (still processed on library side).
     * File sealing is supported on tmpfs - /dev/shm (DYN_CONFIG_CLI_DIR).
     */
    scope_unlink(buf);


    uid_t nsUid = nsInfoTranslateUid(pid);
    gid_t nsGid = nsInfoTranslateGid(pid);

    fd = nsFileOpen(buf, O_WRONLY|O_CREAT, nsUid, nsGid, scope_geteuid(), scope_getegid());
    if (fd == -1) {
        return EXIT_FAILURE;
    }

    if (scope_fchmod(fd, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH) == -1) {
        scope_perror("scope_fchmod() failed");
        return EXIT_FAILURE;
    }

    /*
     * Ensuring that the process being operated on can remove
     * the dyn config file being created here.
     * In the case where a root user executes the command,
     * we need to change ownership of dyn config file.
     */

    if (scope_getuid() == 0) {
        uid_t euid = -1;
        gid_t egid = -1;

        if (osGetProcUidGid(pid, &euid, &egid) == -1) {
            scope_fprintf(scope_stderr, "error: osGetProcUidGid() failed\n");
            scope_close(fd);
            return EXIT_FAILURE;
        }

        if (scope_fchown(fd, euid, egid) == -1) {
            scope_perror("scope_fchown() failed");
            scope_close(fd);
            return EXIT_FAILURE;
        }
    }

    if (first_attach == FALSE) {
        const char *cmd = (attach == TRUE) ? "SCOPE_CMD_ATTACH=true\n" : "SCOPE_CMD_ATTACH=false\n";
        if (scope_write(fd, cmd, scope_strlen(cmd)) <= 0) {
            scope_perror("scope_write() failed");
            scope_close(fd);
            return EXIT_FAILURE;
        }
    }

    if (attach == TRUE) {
        int i;

        if (first_attach == TRUE) {
            scope_printf("First attach to pid %d\n", pid);
        } else {
            scope_printf("Reattaching to pid %d\n", pid);
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
            scope_dprintf(fd, "SCOPE_CONF_RELOAD=TRUE\n");
        } else {
            scope_dprintf(fd, "SCOPE_CONF_RELOAD=%s\n", scopeConfReload);
        }

        for (i = 0; environ[i]; ++i) {
            size_t envLen = scope_strlen(environ[i]);
            if ((envLen > 6) &&
                (scope_strncmp(environ[i], "SCOPE_", 6) == 0) &&
                (!scope_strstr(environ[i], "SCOPE_CONF_RELOAD"))) {
                    scope_dprintf(fd, "%s\n", environ[i]);
            }
        }
    } else {
        scope_printf("Detaching from pid %d\n", pid);
    }

    scope_close(fd);

    int rc = EXIT_SUCCESS;

    if (first_attach == TRUE) {
        scope_memset(buf, 0, PATH_MAX);
        // the only command we do in this case is first attach
        scope_snprintf(buf, sizeof(buf), "/proc/%d/fd/%d", pid, sfd);
        if ((mfd = scope_open(buf, O_RDONLY)) == -1) {
            scope_close(sfd);
            scope_perror("open");
            return EXIT_FAILURE;
        }

        if ((exaddr = scope_mmap(NULL, sizeof(export_sm_t), PROT_READ,
                                 MAP_SHARED, mfd, 0)) == MAP_FAILED) {
            scope_close(sfd);
            scope_close(mfd);
            scope_perror("mmap");
            return EXIT_FAILURE;
        }

        if (injectFirstAttach(pid, exaddr->cmdAttachAddr) == EXIT_FAILURE) {
            scope_fprintf(scope_stderr, "error: %s: you must have administrator privileges to run this command\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }

        //scope_fprintf(scope_stderr, "%s: %s 0x%lx\n", __FUNCTION__, buf, exaddr->cmdAttachAddr);

        scope_close(sfd);
        scope_close(mfd);

        if (scope_munmap(exaddr, sizeof(export_sm_t))) {
            scope_fprintf(scope_stderr, "error: %s: unmapping in the the reattach command failed\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }
    }
        return rc;
}

// long aliases for short options
static struct option options[] = {
    {"help",    no_argument,       0, 'h'},
    {"usage",   no_argument,       0, 'u'},
    {"attach",  required_argument, 0, 'a'},
    {"detach",  required_argument, 0, 'd'},
    {0, 0, 0, 0}
};

int
main(int argc, char **argv, char **env)
{
    // process command line
    char *attachArg = 0;
    char *scopeLibPath;
    char attachType = 'u';

    for (;;) {
        int index = 0;
        int opt = getopt_long(argc, argv, "+:uha:d:", options, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'u':
            case 'h':
                showUsage(scope_basename(argv[0]));
                return EXIT_SUCCESS;
            case 'a':
                attachArg = optarg;
                attachType = 'a';
                break;
            case 'd':
                attachArg = optarg;
                attachType = 'd';
                break;
            case ':':
                // options missing their value end up here
                switch (optopt) {
                    default:
                        scope_fprintf(scope_stderr, "error: missing value for -%c option\n", optopt);
                        showUsage(scope_basename(argv[0]));
                        return EXIT_FAILURE;
                }
                break;
            case '?':
            default:
                scope_fprintf(scope_stderr, "error: invalid option: -%c\n", optopt);
                showUsage(scope_basename(argv[0]));
                return EXIT_FAILURE;
        }
    }

    // either --attach or an executable is required
    if (!attachArg && optind >= argc) {
        scope_fprintf(scope_stderr, "error: missing --attach, --detach or EXECUTABLE argument\n");
        showUsage(scope_basename(argv[0]));
        return EXIT_FAILURE;
    }

    // use --attach or --detach, ignore executable and args
    if (attachArg && optind < argc) {
        scope_fprintf(scope_stderr, "warning: ignoring EXECUTABLE argument with --attach, --detach option\n");
    }

    // SCOPE_LIB_PATH environment variable is required
    scopeLibPath = getenv("SCOPE_LIB_PATH");
    if (!scopeLibPath) {
        scope_fprintf(scope_stderr, "error: SCOPE_LIB_PATH must be set to point to libscope.so\n");
        return EXIT_FAILURE;
    }
    if (scope_access(scopeLibPath, R_OK|X_OK)) {
        scope_fprintf(scope_stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        return EXIT_FAILURE;
    }

    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    pid_t pid;
    void *handle = NULL;

    // Use dlsym to get addresses for everything in g_fn
    initFn();
    setPidEnv(scope_getpid());

    if (attachArg) {
        int pid = scope_atoi(attachArg);
        if (pid < 1) {
            scope_fprintf(scope_stderr, "error: invalid PID for --attach, --detach\n");
            return EXIT_FAILURE;
        }

        uint64_t rc = osFindLibrary("libscope.so", pid, FALSE);

        if (attachType == 'a') {
            int ret;
            char path[PATH_MAX];

            if (rc == -1) {
                scope_fprintf(scope_stderr, "error: can't get path to executable for pid %d\n", pid);
                ret = EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist, a load & attach
                ret = attach(pid, scopeLibPath);
            } else {
                // libscope exists, a first time attach or a reattach
                ret = attachCmd(pid, TRUE);
            }

            // remove the env var file
            scope_snprintf(path, sizeof(path), "/scope_attach_%d.env", pid);
            scope_shm_unlink(path);
            return ret;
        } else if (attachType == 'd') {
            // pid & libscope need to exist before moving forward
            if (rc == -1) {
                scope_fprintf(scope_stderr, "error: pid %d does not exist\n", pid);
                return EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist.
                scope_fprintf(scope_stderr, "error: pid %d has never been attached\n", pid);
                return EXIT_FAILURE;
            }
            return attachCmd(pid, FALSE);
        } else {
            scope_fprintf(scope_stderr, "error: attach or detach with invalid option\n");
            showUsage(scope_basename(argv[0]));
            return EXIT_FAILURE;
        }
    }

    char *inferior_command = getpath(argv[optind]);
    if (!inferior_command) {
        scope_fprintf(scope_stderr,"%s could not find or execute command `%s`.  Exiting.\n", argv[0], argv[optind]);
        exit(EXIT_FAILURE);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {

#if defined(__aarch64__)
    // We don't support go on ARM right now.
    // make sure it runs as if we were never here.
    execve(argv[optind], &argv[optind], environ);
#endif

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

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", scopeLibPath, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }
        
        pid = fork();
        if (pid == -1) {
            perror("fork");
            goto err;
        } else if (pid > 0) {
            int status;
            int ret;
            do {
                ret = waitpid(pid, &status, 0);
            } while (ret == -1 && errno == EINTR);

            if (WIFEXITED(status)) exit(WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        } else {
            execve(inferior_command, &argv[optind], environ);
            perror("execve");
            goto err;
        }
    }

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    // Static executable path
    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        execve(argv[0], argv, environ);
    }

    program_invocation_short_name = scope_basename(argv[1]);

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(argv[optind], &argv[optind], environ);
    }

    if ((handle = dlopen(scopeLibPath, RTLD_LAZY)) == NULL) {
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        goto err;
    }

    sys_exec(ebuf, inferior_command, argc-optind, &argv[optind], env);

    /*
     * We should not return from sys_exec unless there was an error loading the static exec.
     * In this case, just start the exec without being scoped.
     * Was wondering if we should free the mapped elf image.
     * But, since we exec on failure to load, it doesn't matter.
     */
    execve(argv[optind], &argv[optind], environ);

err:
    if (ebuf) scope_free(ebuf);
    exit(EXIT_FAILURE);
}
