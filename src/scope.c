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
      "\n"
      "options:\n"
      "  -u, -h, --usage, --help  display this info\n"
      "  -a, --attach PID         attach to the specified process ID\n"
      "\n"
      "Unless you are an AppScope developer, you are likely in the wrong place.\n"
      "See `scope` or `ldscope` instead.\n"
      "\n"
      "User docs are at https://appscope.dev/docs/. The project is hosted at\n"
      "https://github.com/criblio/appscope. Please direct feature requests and\n"
      "defect reports there.\n"
      "\n",
      SCOPE_VER, prog, prog
    );
}

// long aliases for short options
static struct option options[] = {
    {"help",    no_argument,       0, 'h'},
    {"usage",   no_argument,       0, 'u'},
    {"attach",  required_argument, 0, 'a'},
    {0, 0, 0, 0}
};

int
main(int argc, char **argv, char **env)
{
    // process command line
    char *attachArg = 0;
    for (;;) {
        int index = 0;
        int opt = getopt_long(argc, argv, "+:uha:", options, &index);
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
        scope_fprintf(scope_stderr, "error: missing --attach or EXECUTABLE argument\n");
        showUsage(scope_basename(argv[0]));
        return EXIT_FAILURE;
    }

    // use --attach, ignore executable and args
    if (attachArg && optind < argc) {
        scope_fprintf(scope_stderr, "warning: ignoring EXECUTABLE argument with --attach option\n");
    }

    // SCOPE_LIB_PATH environment variable is required
    char* scopeLibPath = getenv("SCOPE_LIB_PATH");
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
        char *exe_path = NULL;
        elf_buf_t *ebuf;

        int pid = scope_atoi(attachArg);
        if (pid < 1) {
            scope_fprintf(scope_stderr, "error: invalid PID for --attach\n");
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

        // remove the config that `ldscope`
        char path[PATH_MAX];
        scope_snprintf(path, sizeof(path), "/scope_attach_%d.env", pid);
        scope_shm_unlink(path);

        // done
        return ret;
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

    return 0;
err:
    if (ebuf) scope_free(ebuf);
    exit(EXIT_FAILURE);
}
