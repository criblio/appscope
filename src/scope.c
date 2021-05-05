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
#include <string.h>
#include <elf.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <limits.h>
#include <errno.h>

#include "fn.h"
#include "dbg.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include "os.h"
#include "utils.h"
#include "inject.h"

#define DEVMODE 0
#define __NR_memfd_create   319
#define _MFD_CLOEXEC		0x0001U
#define PARENT_PROC_NAME "start_scope"
#define GO_ENV_VAR "GODEBUG"
#define GO_ENV_SERVER_VALUE "http2server"
#define GO_ENV_CLIENT_VALUE "http2client"

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

__attribute__((weak)) uint64_t _dl_argv;

static void
print_usage(char *prog, libscope_info *info, int argc, char **argv) {
    void (*__scope_main)(void);
    void *handle = NULL;

    __scope_main = dlsym(RTLD_NEXT, "__scope_main");
    if (!__scope_main) {
        if ((handle = dlopen(info->path, RTLD_LAZY)) == NULL) {
            fprintf(stderr, "handle error: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }

        __scope_main = dlsym(handle, "__scope_main");
        if (!__scope_main) {
            fprintf(stderr, "symbol error: %s from %s\n", dlerror(), info->path);
            exit(EXIT_FAILURE);
        }
    }

    printf("usage: %s command [args]\n", prog);
    if (argc == 2) {
        strncpy(argv[1], "all", strlen(argv[1]));
    }

    __scope_main();
}

// If possible, we want to set GODEBUG=http2server=0,http2client=0
// This tells go not to upgrade to http2, which allows
// our http1 protocol capture stuff to do it's thing.
// We consider this temporary, because when we support http2
// it will not be necessary.
static void
setGoHttpEnvVariable(void)
{
    char *cur_val = getenv(GO_ENV_VAR);

    // If GODEBUG isn't set, try to set it to http2server=0,http2client=0
    if (!cur_val) {
        if (setenv(GO_ENV_VAR, GO_ENV_SERVER_VALUE "=0," GO_ENV_CLIENT_VALUE "=0", 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        return;
    }

    // GODEBUG is set.
    // If http2server wasn't specified, let's append ",http2server=0"
    if (!strstr(cur_val, GO_ENV_SERVER_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_SERVER_VALUE) == -1)) {
            perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) free(new_val);
    }

    cur_val = getenv(GO_ENV_VAR);

    // If http2client wasn't specified, let's append ",http2client=0"
    if (!strstr(cur_val, GO_ENV_CLIENT_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_CLIENT_VALUE) == -1)) {
            perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) free(new_val);
    }
}

int
main(int argc, char **argv, char **env)
{
    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    pid_t pid;
    void *handle = NULL;
    bool attach = FALSE;
    libscope_info info;

    // Use dlsym to get addresses for everything in g_fn
    initFn();
    setPidEnv(getpid());

    if (argc == 3 && (strncmp(argv[1], "--attach", 8) == 0)) {
        attach = TRUE;
    }

    if (extract_bin("libscope", &info,
                    &_binary___lib_linux_libscope_so_start,
                    &_binary___lib_linux_libscope_so_end) != 0) {
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    //check command line arguments 
    char *scope_cmd = argv[0];
    if ((argc < 2) || ((argc == 2) && (strncmp(argv[1], "--help", 6) == 0))) {
        print_usage(scope_cmd, &info, argc, argv);
        exit(EXIT_FAILURE);
    }

    if (attach) {
        int pid = atoi(argv[2]);
        printf("Attaching to process %d\n", pid);
        injectScope(pid, info.path);
        return 0;
    }

    char *inferior_command = getpath(argv[1]);
    if (!inferior_command) {
        fprintf(stderr,"%s could not find or execute command `%s`.  Exiting.\n", scope_cmd, argv[1]);
        exit(EXIT_FAILURE);
    }
    argv[1] = inferior_command; // update args with resolved inferior_command

    // before processing, try to set SCOPE_EXEC_PATH for execve
    char *sep;
    if (osGetExePath(&sep) == 0) {
        // doesn't overwrite an existing env var if already set
        setenv("SCOPE_EXEC_PATH", sep, 0);
        free(sep);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }

        //setGoHttpEnvVariable();
        setEnvVariable(GO_ENV_VAR, GO_ENV_SERVER_VALUE "=0");
        setEnvVariable(GO_ENV_VAR, GO_ENV_CLIENT_VALUE "=0");

    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", info.path, 0) == -1) {
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

            release_bin(&info);
            if (WIFEXITED(status)) exit(WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        } else {
            execve(inferior_command, &argv[1], environ);
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

    program_invocation_short_name = basename(argv[1]);

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(argv[1], &argv[1], environ);
    }

    if ((handle = dlopen(info.path, RTLD_LAZY)) == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }

    release_bin(&info);

    sys_exec(ebuf, inferior_command, argc, argv, env);

    return 0;
err:
    release_bin(&info);
    if (ebuf) free(ebuf);
    exit(EXIT_FAILURE);
}
