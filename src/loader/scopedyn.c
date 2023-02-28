#define _GNU_SOURCE

#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "libdir.h"
#include "loaderutils.h"
#include "scopetypes.h"

int g_log_level = CFG_LOG_WARN;
unsigned long g_libscopesz;
unsigned long g_scopedynsz;

// This code is only intended to run static go apps
int
main(int argc, char** argv)
{
    elf_buf_t *ebuf = NULL;
    char *inferior_command = NULL;
    char *scopeLibPath = NULL;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    void *handle = NULL;

    inferior_command = getpath(argv[1]); // TODO assumption
    if (!inferior_command) {
        fprintf(stderr,"scope could not find or execute command `%s`.  Exiting.\n", argv[0]);
        goto out;
    }

    scopeLibPath = (char *)libdirGetPath();
    if (!scopeLibPath) {
        fprintf(stderr, "error: libdirGetPath\n");
        goto out;
    }

#if 0
    // The loader code sets these environment variables for us
    // If you run this executable without coming from the loader
    // You will need to uncomment this code
    if (setenv("SCOPE_LIB_PATH", scopeLibPath, 1)) {
        perror("setenv(SCOPE_LIB_PATH) failed");
        goto out;
    }

    // Set SCOPE_PID_ENV
    setPidEnv(getpid());

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

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto out;
    }
#endif

    program_invocation_short_name = basename(argv[0]);

    if ((handle = dlopen(scopeLibPath, RTLD_LAZY)) == NULL) {
        perror("dlopen");
        goto out;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        fprintf(stderr, "error: sysexec\n");
        goto out;
    }

    ebuf = getElf(inferior_command);
    if (!ebuf) {
        fprintf(stderr, "error: getElf\n");
        goto out;
    }

    sys_exec(ebuf, inferior_command, argc - 1, &argv[1], environ); // TODO assumption

out:
    // Cleanup and exec the user app (where possible)
    // If there are errors, the app will run without scope
    
    if (ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
    }

    if (inferior_command) {
        execve(inferior_command, &argv[0], environ);
        perror("execve");
    }

    exit(EXIT_FAILURE);
}













