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

#include "utils.h"

#define EXE_TEST_FILE "/bin/cat"
#define DEFAULT_BIN_DIR "/tmp"
#define DEFAULT_BIN_FNAME "ldscopedyn"

extern unsigned char _binary___bin_linux_ldscopedyn_start;
extern unsigned char _binary___bin_linux_ldscopedyn_end;

void *g_fn;
void *dbgAddLine;

int main(int argc, char **argv, char **env)
{
    int rc;
    libscope_info info;
    char *verstr;
    char scopever[64];

    // compiler warning
    g_fn = NULL;
    dbgAddLine = NULL;

    printf("Starting scope static...first extract\n");

    strncpy(scopever, SCOPE_VER, strlen(SCOPE_VER) + 1);
    verstr = strtok(scopever, "-");
    if (asprintf(&info.path, "%s/libscope-%s/%s",
                 DEFAULT_BIN_DIR, verstr, DEFAULT_BIN_FNAME) == -1) {
        perror("ldscope:path");
        exit(EXIT_FAILURE);
    }

    if ((rc = extract_bin(&info,
                          &_binary___bin_linux_ldscopedyn_start,
                          &_binary___bin_linux_ldscopedyn_end)) == -1) {
        release_bin(&info);
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (rc == 1) {
        // new extraction of the bin
        printf("path to ldscope: %s\n", info.path);
        setup_loader(EXE_TEST_FILE, info.path);
    } else {
        setLdsoEnv(info.path);
    }

    execve(info.path, argv, environ);
    perror("execve");
    release_bin(&info);
    exit(EXIT_FAILURE);
}
