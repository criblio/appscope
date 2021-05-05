#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "utils.h"

#define EXE_TEST_FILE "/bin/cat"

extern unsigned char _binary___bin_linux_ldscopedyn_start;
extern unsigned char _binary___bin_linux_ldscopedyn_end;

void *g_fn;
void *dbgAddLine;

int main(int argc, char **argv, char **env)
{
    libscope_info info;
    
    // compiler warning
    g_fn = NULL;
    dbgAddLine = NULL;

    printf("Starting scope static...first extract\n");

    if (extract_bin("ldscope", &info,
                    &_binary___bin_linux_ldscopedyn_start,
                    &_binary___bin_linux_ldscopedyn_end) != 0) {
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("path to ldscope: %s\n", info.path);
    setup_loader(EXE_TEST_FILE, info.path);

    execve(info.path, argv, environ);
    perror("execve");

    while (1) {
        sleep(1);
        printf(".");
    }
    return 0;
}
