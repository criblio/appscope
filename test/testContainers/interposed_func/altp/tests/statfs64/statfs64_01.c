#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/statfs.h>

#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;   
    struct statfs fs;

    if(statfs64("/", &fs) < 0) {
        TEST_ERROR();
    }

    FILE* fp = popen("stat / -c \"%s\"", "r");

    if (fp == NULL)  {
        TEST_ERROR();
    } else {
        char buf[BUFSIZ];
        int f_bsize = 0;
        size_t byte_count = fread(buf, 1, BUFSIZ - 1, fp);
        buf[byte_count] = 0;
        pclose(fp);

        sscanf(buf, "%d", &f_bsize);

        if(fs.f_bsize != f_bsize) {
            TEST_ERROR();
        }
    }

    return test_result;
}