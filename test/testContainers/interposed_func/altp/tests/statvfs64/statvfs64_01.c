#include <fcntl.h>
#include <sys/statvfs.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;   
    struct statvfs64 fs;

    if(statvfs64("/", &fs) < 0) {
        TEST_ERROR();
    }

    FILE* fp = popen("stat / -c \"%s\"", "r");

    if (fp == NULL)  {
        TEST_ERROR();
    } else {
        char buf[BUFSIZ];
        unsigned int f_bsize = 0;
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