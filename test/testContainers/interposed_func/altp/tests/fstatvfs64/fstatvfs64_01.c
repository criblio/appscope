#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/statvfs.h>

#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    struct statvfs64 vfs;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open64(tmp_file_name, O_CREAT | O_RDONLY);
    
    if(f != EOF) {
        if(fstatvfs64(f, &vfs) < 0) {
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

            if(vfs.f_bsize != f_bsize) {
                TEST_ERROR();
            }
        }
        
        if(close(f) == EOF) {
            TEST_ERROR();
        }
        
    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}