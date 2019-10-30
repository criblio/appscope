#include <fcntl.h>
#include <sys/statvfs.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    struct statvfs vfs;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open(tmp_file_name, O_CREAT | O_RDONLY);
    
    if(f != EOF) {
        if(fstatvfs(f, &vfs) < 0) {
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