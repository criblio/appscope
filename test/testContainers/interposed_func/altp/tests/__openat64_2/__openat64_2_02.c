#include <fcntl.h>
#include <sys/types.h>

#include "test_utils.h"

int __openat64_2(int, const char *, int);

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);
    
    for(i = 0; i< TEST_COUNT; i ++) {
        int dirfd = open64(tmp_dir_name, O_RDONLY);

        int f = open64(tmp_file_name, O_CREAT | O_WRONLY);
        
        if(f != EOF) {
            if(close(f) == EOF) {
                TEST_ERROR();
            }

            f = __openat64_2(dirfd, "file",  O_WRONLY);
            
            if(f != EOF) {
                if(write(f, TEST_MSG, sizeof(TEST_MSG)) != sizeof(TEST_MSG)) {
                    TEST_ERROR();
                }

                if(close(f) == EOF) {
                    TEST_ERROR();
                }

                if(close(dirfd) == EOF) {
                    TEST_ERROR();
                }  
            } else {
                TEST_ERROR();
            }

            unlink(tmp_file_name);
        } else {
            TEST_ERROR();
            break;
        }
    }

    REMOVE_TMP_DIR();

    return test_result;
}