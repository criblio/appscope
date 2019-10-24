#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>

#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);
    
    for(i = 0; i< 100; i ++) {
        int dirfd = open(tmp_dir_name, O_RDONLY);

        int f = open(tmp_file_name, O_CREAT | O_WRONLY);
        
        if(f != EOF) {
            if(close(f) == EOF) {
                TEST_ERROR();
            }

            f = __openat_2(dirfd, "file",  O_WRONLY);
            
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