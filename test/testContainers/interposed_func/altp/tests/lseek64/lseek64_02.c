#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];
    int n = 0;
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open64(tmp_file_name, O_CREAT | O_WRONLY);
    
    if(f != EOF) {
        for(i = 0; i < 100; i++) {
            if(write(f, TEST_MSG, sizeof(TEST_MSG)) != sizeof(TEST_MSG)) {
                TEST_ERROR();
                break;
            }

            if(lseek64(f, n, SEEK_CUR) != (off64_t)((i + 1)*sizeof(TEST_MSG))) {
                TEST_ERROR();
                break;
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