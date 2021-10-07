#include <fcntl.h>

#include "test_utils.h"

ssize_t __pread_chk(int fd, void * buf, size_t nbytes, off_t offset, size_t buflen);

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open(tmp_file_name, O_CREAT | O_WRONLY);

    if (f != EOF) {
        if (write(f, buffer, sizeof(buffer) - 1) == -1) {
           TEST_ERROR();
        }

        if (close(f) != 0) {
           TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    for(i = 0; i < TEST_COUNT; i++) {
        f = open(tmp_file_name, O_RDONLY);

        if (f != EOF) {
            memset(buffer, 0, sizeof(buffer));

            if(__pread_chk(f, buffer, sizeof(buffer), 0, sizeof(buffer)) == EOF) {
                TEST_ERROR();
                break;
            }

            if(strcmp(buffer, TEST_MSG) != 0) {
                TEST_ERROR();
                break;
            }

            if (close(f)!= 0) {
               TEST_ERROR();
               break;
            }
        } else {
            TEST_ERROR();
            break;
        }
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}