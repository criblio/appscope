#include <fcntl.h>

#include "test_utils.h"

ssize_t __read_chk(int fd, void * buf, size_t nbytes, size_t buflen);

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;
    int fd;
    int i = 0;
    ssize_t read = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);
    if ((fd = open(tmp_file_name, O_WRONLY | O_CREAT)) < 0 ) {
        TEST_ERROR();
    } else {
        for(i = 0; i < TEST_COUNT; i++) {
            if(write(fd, TEST_MSG, strlen(TEST_MSG)) != strlen(TEST_MSG)) {
                TEST_ERROR();
                break;
            }
        }

        if(close(fd) == EOF) {
            TEST_ERROR();
        }
    }

    if ((fd = open(tmp_file_name, O_RDONLY)) < 0) {
            TEST_ERROR();
    } else {
        for(i = 0; i < TEST_COUNT; i++) {
            if ((read = __read_chk(fd, buffer, strlen(TEST_MSG), strlen(TEST_MSG))) < 0) {
                TEST_ERROR();
                break;
            } else {
                if(strcmp(buffer, TEST_MSG) != 0) {
                    TEST_ERROR();
                    break;
                }
            }
        }

        if(close(fd) == EOF) {
            TEST_ERROR();
        }
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}