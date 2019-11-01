#include <fcntl.h>
#include <sys/sendfile.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_from_name[255];
    char tmp_file_to_name[255];
    char buffer[] = TEST_MSG;
    int fromfd, tofd;
    off_t off = 0;
    int rv = 0;
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_from_name, "%s/fileFrom", tmp_dir_name);
    sprintf(tmp_file_to_name, "%s/fileTo", tmp_dir_name);

    if ((fromfd = open64(tmp_file_from_name, O_WRONLY | O_CREAT)) < 0 ) {
        TEST_ERROR();
    } else {
        for(i = 0; i < TEST_COUNT; i++) {
            if(write(fromfd, TEST_MSG, strlen(TEST_MSG)) != strlen(TEST_MSG)) {
                TEST_ERROR();
                break;
            }
        }

        if(close(fromfd) == EOF) {
            TEST_ERROR();
        }

        if ((fromfd = open64(tmp_file_from_name, O_RDONLY)) < 0 || (tofd = open64(tmp_file_to_name, O_WRONLY | O_CREAT)) < 0) {
            TEST_ERROR();
        } else {
            if ((rv = sendfile64(tofd, fromfd, &off, TEST_COUNT*strlen(TEST_MSG))) < 0) {
                TEST_ERROR();
            }

            if(close(fromfd) == EOF) {
                TEST_ERROR();
            }

            if(close(tofd) == EOF) {
                TEST_ERROR();
            }
        }
    }

    if ((tofd = open64(tmp_file_to_name, O_RDONLY)) < 0) {
            TEST_ERROR();
    } else {
        for(i = 0; i < TEST_COUNT; i++) {
            if ((rv = read(tofd, buffer, strlen(TEST_MSG))) < 0) {
                TEST_ERROR();
                break;
            } else {
                if(strcmp(buffer, TEST_MSG) != 0) {
                    TEST_ERROR();
                    break;
                }
            }
        }

        if(close(tofd) == EOF) {
            TEST_ERROR();
        }
    }

    unlink(tmp_file_from_name);
    unlink(tmp_file_to_name);

    REMOVE_TMP_DIR();

    return test_result;
}