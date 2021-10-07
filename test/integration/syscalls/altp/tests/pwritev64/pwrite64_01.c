#include <fcntl.h>
#include <sys/uio.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;
    struct iovec iov[1];

    iov[0].iov_base = buffer;
    iov[0].iov_len = strlen(buffer);

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open64(tmp_file_name, O_CREAT | O_WRONLY);

    if (f != EOF) {
        if (pwritev64(f, iov, 1, 0) == -1) {
           TEST_ERROR();
        }

        if (close(f) != 0) {
           TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    f = open64(tmp_file_name, O_RDONLY);

    if (f != EOF) {
        memset(buffer, 0, sizeof(buffer));

        if(pread64(f, buffer, sizeof(buffer), 0) == EOF) {
            TEST_ERROR();
        }

        if(strcmp(buffer, TEST_MSG) != 0) {
            TEST_ERROR();
        }

        if (close(f)!= 0) {
           TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}