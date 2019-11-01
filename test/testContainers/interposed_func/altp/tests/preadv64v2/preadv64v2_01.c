#include <fcntl.h>
#include <sys/uio.h>

#include "test_utils.h"

ssize_t preadv64v2 (int fd, const struct iovec *vector, int count, off64_t offset, int flags);

int do_test() {
    int test_result = EXIT_SUCCESS;

#if (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 28)
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;
    off64_t offset = 0;
    struct iovec iov[1];

    iov[0].iov_base = buffer;
    iov[0].iov_len = strlen(buffer);

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open64(tmp_file_name, O_CREAT | O_WRONLY);

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

    f = open64(tmp_file_name, O_RDONLY);

    if (f != EOF) {
        memset(buffer, 0, sizeof(buffer));

        if(preadv64v2(f, iov, 1, offset, 0) == EOF) {
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
#endif

    return test_result;
}
