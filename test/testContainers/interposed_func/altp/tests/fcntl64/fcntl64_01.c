#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include "test_utils.h"

int fcntl64 (int fd, int cmd, ...);

int do_test() {
    int test_result = EXIT_SUCCESS;

#if (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 28)
    char tmp_file_name[NAME_MAX];

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    struct flock fl;

    int f = open64(tmp_file_name, O_CREAT | O_RDWR);

    if (f != EOF) {
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 100;
        fl.l_len = 10;

        if (fcntl64(f, F_SETLK, &fl) == -1) {
            if (errno != EACCES && errno != EAGAIN) {
                TEST_ERROR();
            }
        } else {
            fl.l_type = F_UNLCK;
            fl.l_whence = SEEK_SET;
            fl.l_start = 100;
            fl.l_len = 10;
            if (fcntl64(f, F_SETLK, &fl) == -1) {
                TEST_ERROR();
            }
        }
    } else {
        TEST_ERROR();
    }

    if(close(f) == EOF) {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();
#endif

    return test_result;
}