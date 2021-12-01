#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    struct stat64 fs;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int f = open64(tmp_file_name, O_CREAT | O_RDONLY);

    if (f != EOF) {
        if (__fxstatat64(1, f, tmp_file_name, &fs, 0) < 0) {
            TEST_ERROR();
        }

        if (fs.st_size != 0) {
            TEST_ERROR();
        }

        if (close(f) == EOF) {
            TEST_ERROR();
        }

    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    f = open64(tmp_file_name, O_CREAT | O_WRONLY);

    if (f != EOF) {
        if (write(f, TEST_MSG, sizeof(TEST_MSG)) != sizeof(TEST_MSG)) {
            TEST_ERROR();
        }

        if (__fxstatat64(1, f, tmp_file_name, &fs, 0) < 0) {
            TEST_ERROR();
        }

        if (fs.st_size != sizeof(TEST_MSG)) {
            TEST_ERROR();
        }

        if (close(f) == EOF) {
            TEST_ERROR();
        }

    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}