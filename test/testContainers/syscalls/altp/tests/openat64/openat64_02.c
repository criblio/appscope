#include <fcntl.h>
#include <sys/types.h>

#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    for (i = 0; i < TEST_COUNT; i++) {
        int dirfd = open64(tmp_dir_name, O_RDONLY);

        int f = openat64(dirfd, "file", O_CREAT | O_WRONLY);

        if (f != EOF) {
            if (write(f, TEST_MSG, sizeof(TEST_MSG)) != sizeof(TEST_MSG)) {
                TEST_ERROR();
                break;
            }

            if (close(f) == EOF) {
                TEST_ERROR();
            }

            if (close(dirfd) == EOF) {
                TEST_ERROR();
            }

        } else {
            TEST_ERROR();
        }

        unlink(tmp_file_name);
    }

    REMOVE_TMP_DIR();

    return test_result;
}