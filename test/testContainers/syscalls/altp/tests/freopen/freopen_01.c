#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE *pFile = freopen(tmp_file_name, "w", stderr);

    if (pFile != NULL) {
        fprintf(stderr, TEST_MSG);

        if (fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    pFile = fopen(tmp_file_name, "r");

    if (pFile != NULL) {
        memset(buffer, 0, sizeof(buffer));
        if (fread(buffer, 1, sizeof(buffer), pFile) == 0) {
            TEST_ERROR();
        } else {
            if (strcmp(buffer, TEST_MSG) != 0) {
                TEST_ERROR();
            }
        }

        if (fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}