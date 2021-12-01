#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char c = TEST_CHAR;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE *pFile = fopen(tmp_file_name, "w");

    if (pFile != NULL) {
        if (fputc(c, pFile) == EOF) {
            TEST_ERROR();
        }
        if (fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    pFile = fopen(tmp_file_name, "r");

    if (pFile != NULL) {
        c = fgetc(pFile);
        if (c == EOF || c != TEST_CHAR) {
            TEST_ERROR();
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