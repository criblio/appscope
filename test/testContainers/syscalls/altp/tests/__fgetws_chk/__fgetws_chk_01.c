#include <wchar.h>

#include "test_utils.h"

wchar_t *__fgetws_chk(wchar_t *ws, size_t size, int strsize, FILE *stream);

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    wchar_t buffer[] = TEST_MSGW;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE *pFile = fopen(tmp_file_name, "w");

    if (pFile != NULL) {
        if (fputws(buffer, pFile) == EOF) {
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
        memset(buffer, 0, sizeof(buffer));

        if (__fgetws_chk(buffer, sizeof(buffer), sizeof(buffer), pFile) == NULL) {
            TEST_ERROR();
        } else {
            if (wcscmp(buffer, TEST_MSGW) != 0) {
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