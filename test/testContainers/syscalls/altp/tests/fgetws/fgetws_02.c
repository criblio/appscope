#include <wchar.h>

#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    int i = 0;
    wchar_t buffer[] = TEST_MSGW;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE *pFile = fopen(tmp_file_name, "w");

    if (pFile != NULL) {
        for (i = 0; i < TEST_COUNT; i++) {
            if (fputws(buffer, pFile) == EOF) {
                TEST_ERROR();
                break;
            }
        }

        if (fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    pFile = fopen(tmp_file_name, "r");

    if (pFile != NULL) {
        for (i = 0; i < TEST_COUNT; i++) {
            memset(buffer, 0, sizeof(buffer));

            if (fgetws(buffer, sizeof(buffer) / sizeof(wchar_t), pFile) == NULL) {
                TEST_ERROR();
                break;
            } else {
                if (wcscmp(buffer, TEST_MSGW) != 0) {
                    TEST_ERROR();
                    break;
                }
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