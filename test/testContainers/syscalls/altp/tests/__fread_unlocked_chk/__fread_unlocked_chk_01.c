#include "test_utils.h"

size_t __fread_unlocked_chk(void *ptr, size_t ptrlen, size_t size, size_t nmemb, FILE *stream);

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[] = TEST_MSG;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");

    if(pFile != NULL) {
        if(sizeof(buffer) != fwrite(buffer, 1, sizeof(buffer), pFile)) {
            TEST_ERROR();
        }
        if(fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    pFile = fopen(tmp_file_name, "r");

    if(pFile != NULL) {
        memset(buffer, 0, sizeof(buffer));
        if(__fread_unlocked_chk(buffer, sizeof(buffer), sizeof(buffer), 1, pFile) == 0) {
            TEST_ERROR();
        } else {
            if(strcmp(buffer, TEST_MSG) != 0) {
                TEST_ERROR();
            }
        }

        if(fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    unlink(tmp_file_name);

    REMOVE_TMP_DIR();

    return test_result;
}