#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    int i = 0;
    char buffer[] = TEST_MSG;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");

    if(pFile != NULL) {
        for(i = 0; i < TEST_COUNT; i++) {
            if(fwrite(buffer, 1, strlen(buffer), pFile) != strlen(buffer)) {
                TEST_ERROR();
                break;
            }
        }

        if(fclose(pFile) == EOF) {
            TEST_ERROR();
        }
    } else {
        TEST_ERROR();
    }

    pFile = fopen(tmp_file_name, "r");

    if(pFile != NULL) {
        char buf[1024];
        if(fscanf(pFile, "%s", buf) == EOF) {
            TEST_ERROR();

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