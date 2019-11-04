#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    int i = 0;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    for(i = 0; i < TEST_COUNT; i++) {
        char file_name[PATH_MAX];
        sprintf(file_name, "%s%d", tmp_file_name, i);

        FILE* pFile = fopen(file_name, "w");

        if(pFile == NULL) {
            TEST_ERROR();
            break;
        }
    }

    if(fcloseall() == EOF) {
        TEST_ERROR();
    }

    for(i = 0; i < TEST_COUNT; i++) {
        char file_name[PATH_MAX];
        sprintf(file_name, "%s%d", tmp_file_name, i);
        unlink(file_name);
    }

    REMOVE_TMP_DIR();

    return test_result;
}