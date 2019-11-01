#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    char buffer[5] = TEST_MSG_N;
    
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
        size_t bufsize = 0;
        char *b = NULL;
        memset(&buffer, 0, sizeof(buffer));
        if(getline(&b, &bufsize, pFile) <= 0) {
            TEST_ERROR();
        }

        if(strcmp(b, TEST_MSG_N) != 0) {
            TEST_ERROR();
        }

        free(b);

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