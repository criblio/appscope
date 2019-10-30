#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    char buffer[] = TEST_MSG;
    size_t len;
    ssize_t read;
    char *lineptr = NULL;

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
        while((read = getdelim(&lineptr, &len, 101, pFile)) != EOF) {
            if(strcmp(lineptr, "te") != 0) {
                TEST_ERROR();
            }
            break;
        }

        free(lineptr);

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