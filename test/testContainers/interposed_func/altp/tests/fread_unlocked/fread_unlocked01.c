#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
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
        if(sizeof(buffer) != fread_unlocked(buffer, 1, sizeof(buffer), pFile)) {
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