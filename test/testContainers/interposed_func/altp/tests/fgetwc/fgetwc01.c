#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <wchar.h>

#include "test_utils.h"

#define TEST_CHAR L'А'

int do_test() {
    setlocale(LC_ALL, "en_US.utf8");
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    wint_t c = TEST_CHAR;

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        if(fputwc(c, pFile) == EOF) {
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
        c = fgetwc(pFile);
        if(c == EOF || c != TEST_CHAR) {
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