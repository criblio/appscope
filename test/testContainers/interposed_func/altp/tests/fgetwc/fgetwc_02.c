#include <locale.h>
#include <wchar.h>

#include "test_utils.h"

int do_test() {
    setlocale(LC_ALL, "en_US.utf8");
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    int i = 0;
    wint_t c = TEST_CHARW;
    
    CREATE_TMP_DIR();
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        for(i = 0; i < TEST_COUNT; i++) {
            if(fputwc(c, pFile) == WEOF) {
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
        for(i = 0; i < TEST_COUNT; i++) {
            c = fgetwc(pFile);
            if(c == WEOF || c != TEST_CHARW) {
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

    unlink(tmp_file_name);
    
    REMOVE_TMP_DIR();

    return test_result;
}