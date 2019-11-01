#include <locale.h>
#include <wchar.h>

#include "test_utils.h"

int do_test() {
    setlocale(LC_ALL, "en_US.utf8");
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    wint_t c = L'А';
    
    CREATE_TMP_DIR();
        
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        if(fputwc(c, pFile) == WEOF) {
            TEST_ERROR();
        }        
        if(fclose(pFile) == EOF) {
            TEST_ERROR();
        }
        unlink(tmp_file_name);
    } else {
        TEST_ERROR();
    }
    
    REMOVE_TMP_DIR();
        
    return test_result;
}