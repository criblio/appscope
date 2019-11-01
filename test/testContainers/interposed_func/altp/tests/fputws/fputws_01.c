#include <wchar.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[NAME_MAX];
    wchar_t buffer[] = TEST_MSGW;

    CREATE_TMP_DIR();
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        if(fputws(buffer, pFile) == EOF) {
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