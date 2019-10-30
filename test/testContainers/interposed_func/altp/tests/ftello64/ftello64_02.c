#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    int i = 0;
    char buffer[] = TEST_MSG;
    off64_t pos;

    CREATE_TMP_DIR();
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen64(tmp_file_name, "w");
    
    if(pFile != NULL) {
        for(i = 0; i < TEST_COUNT; i++) {
            if(sizeof(buffer) != fwrite(buffer, 1, sizeof(buffer), pFile)) {
                TEST_ERROR();
                break;
            }
        }
        
        pos = ftello64(pFile);
        if(pos != TEST_COUNT * sizeof(buffer)) {
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