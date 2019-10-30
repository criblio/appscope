#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    fpos_t position;

    CREATE_TMP_DIR();
        
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        if(!(fgetpos(pFile, &position) == 0 && position.__pos == 0)) {
            TEST_ERROR();
        } else {
            fputs("test", pFile);
            if(!(fgetpos(pFile, &position) == 0 && position.__pos == 4)) {
                TEST_ERROR();
            }
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