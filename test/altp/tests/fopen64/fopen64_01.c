#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen64(tmp_file_name, "w");
    
    if(pFile != NULL) {
        if(fclose(pFile) == EOF) {
            test_result = EXIT_FAILURE;
        }
        unlink(tmp_file_name);
    } else {
        test_result = EXIT_FAILURE;
    }

    REMOVE_TMP_DIR();

    return test_result;
}