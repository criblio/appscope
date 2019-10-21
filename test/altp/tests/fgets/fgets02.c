#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    int i = 0;
    char buffer[] = TEST_MSG;
    
    CREATE_TMP_DIR();
        
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen(tmp_file_name, "w");
    
    if(pFile != NULL) {
        for(i = 0; i < 100; i++) {
            if(fputs(buffer, pFile) == EOF) {
                test_result = EXIT_FAILURE;
                break;
            }
        }
    
        if(fclose(pFile) == EOF) {
            test_result = EXIT_FAILURE;
        }
    } else {
        test_result = EXIT_FAILURE;
    }

    pFile = fopen(tmp_file_name, "r");
    
    if(pFile != NULL) {
	    for(i = 0; i < 100; i++) {
	        memset(buffer, 0, sizeof(buffer));
	        
	        if(fgets(buffer, sizeof(buffer), pFile) == NULL) {
	            test_result = EXIT_FAILURE;
	            break;
	        } else {
	            if(strcmp(buffer, TEST_MSG) != 0) {
	                test_result = EXIT_FAILURE;
	            }
	        }
	    }
    
        if(fclose(pFile) == EOF) {
            test_result = EXIT_FAILURE;
        }
    } else {
    	test_result = EXIT_FAILURE;
    }
    
    unlink(tmp_file_name);
    
    REMOVE_TMP_DIR();
        
    return test_result;
}