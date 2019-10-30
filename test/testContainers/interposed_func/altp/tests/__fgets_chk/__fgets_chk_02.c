#include "test_utils.h"

#define TEST_MSG "test"

char * __fgets_chk(char * s, size_t size, int strsize, FILE * stream);

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
	    for(i = 0; i < 100; i++) {
	        memset(buffer, 0, sizeof(buffer));
	        
	        if(__fgets_chk(buffer, sizeof(buffer), sizeof(buffer), pFile) == NULL) {
	            TEST_ERROR();
	            break;
	        } else {
	            if(strcmp(buffer, TEST_MSG) != 0) {
	                TEST_ERROR();
                    break;
	            }
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