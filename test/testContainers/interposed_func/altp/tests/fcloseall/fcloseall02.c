#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    int i = 0;

    CREATE_TMP_DIR();
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    for(i = 0; i < 100; i++) {
        char file_name[255];    
	sprintf(file_name, "%s%d", tmp_file_name, i);
	
	FILE* pFile = fopen(file_name, "w");
    
	if(pFile == NULL) {
	    TEST_ERROR();
	    break;
	}
    }
    
    if(fcloseall() == EOF) {
        TEST_ERROR();
    }

    for(i = 0; i < 100; i++) {
        char file_name[255];    
	sprintf(file_name, "%s%d", tmp_file_name, i);
	unlink(file_name);	
    }
    
    REMOVE_TMP_DIR();

    return test_result;
}