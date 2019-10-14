#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    char tmp_dir_template[] = "/tmp/tmpdir.XXXXXX";
    char *tmp_dir_name = mkdtemp(tmp_dir_template);
    int i = 0;
    
    if(tmp_dir_name == NULL) {
        perror("mkdtemp failed: ");
        return EXIT_FAILURE;
    }
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    for(i = 0; i < 100; i++) {
        char file_name[255];    
	sprintf(file_name, "%s%d", tmp_file_name, i);
	
	FILE* pFile = fopen (file_name, "w");
    
	if (pFile != NULL) {
    	    fclose (pFile);
    	    unlink(file_name);
	} else {
	    test_result = EXIT_FAILURE;
	    break;
	}
    }
    
    if(rmdir(tmp_dir_name) == -1) {
        perror("rmdir failed: ");
        return EXIT_FAILURE;
    }
    
    return test_result;
}