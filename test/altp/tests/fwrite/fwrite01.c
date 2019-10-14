#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    
    char tmp_dir_template[] = "/tmp/tmpdir.XXXXXX";
    char *tmp_dir_name = mkdtemp(tmp_dir_template);
    char buffer[] = "test";
	
    if(tmp_dir_name == NULL) {
        perror("mkdtemp failed: ");
        return EXIT_FAILURE;
    }
    
    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    FILE* pFile = fopen (tmp_file_name, "w");
    
    if (pFile != NULL) {
	fwrite (buffer , 1, sizeof(buffer), pFile);
        fclose (pFile);
        unlink(tmp_file_name);
    } else {
	test_result = EXIT_FAILURE;
    }
    
    if(rmdir(tmp_dir_name) == -1) {
        perror("rmdir failed: ");
        return EXIT_FAILURE;
    }
    
    return test_result;
}