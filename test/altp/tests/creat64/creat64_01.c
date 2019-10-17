#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char tmp_file_name[255];    

    CREATE_TMP_DIR();

    sprintf(tmp_file_name, "%s/file", tmp_dir_name);

    int fd = creat64(tmp_file_name, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP );
    if(fd != EOF ) {
        if(close(fd) == EOF) {
            test_result = EXIT_FAILURE;
        }
        unlink(tmp_file_name);
    } else {
        test_result = EXIT_FAILURE;
    }

    REMOVE_TMP_DIR();
        
    return test_result;
}