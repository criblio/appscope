#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>

#include "common.h"

#define TEST_ERROR() \
    test_result = EXIT_FAILURE; \
    fprintf(stderr, "[ERROR] Error at line number %d in file %s\n", __LINE__, __FILE__);
        
#define CREATE_TMP_DIR() \
    char tmp_dir_template[] = "/tmp/tmpdir.XXXXXX"; \
    char *tmp_dir_name = mkdtemp(tmp_dir_template); \
    if(tmp_dir_name == NULL) { \
        perror("mkdtemp failed: "); \
        return EXIT_FAILURE; \
    } 
    
#define REMOVE_TMP_DIR() \
    if(rmdir(tmp_dir_name) == -1) { \
        perror("rmdir failed: "); \
        return EXIT_FAILURE; \
    } 

#endif /* __TEST_UTILS_H */