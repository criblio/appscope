#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "test_utils.h"

#define TEST_MSG "test"

int do_test() {
    int test_result = EXIT_SUCCESS;
    char buffer[] = TEST_MSG;
    fpos_t position;

    FILE *pFile = fmemopen((char*)TEST_MSG, sizeof(TEST_MSG), "r");
    fgetpos(pFile, &position);

    memset(buffer, 0, sizeof(buffer));

    if(fgets(buffer, sizeof(buffer), pFile) == NULL) {
        TEST_ERROR();
    } else {
        if(strcmp(buffer, TEST_MSG) != 0) {
            TEST_ERROR();
        }
    }

    fsetpos(pFile, &position);
    memset(buffer, 0, sizeof(buffer));

    if(!fgets_unlocked(buffer, sizeof(buffer), pFile)) {
        TEST_ERROR();
    } else {
        if(strcmp(buffer, TEST_MSG) != 0) {
            TEST_ERROR();
        }
    }

    fclose(pFile);
        
    return test_result;
}