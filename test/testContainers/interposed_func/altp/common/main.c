#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "print_result.h"

int main(int argc, char ** argv)
{
    int ret = do_test();
    
    char* test_name = strrchr(argv[0], '/');
    printf("%s: \t", test_name ? ++test_name : argv[0]);
            
    ret == EXIT_SUCCESS ? print_passed() :print_failure();
    
    return ret;
}