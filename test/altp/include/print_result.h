#include <stdio.h>

void reset () {
    printf("\033[0m");
}

void print_failure() {
    printf("\033[1;31m");
    
    printf("FAILURE\n");
    
    reset();
}
  
void print_passed() {
    printf("\033[0;32m");
    
    printf("PASSED\n");
    
    reset();
}
    
