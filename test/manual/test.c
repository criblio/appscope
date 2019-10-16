/*
 * Testing the operation of Interposing Functions in Dependent Libraries
 * The libscope library has intgerposed malloc. Call malloc and see that we get the interposed function.
 */

#include <stdio.h>
#include <stdlib.h>
#include <syslog.h>

int main(int argc, char **argv)
{
    va_list ap;
    char *buf;
    
    printf("Starting interpose test\n");

    syslog (LOG_INFO, "A tree falls in a forest");
    vsyslog(LOG_INFO, "Hello World", ap);

    if ((buf = malloc(1024)) == NULL) {
        perror("malloc");
        return -1;
    }
    
    return 0;
}
