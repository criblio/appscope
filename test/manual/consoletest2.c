// gcc -g test/manual/consoletest2.c -o consoletest2 && LD_PRELOAD=lib/linux/libscope.so SCOPE_EVENT_CONSOLE=true ./consoletest2
//
#include <stdio.h>
int main(void)
{
    char buf[128];
    printf("starting consoletest2.c\n");
    int i;
    for (i=0; i<1000000; i++) {
        snprintf(buf, sizeof(buf), "%d", i);
        printf("%s\n", buf);
    }
    printf("ending consoletest2.c\n");
}

