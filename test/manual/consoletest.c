// gcc -g test/manual/consoletest.c -o consoletest && LD_PRELOAD=lib/linux/libscope.so SCOPE_EVENT_CONSOLE=true ./consoletest

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char filler[] = "0123456789abcde\n";
char start[5] = "START";
char end[3] = "END";

void
print_large_buf(int size)
{
    char *buf = calloc(1, size);
    int i;
    for (i=0; i<size-1; i++) {
        snprintf(&buf[i], 2, "%c", filler[i%16]);
    }
    memcpy(buf, start, sizeof(start));
    memcpy(&buf[size-1-sizeof(end)], end, sizeof(end));
    printf("\n");

    //printf("%s", buf); // Seems like this should work, but doesn't.
    puts(buf);
    printf("\n");

    free(buf);
}

int
main()
{
    printf("running consoletest.c\n");

    print_large_buf(54096);

    printf("exiting consoletest.c\n");

    return 0;
}
