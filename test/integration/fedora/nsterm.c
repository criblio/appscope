#define _GNU_SOURCE
#include <curses.h>
#include <term.h>

int
main(int argc, char *argv[])
{
    int res = setupterm((char *)0, 1, (int *)0);

    printf("setupterm res %d\n", res);
    return 0;
}
