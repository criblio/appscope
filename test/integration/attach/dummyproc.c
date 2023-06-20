#include <limits.h>
#include <stdio.h>
#include <time.h>

int
main(void)
{
    struct timespec time;

    // loop exists to handle interrupts without exiting
    while (1) {
        time.tv_sec = INT_MAX;
        time.tv_nsec = LONG_MAX;
        nanosleep(&time, &time);
    }

    return 0;
}
