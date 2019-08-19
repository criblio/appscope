#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

int
main()
{
    // Functionally, what this actually does doesn't matter.
    // This exists to see if sizeof(struct statx) compiles or not.
    // Is the return code of `gcc statx_exists.c` zero or non-zero?
    //
    // Ubuntu's struct statx definition seems to have moved around
    // and this is part of how I'm trying to deal with it.
    return sizeof(struct statx);
}
