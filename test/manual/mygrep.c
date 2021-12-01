#include <regex.h>
#include <stdio.h>
#include <unistd.h>

// gcc -g test/manual/mygrep.c -o mygrep

// This was written to be able to compare the behavior of gnu extended regex
// (glibc default) with the pcre2 implementation we're evauating.
// The grep on my system doesn't use gnu extended regex (it's actually using
// libpcre.so.3!) so it wasn't a useful tool to evaluate the behavior.
//
// for example:
//   $ echo "{HEY}" | contrib/pcre2/build/pcre2grep '{HEY}'
//   {HEY}
//   $ echo "{HEY}" | grep -E '{HEY}'
//   {HEY}
//   $ echo "{HEY}" | ./mygrep '{HEY}'
//   executing mygrep
//     compiling {HEY} ... failed.

int
main(int argc, char *argv[])
{
    printf("executing mygrep\n");

    if (argc != 2) {
        printf("exiting with error.  Expected one argument.\n");
        return -1;
    }

    printf("  compiling %s ...", argv[1]);
    regex_t regex;
    if (regcomp(&regex, argv[1], REG_EXTENDED | REG_NOSUB)) {
        printf(" failed.\n");
        return -1;
    } else {
        printf(" success.\n");
    }

    printf("  reading from stdin...");
    char stdinbuf[4096];
    int stdinbytes = read(STDIN_FILENO, stdinbuf, sizeof(stdinbuf) - 1);
    if (stdinbytes == sizeof(stdinbuf) - 1) {
        printf(" failed.  Too many bytes read\n");
        return -1;
    } else if (stdinbytes == -1) {
        printf(" failed.\n");
        return -1;
    } else {
        printf(" success.\n");
    }
    stdinbuf[stdinbytes] = '\0';

    printf("  executing %s against input %s\n", argv[1], stdinbuf);
    if (regexec(&regex, stdinbuf, 0, NULL, 0)) {
        printf("No match\n");
        return -1;
    } else {
        printf("Match!\n");
    }

    return 0;
}
