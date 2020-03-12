#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// gcc -g test/manual/simpleopenreadclose.c -o simpleopenreadclose && ./simpleopenreadclose

#define TESTFILE "./testfile.out"
int stopLooping = 0;
int interval = 5; // 5 second default

void
sigHandler(int sig_num)
{
    stopLooping = (sig_num == SIGINT);
}

int
main(int argc, char* argv[])
{
    printf("Running simpleopenreadclose.c\n");
    printf("... Hit ctl-c when done...\n");

    // Accept a single argument that specifies an integer sleep interval
    // (in seconds)
    if (argc == 2) {
        errno = 0;
        unsigned long temp = strtoul(argv[1], NULL, 10);
        if (temp && !errno) interval = temp;
    }

    signal(SIGINT, sigHandler);

    while (!stopLooping) {
        int fd = open(TESTFILE, O_RDWR | O_CREAT, 0666);
        write(fd, "something...\n", sizeof("something...\n"));
        write(fd, "something else...\n", sizeof("something else...\n"));
        close(fd);
        sleep(interval);
    }

    printf("Received ctl-c.  Exiting.\n");
    unlink(TESTFILE);
}
