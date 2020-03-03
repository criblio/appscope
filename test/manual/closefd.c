#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

//gcc -g test/manual/closefd.c -o closefd && ./closefd


/* This test was written because we've observed processes (splunk maybe)
   that close all file descriptors during initialization, probably to
   avoid leaking file descritors during subsequent fork/exec sequences.

   So, it's possible that the file descriptors for  g_mtc, g_log, and
   g_ctl can get closed out from under the libscope.so library.

   This was written to help verify that we correctly detect that our
   file descriptor was closed (errno of EBADF), and that we reestablish
   the connection.  Its a process that closes file descriptors that it
   didn't open.
*/


int
closeScopeFds() {
    int fd;
    for (fd=200; fd<1000; fd++) {
        close(fd);
    }
}


int
main()
{
    printf("Running closefd as pid %d\n", getpid());

    int i;
    // Loop for at least 45s  (Took 1m20s on my machine.)
    for (i=0; i<45000; i++) {
        // After at least 6s, close the file descriptors.
        // Repeat a while later...
        if ((i % 15000) == 6000) closeScopeFds();

        // Create arbitrary ongoing "activity"
        int fd=open("/tmp/scope.test.out", O_APPEND|O_CREAT, 0666);
        if (fd != -1) close(fd);

        // sleep for 10 ms each loop (in addition to the time open/close takes)
        struct timespec ts = {.tv_sec=0, .tv_nsec=010000000};
        nanosleep(&ts, NULL);
    }

    printf("Exiting closefd\n");
    return 0;
}
