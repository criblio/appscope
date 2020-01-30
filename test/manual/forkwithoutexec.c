#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

// nc -klu4w 0 localhost 8125
// gcc -g test/manual/forkwithoutexec.c -o forkwithoutexec && ./forkwithoutexec

int
do_child()
{
    pid_t pid = getpid();
    printf("child running w/pid %d\n", pid);
    sleep(1);
    int i;
    for (i=3; i<1023; i++) {
        errno = 0;
        int rv = close(i);
        if (i < 12 || i > 990) {
             printf("%d closing %d ", pid, i);
             if (rv == -1)
                 printf("... failed w/errno %d\n", errno);
             else
                 printf("... successful\n");
        }
    }
    for (i=0; i<100; i++) {
        // This provides activity for our library to create the periodic thread
        open("anotherfilethatdoesntexist", O_RDONLY);
        sleep(1);
    }
}

int
do_parent()
{
    pid_t pid = getpid();
    printf("parent running w/pid %d\n", pid);
    int i;
    for (i=0; i<100; i++) {
        // This provides activity for our library to create the periodic thread
        open("file_that_doesnt_exist", O_RDONLY);
        sleep(1);
    }
}

int
do_common()
{
    pid_t pid = getpid();
    printf("common running w/pid %d\n", pid);
    printf("%d opening file %d\n", pid,
          open("/tmp/file.1", O_CLOEXEC | O_RDWR | O_CREAT, 0666));
    printf("%d opening socket %d\n", pid,
          socket(AF_UNIX, SOCK_DGRAM, 0));
}

int
main()
{
    printf("running forkwithoutexec.c\n");

    do_common();

    pid_t pid = fork();

    if (pid == 0) {
        do_child();
    } else {
        do_parent();
    }

    printf("pid %d exiting\n", getpid());
    return 0;
}
