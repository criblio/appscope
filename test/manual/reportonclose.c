#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// nc -klu4w 0 localhost 8125
// gcc -g reportonclose.c -o reportonclose && ./reportonclose


int
main()
{
    printf("running reportonclose.c\n");
    sleep (11);

    int fd, i;
    int byteswritten = 0;

    // write to a file descriptor and close the file
    char* path = "/tmp/reportonclose.tmp";
    fd = open(path, O_CREAT | O_RDWR, 0666);
    assert(fd != -1);
    for (i = 0; i<100; i++) {
        char buf[256] = {0};
        int bytes = snprintf(buf, sizeof(buf), "Writing a bunch of stuff %3d\n", i);
        byteswritten += write(fd, buf, bytes);
    }
    close(fd);
    unlink(path);
    printf("wrote %d bytes to %s\n", byteswritten, path);

    // reuse the file descriptor to write a different file
    byteswritten = 0;
    path = "/tmp/reportonclose2.tmp";
    fd = open(path, O_CREAT | O_RDWR, 0666);
    assert(fd != -1);

    for (i = 0; i<100; i++) {
        char buf[256] = {0};
        int bytes = snprintf(buf, sizeof(buf), "Writing a bunch of stuff %3d\n", i);
        byteswritten += write(fd, buf, bytes);
    }
    close(fd);
    unlink(path);
    printf("wrote %d bytes to %s\n", byteswritten, path);

    sleep(11);

    return 0;
}
