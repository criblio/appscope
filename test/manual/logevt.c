#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// gcc -g test/manual/logevt.c -o logevt && ./logevt

int
main(int argc, char *argv[])
{
    printf("running logevt\n");

    // provide default path
    char *path = "/var/log/input.log";
    // provide way to override path
    if (argc == 2)
        path = argv[1];

    // Wait for >10s to allow the periodic thread to be created.
    sleep(11);

    printf("...opening %s\n", path);

    FILE *f = fopen(path, "a");
    if (!f) {
        printf("fopen of %s failed\n", path);
        exit(-1);
    }

    printf("...writing to %s\n", path);

    int total_append_size = 0;
    int iterations = 0;
    char buf[200];
    while (total_append_size < 572396456) {

        int len = snprintf(buf, sizeof(buf), "%d Jan 27 01:44:35 ubuntu18_4 systemd-timesyncd[623]: Synchronized to time server 91.189.94.4:123 (ntp.ubuntu.com).\n", iterations++);
        if (len == -1) {
            printf("snprintf failed\n");
            exit(-1);
        }
        total_append_size += len;

        if (fwrite(buf, 1, len, f) == -1) {
            printf("fwrite to %s failed\n", path);
            exit(-1);
        }
    }

    printf("...done writing %d bytes to %s\n", total_append_size, path);

    fclose(f);

    return 0;
}
