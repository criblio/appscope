/*
 * shmserver.c - A simple /dev/shm reader
 *
 * gcc -g test/manual/shmserver.c -lpthread -o shmserver
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DEFAULT_PATH "/dev/shm/"
#define MAXFDS       2
#define INSIZE       4096

void
usage(char *prog)
{
    fprintf(stderr, "usage: %s [-v] -f pipe/file\n", prog);
    exit(-1);
}

int
main(int argc, char **argv)
{
    char *buf, *path = NULL;
    int optval, rc, fd, infd, opt, watchid, verbose = 0;
    DIR *dirp;
    struct dirent *dent;
    struct stat stat;
    struct inotify_event inevt;

    if (argc < 2) {
        usage(argv[0]);
    }

    while ((opt = getopt(argc, argv, "vhf:")) > 0) {
        switch (opt) {
            case 'v':
                verbose++;
                break;
            case 'f':
                path = strdup(optarg);
                break;
            case 'h':
            default:
                usage(argv[0]);
                break;
        }
    }

    if (path == NULL) {
        strncpy(path, DEFAULT_PATH, strlen(DEFAULT_PATH));
    }

    do {
        if ((infd = inotify_init()) == -1) {
            perror("inotify_init1");
            exit(-1);
        }

        if ((watchid = inotify_add_watch(infd, path, IN_MODIFY)) == -1) {
            fprintf(stderr, "Cannot watch '%s'\n", path);
            perror("inotify_add_watch");
            exit(-1);
        }

        if ((buf = calloc(1, INSIZE)) == NULL) {
            perror("calloc-inotify");
            exit(-1);
        }

        // wait for files to be modified
        if ((rc = read(infd, buf, INSIZE)) <= 0) {
            perror("read-inotify-event");
            exit(-1);
        }

        free(buf);

        if ((dirp = opendir(path)) == NULL) {
            perror("opendir");
            exit(-1);
        }

        while ((dent = readdir(dirp)) != NULL) {
            char fpath[NAME_MAX];

            if (dent->d_type == DT_DIR)
                continue;

            strncpy(fpath, path, sizeof(fpath));
            strncat(fpath, "/", sizeof(fpath) - strlen(fpath));
            strncat(fpath, dent->d_name, sizeof(fpath) - strlen(fpath));

            if ((fd = open(fpath, O_RDWR)) < 0) {
                perror("open-dent");
                fprintf(stderr, "%s:%d fd:%d %s\n", __FUNCTION__, __LINE__, fd, dent->d_name);
                exit(-1);
            }

            if (fstat(fd, &stat) == -1) {
                perror("fstat");
                exit(-1);
            }

            if (stat.st_size == 0) {
                close(fd);
                continue;
            }

            if (verbose > 0)
                printf("%s:%d %s %s %s\n", __FUNCTION__, __LINE__, path, dent->d_name, fpath);

            if ((buf = calloc(1, stat.st_size)) == NULL) {
                perror("calloc");
                exit(-1);
            }

            if ((rc = read(fd, buf, stat.st_size)) <= 0) {
                perror("read");
                exit(-1);
            }

            if (verbose > 0)
                printf("%s:%d read %d bytes\n", __FUNCTION__, __LINE__, rc);

            if (flock(fd, LOCK_EX) == -1) {
                perror("flock EX");
                exit(-1);
            }

            if (unlink(fpath) == -1) {
                perror("unlink");
                fprintf(stderr, "file: %s\n", fpath);
            }

            if (flock(fd, LOCK_UN) == -1) {
                perror("flock UN");
                exit(-1);
            }

            if (close(fd) == -1) {
                perror("close");
                exit(-1);
            }

            free(buf);
            // echo input to stdout
            if (verbose > 1)
                write(1, buf, rc);
        }

        if (closedir(dirp) == -1) {
            perror("closedir");
            exit(-1);
        }

        if (close(infd) == -1) {
            perror("close-infd");
            exit(-1);
        }
    } while (1);
}
