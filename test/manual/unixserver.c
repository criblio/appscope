/*
 * unixserver.c - A simple UNIX Domain echo server
 *
 * gcc -g test/manual/unixserver.c -lpthread -o unixserver
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#define BUFSIZE 4096 * 4096
#define MAXFDS  500
#define CMDFILE "/tmp/cmdin"

void
usage(char *prog)
{
    fprintf(stderr, "usage: %s [-v] -f pipe/file\n", prog);
    exit(-1);
}

int
main(int argc, char **argv)
{
    int parentfd;  /* parent socket */
    int childfd;   /* child socket */
    int clientlen; /* byte size of client's address */
    struct sockaddr_un serveraddr;
    struct sockaddr_un clientaddr;
    char *buf;
    char *pfile; /* pipe/file */
    int optval;  /* flag value for setsockopt */
    int rc, i, j, fd, arr, opt;
    int numfds;
    int timeout;
    int verbose = 0;
    struct pollfd fds[MAXFDS];

    if (argc < 2) {
        usage(argv[0]);
    }

    while ((opt = getopt(argc, argv, "vhf:")) > 0) {
        switch (opt) {
            case 'v':
                verbose++;
                break;
            case 'f':
                pfile = strdup(optarg);
                break;
            case 'h':
            default:
                usage(argv[0]);
                break;
        }
    }

    if ((buf = calloc(1, BUFSIZE)) == NULL) {
        perror("calloc");
        exit(-1);
    }

    // socket: create the parent socket
    parentfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (parentfd < 0) {
        perror("ERROR opening socket");
        exit(1);
    }

    /* setsockopt: Handy debugging trick that lets
     * us rerun the server immediately after we kill it;
     * otherwise we have to wait about 20 secs.
     * Eliminates "ERROR on binding: Address already in use" error.
     */
    optval = 1;
    setsockopt(parentfd, SOL_SOCKET, SO_REUSEADDR, (const void *)&optval, sizeof(int));

    bzero((char *)&serveraddr, sizeof(serveraddr));

    serveraddr.sun_family = AF_UNIX;

    // pfile is the socket path
    strncpy(serveraddr.sun_path, pfile, sizeof(serveraddr.sun_path) - 1);
    if (unlink(pfile) == -1) {
        perror("unlink");
    }

    if (bind(parentfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
        perror("ERROR on binding");
        exit(1);
    }

    // listen: make this socket ready to accept connection requests
    if (listen(parentfd, 15) < 0) { /* allow 15 requests to queue up */
        perror("ERROR on listen");
        exit(1);
    }

    if (chmod(pfile, 0777) == -1) {
        perror("chmod");
        exit(-1);
    }

    // wait for a connection request then echo
    clientlen = sizeof(clientaddr);

    timeout = 10 * 1000;
    bzero(fds, sizeof(fds));
    fds[0].fd = parentfd;
    fds[0].events = POLLIN;
    fds[1].fd = 0;
    fds[1].events = POLLIN;
    numfds = 2;

    while (1) {
        rc = poll(fds, numfds, -1);

        // Error or timeout from poll;
        if (rc <= 0)
            continue;

        for (i = 0; i < numfds; ++i) {
            // printf("%s:%d fds[%d].fd = %d\n", __FUNCTION__, __LINE__, i, fds[i].fd);
            if (fds[i].revents == 0) {
                // printf("%s:%d No event\n", __FUNCTION__, __LINE__);
                continue;
            }

            if (fds[i].revents & POLLHUP) {
                printf("%s:%d Disconnect on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[1].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].revents & POLLERR) {
                printf("%s:%d Error on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[i].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].revents & POLLNVAL) {
                printf("%s:%d Invalid on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[i].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].fd == parentfd) {
                childfd = accept(parentfd, (struct sockaddr *)&clientaddr, &clientlen);
                if (childfd < 0) {
                    perror("ERROR on accept");
                    continue;
                }

                if (numfds > MAXFDS) {
                    printf("%s:%d exceeded max FDs supported\n", __FUNCTION__, __LINE__);
                    continue;
                }

                // try to re-use an entry
                for (j = 0; j < numfds; j++) {
                    if (fds[j].fd == -1) {
                        fds[j].fd = childfd;
                        fds[j].events = POLLIN;
                        arr = j;
                        break;
                    }
                }

                // if not, use a new entry
                if (j >= numfds) {
                    fds[numfds].fd = childfd;
                    fds[numfds].events = POLLIN;
                    arr = numfds;
                    numfds++;
                }

                printf("server established connection on [%d].%d\n", arr, childfd);
                break;
            } else if (fds[i].fd == 0) {
                // command input from stdin
                char *cmd;

                if (fgetc(stdin) == 'U') {
                    printf("%s:%d\n", __FUNCTION__, __LINE__);
                    if ((fd = open(CMDFILE, O_RDONLY)) < 0) {
                        perror("open");
                        continue;
                    }

                    if ((cmd = calloc(1, BUFSIZE)) == NULL) {
                        perror("calloc");
                        close(fd);
                        continue;
                    }

                    rc = read(fd, cmd, (size_t)BUFSIZE);
                    if (rc <= 0) {
                        perror("read");
                        free(cmd);
                        close(fd);
                        continue;
                    }

                    for (j = 2; j < numfds; j++) {
                        if ((fds[j].fd != -1) && (fds[j].fd > 2)) {
                            printf("%s:%d fds[%d].fd=%d rc %d\n%s\n", __FUNCTION__, __LINE__, j, fds[j].fd, rc, cmd);
                            if (write(fds[j].fd, cmd, rc) < 0) {
                                perror("write");
                            }
                        }
                    }

                    close(fd);
                    free(cmd);
                }
            } else {
                u_int64_t total = 0;
                int pfd;
                printf("%s:%d\n", __FUNCTION__, __LINE__);

                if ((verbose > 2) && (((pfd = open("/tmp/protstats.log", O_APPEND | O_RDWR | O_CREAT))) < 0)) {
                    perror("open");
                    exit(-1);
                }

                do {
                    rc = read(fds[i].fd, buf, (size_t)BUFSIZE);
                    if (rc < 0)
                        perror("read");
                    if (rc > 0) {
                        if (verbose > 0)
                            printf("%d\n", rc);
                        total += rc;
                    }

                    if (rc == 0) {
                        char msg[64];

                        close(fds[i].fd);
                        snprintf(msg, sizeof(msg), "Closed: total %ld\n", total);
                        puts(msg);
                        if (verbose > 2) {
                            write(pfd, msg, strlen(msg));
                            close(pfd);
                        }
                        break;
                    }
                    // echo input to stdout
                    if (verbose > 1)
                        write(1, buf, rc);
                } while (1);
            }
        }
    }
}
