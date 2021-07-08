/* 
 * tcpserver.c - A simple TCP echo server 
 *
 * usage: tcpserver [OPTIONS] PORT
 * options: -t, --tls       use TLSv1.3
 *
 * requires: libssl-dev
 * build: gcc -g test/manual/tcpserver.c -lpthread -lssl -lcrypto -o tcpserver
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/poll.h>
#include <stdbool.h>
#include <getopt.h>
#include <sys/wait.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <signal.h>

#define BUFSIZE 4096
#define MAXFDS 500
#define CMDFILE "/tmp/cmdin"

int socket_setup(int port);
void tcp(int socket);
void tcp_ssl(int socket);

// long aliases for short options
static struct option options[] = {
    {"tls", no_argument, 0, 't'}
};

// program helper
void
showUsage()
{
    printf("usage: tcpserver [OPTIONS] PORT\n");
    printf("options: -t, --tls       use TLSv1.3\n");
}

int
main(int argc, char *argv[])
{
    int opt = -1;
    int option_index = 0;
    bool tls = false;

    // get options
    while (1) {
        opt = getopt_long(argc, argv, "t", options, &option_index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
        case 't':
            tls = true;
            break;
        default: /* '?' */
            showUsage();
            exit(EXIT_FAILURE);
        }
    }

    // get port argument
    if (optind >= argc) {
        fprintf(stderr, "error: missing PORT argument\n");
        showUsage();
        exit(EXIT_FAILURE);
    }
    int port = atoi(argv[optind]);
    if (port < 1) {
        showUsage();
        return EXIT_FAILURE;
    }
    
    int socket = socket_setup(port);
    if (tls) {
        tcp_ssl(socket);
    } else {
        tcp(socket);
    }

    exit(EXIT_SUCCESS);
}

int
socket_setup(int port)
{
    struct sockaddr_in serveraddr; /* server's addr */
    int optval; /* flag value for setsockopt */
    int parentfd; /* parent socket */

    // socket: create the parent socket 
    parentfd = socket(AF_INET, SOCK_STREAM, 0);
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
    setsockopt(parentfd, SOL_SOCKET, SO_REUSEADDR, 
            (const void *)&optval , sizeof(int));

    /*
     * build the server's Internet address
     */
    bzero((char *) &serveraddr, sizeof(serveraddr));

    /* this is an Internet address */
    serveraddr.sin_family = AF_INET;

    /* let the system figure out our IP address */
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);

    /* this is the port we will listen on */
    serveraddr.sin_port = htons((unsigned short)port);

    /* 
     * bind: associate the parent socket with a port 
     */
    if (bind(parentfd, (struct sockaddr *) &serveraddr, 
                sizeof(serveraddr)) < 0) {
        perror("ERROR on binding");
        exit(1);
    }

    /* 
     * listen: make this socket ready to accept connection requests 
     */
    if (listen(parentfd, 15) < 0) { /* allow 15 requests to queue up */ 
        perror("ERROR on listen");
        exit(1);
    }

    return parentfd;
}

// wait for a connection request then echo
void
tcp(int socket)
{
    struct sockaddr_in clientaddr; /* client addr */
    int childfd; /* child socket */
    int clientlen;
    int timeout;
    int numfds;
    struct pollfd fds[MAXFDS];
    int rc, i, j, fd, arr;
    struct hostent *hostp; /* client host info */
    char buf[BUFSIZE]; /* message buffer */
    char *hostaddrp; /* dotted decimal host addr string */

    clientlen = sizeof(clientaddr); /* byte size of client's address */
    timeout = 10 * 1000;
    bzero(fds, sizeof(fds));
    fds[0].fd = socket;
    fds[0].events = POLLIN;
    fds[1].fd = 0;
    fds[1].events = POLLIN;
    numfds = 2;

    while (1) {
        rc = poll(fds, numfds, -1);

        // Error or timeout from poll;
        if (rc <= 0) continue;

        for (i = 0; i < numfds; ++i) {
            //fprintf(stderr, "%s:%d fds[%d].fd = %d\n", __FUNCTION__, __LINE__, i, fds[i].fd);
            if (fds[i].revents == 0) {
                //fprintf(stderr, "%s:%d No event\n", __FUNCTION__, __LINE__);
                continue;
            }

            if (fds[i].revents & POLLHUP) {
                fprintf(stderr, "%s:%d Disconnect on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[1].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].revents & POLLERR) {
                fprintf(stderr, "%s:%d Error on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[i].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].revents & POLLNVAL) {
                fprintf(stderr, "%s:%d Invalid on fd %d\n", __FUNCTION__, __LINE__, fd);
                close(fds[i].fd);
                fds[i].fd = -1;
                fds[i].events = 0;
                continue;
            }

            if (fds[i].fd == socket) {
                childfd = accept(socket, (struct sockaddr *) &clientaddr, &clientlen);
                if (childfd < 0) {
                    perror("ERROR on accept");
                    continue;
                }

                if (numfds > MAXFDS) {
                    fprintf(stderr, "%s:%d exceeded max FDs supported\n", __FUNCTION__, __LINE__);
                    continue;
                }

                // try to re-use an entry
                for (j=0; j < numfds; j++) {
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

                // who sent the message 
                hostp = gethostbyaddr((const char *)&clientaddr.sin_addr.s_addr, 
                        sizeof(clientaddr.sin_addr.s_addr), AF_INET);
                if (hostp == NULL) {
                    //perror("ERROR on gethostbyaddr");
                    fprintf(stderr, "server established connection on [%d].%d\n", arr, childfd);
                    continue;
                }

                hostaddrp = inet_ntoa(clientaddr.sin_addr);
                if (hostaddrp == NULL) {
                    //perror("ERROR on inet_ntoa\n");
                    fprintf(stderr, "server established connection on [%d].%d with %s\n",
                            arr, childfd, hostp->h_name);
                    continue;
                }

                fprintf(stderr, "server established connection on [%d].%d with %s (%s:%d)\n",
                        arr, childfd, hostp->h_name, hostaddrp, htons(clientaddr.sin_port));
                break;
            } else if (fds[i].fd == 0) {
                // command input from stdin
                char *cmd;

                if (fgetc(stdin) == 'U') {
                    fprintf(stderr, "%s:%d\n", __FUNCTION__, __LINE__);
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
                            fprintf(stderr, "%s:%d fds[%d].fd=%d rc %d\n%s\n", __FUNCTION__, __LINE__,
                                    j, fds[j].fd, rc, cmd);
                            if (send(fds[j].fd, cmd, rc, 0) < 0) { // MSG_DONTWAIT
                                perror("send");
                            }
                        }
                    }

                    close(fd);
                    free(cmd);
                }
            } else {
                do {
                    bzero(buf, BUFSIZE);
                    rc = recv(fds[i].fd, buf, (size_t)BUFSIZE, MSG_DONTWAIT);
                    if (rc < 0) {
                        break;
                    } else if (rc == 0) {
                        // EOF
                        close(fds[i].fd);
                        fds[i].fd = -1;
                        fds[i].events = 0;
                    }
                    // echo input to stdout
                    write(1, buf, rc);

                    // Artifical delay...
                    //struct timespec ts = {.tv_sec=0, .tv_nsec=001000000}; // 1 ms
                    //nanosleep(&ts, NULL);

                } while (1);
            }
        }
    }
    return;
}

// wait for a connection request then echo.
// tls 1.3 implementation
void
tcp_ssl(int socket)
{
    pid_t server_pid;
    int wstatus;
    char buffer[1024];
    int processed;
    SSL_CTX *ssl_ctx;
    SSL *ssl;
    int fd;
    int retval;

    fd = accept(socket, NULL, 0);
    if (fd < 0) {
        perror("accept failed");
        exit(EXIT_FAILURE);
    }
    printf("TCP accepted.\n");
    /*if (!OPENSSL_init_ssl(0, NULL)) {
        fprintf(stderr, "OPENSSL_init_ssl failed\n");
        exit(EXIT_FAILURE);
    }*/
    /*OpenSSL_add_ssl_algorithms();*/
    ssl_ctx = SSL_CTX_new(TLS_server_method());
    if (!ssl_ctx) {
        fprintf(stderr, "SSL_CTX_new failed\n");
        exit(EXIT_FAILURE);
    }
    if (1 != SSL_CTX_use_PrivateKey_file(ssl_ctx, "key.pem", SSL_FILETYPE_PEM)) {
        fprintf(stderr, "SSL_CTX_use_PrivateKey_file failed: ");
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
    if (1 != SSL_CTX_use_certificate_file(ssl_ctx, "cert.pem", SSL_FILETYPE_PEM)) {
        fprintf(stderr, "SSL_CTX_use_certificate_file failed: ");
        ERR_print_errors_fp(stderr);
        exit(EXIT_FAILURE);
    }
    ssl = SSL_new(ssl_ctx);
    if (!ssl) {
        fprintf(stderr, "SSL_new failed\n");
        exit(EXIT_FAILURE);
    }
    if (!SSL_set_fd(ssl, fd)) {
        fprintf(stderr, "SSL_set_fd failed\n");
        exit(EXIT_FAILURE);
    }
#if OPENSSL_VERSION_NUMBER >= 0x1010100fL
    /* TLS 1.3 server sends session tickets after a handhake as part of
     * the SSL_accept(). If a client finishes all its job before server
     * sends the tickets, SSL_accept() fails with EPIPE errno. Since we
     * are not interested in a session resumption, we can not to send the
     * tickets. */
    /*if (1 != SSL_set_num_tickets(ssl, 0)) {
        fprintf(stderr, "SSL_set_num_tickets failed\n");
        exit(EXIT_FAILURE);
    }
    Or we can perform two-way shutdown. Client must call SSL_read() before
    the final SSL_shutdown(). */
#endif
    retval = SSL_accept(ssl);
    if (retval  <= 0) {
        fprintf(stderr, "SSL_accept failed ssl_err=%d errno=%s: ",
                SSL_get_error(ssl, retval), strerror(errno));
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
    printf("SSL accepted.\n");
    while (1) {
        processed = SSL_read(ssl, buffer, sizeof(buffer));
        printf("Server SSL_read returned %d\n", processed);
        if (processed > 0) {
            printf("%.*s", (int)processed, buffer);
        } else {
            int ssl_error = SSL_get_error(ssl, processed);
            if (ssl_error == SSL_ERROR_ZERO_RETURN) {
                printf("Server thinks a client closed a TLS session\n");
                break;
            }
            if (ssl_error != SSL_ERROR_WANT_READ &&
                    ssl_error != SSL_ERROR_WANT_WRITE) {
                fprintf(stderr, "server read failed: ssl_error=%d:", ssl_error);
                ERR_print_errors_fp(stderr);
                fprintf(stderr, "\n");
                exit(EXIT_FAILURE);
            }
        }
    };
    printf("Server read finished.\n");
    retval = SSL_shutdown(ssl);
    if (retval < 0) {
        int ssl_err = SSL_get_error(ssl, retval);
        fprintf(stderr, "Server SSL_shutdown failed: ssl_err=%d\n", ssl_err);
        kill(server_pid, SIGTERM);
        exit(EXIT_FAILURE);
    }
    printf("Server shut down a TLS session.\n");
    if (retval != 1) {
        retval = SSL_shutdown(ssl);
        if (retval != 1) {
            int ssl_err = SSL_get_error(ssl, retval);
            fprintf(stderr,
                    "Waiting for client shutdown using SSL_shutdown failed: "
                    "ssl_err=%d\n", ssl_err);
            kill(server_pid, SIGTERM);
            exit(EXIT_FAILURE);
        }
    }
    printf("Server thinks a client shut down the TLS session.\n");
    SSL_free(ssl);
    SSL_CTX_free(ssl_ctx);
    close(fd);

    return;
}
