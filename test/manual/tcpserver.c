/* 
 * tcpserver.c - A simple TCP server 
 *
 * usage: tcpserver [OPTIONS] PORT
 * options: -t, --tls       use TLSv1.3
 *
 * requires: libssl-dev
 * build: gcc -g test/manual/tcpserver.c -Icontrib/build/openssl/include -Icontrib/openssl/include -L contrib/build/openssl -Wl,-R$PWD/contrib/build/openssl -lpthread -lssl -lcrypto -o tcpserver
 *
 * generate unencrypted TLS key and certificate:
 * openssl req -nodes -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem
 *
 * example scope client usage:
 * SCOPE_CRIBL_TLS_VALIDATE_SERVER=false scope run -c tls://127.0.0.1:9000 -- top
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

// Forward declarations
int socket_setup(int);
int socket_teardown(int);
int tcp(int);
int tcp_ssl(int);

// Long aliases for short options
static struct option options[] = {
    {"tls", no_argument, 0, 't'}
};

// Flag that tells the server to exit
static volatile sig_atomic_t exit_request = 0;
 
// Signal handler
void
hdl(int sig)
{
	exit_request = 1;
}

// Program helper
void
showUsage()
{
    printf("usage: tcpserver [OPTIONS] PORT\n");
    printf("options: -t, --tls       use TLSv1.3\n");
}

int
main(int argc, char *argv[])
{
    // ignore SIGPIPE when writing to a closed socket
    signal(SIGPIPE, SIG_IGN);

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
    
    int socket;
    if ((socket = socket_setup(port)) < 0) {
        exit(EXIT_FAILURE);
    }

    if (tls) {
        if (tcp_ssl(socket) < 0) {
            exit(EXIT_FAILURE);
        }
    } else {
        if (tcp(socket) < 0) {
            exit(EXIT_FAILURE);
        }
    }

    if (socket_teardown(socket) < 0) {
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}

int
socket_setup(int port)
{
    int optval; /* flag value for setsockopt */

    // create socket
    int parentfd;
    parentfd = socket(AF_INET, SOCK_STREAM, 0);
    if (parentfd < 0) {
        fprintf(stderr, "Error opening socket\n");
        return -1;
    }

    /* setsockopt: Handy debugging trick that lets 
     * us rerun the server immediately after we kill it; 
     * otherwise we have to wait about 20 secs. 
     * Eliminates "ERROR on binding: Address already in use" error. 
     */
    optval = 1;
    setsockopt(parentfd, SOL_SOCKET, SO_REUSEADDR, 
            (const void *)&optval , sizeof(optval));

    // server properties
    struct sockaddr_in serveraddr; /* server's addr */
    bzero((char *) &serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons((unsigned short)port);

    // bind: associate the parent socket with a port 
    if (bind(parentfd, (struct sockaddr *) &serveraddr, 
                sizeof(serveraddr)) < 0) {
        fprintf(stderr, "Error on binding\n");
        return -1;
    }

    // listen: make this socket ready to accept connection requests 
    if (listen(parentfd, 15) < 0) { /* allow 15 requests to queue up */ 
        fprintf(stderr, "Error on listen\n");
        return -1;
    }

    printf("Server set up parent TCP socket.\n");

    return parentfd;
}

// Socket teardown
int 
socket_teardown(int socket)
{
    if (shutdown(socket, SHUT_RDWR)) {
        fprintf(stderr, "server shutdown failed\n");
        return -1;
    }
    printf("Server shut down parent TCP socket.\n");
    close(socket);
    printf("Server closed parent TCP socket.\n");
    return 0;
}

// Wait for a connection request then print messages to stdout.
int
tcp(int socket)
{
    struct sockaddr_in clientaddr; /* client addr */
    int childfd; /* child socket */
    socklen_t clientlen;
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
                    // print input to stdout
                    write(1, buf, rc);

                    // Artificial delay...
                    // struct timespec ts = {.tv_sec=0, .tv_nsec=001000000}; // 1 ms
                    // nanosleep(&ts, NULL);

                } while (1);
            }
        }
    }
    return 0;
}

// Generate SSL context and load certificates
SSL_CTX*
ssl_ctx_new()
{
    SSL_CTX *ssl_ctx;
    ssl_ctx = SSL_CTX_new(TLS_server_method());
    if (!ssl_ctx) {
        fprintf(stderr, "SSL_CTX_new failed\n");
        return NULL;
    }
    if (1 != SSL_CTX_use_PrivateKey_file(ssl_ctx, "key.pem", SSL_FILETYPE_PEM)) {
        fprintf(stderr, "SSL_CTX_use_PrivateKey_file failed: ");
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\n");
        return NULL;
    }
    if (1 != SSL_CTX_use_certificate_file(ssl_ctx, "cert.pem", SSL_FILETYPE_PEM)) {
        fprintf(stderr, "SSL_CTX_use_certificate_file failed: ");
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\n");
        return NULL;
    }
    return ssl_ctx;
}

// Generate SSL from context
SSL*
ssl_new(int fd, SSL_CTX *ssl_ctx)
{
    SSL* ssl;
    ssl = SSL_new(ssl_ctx);
    if (!ssl) {
        fprintf(stderr, "SSL_new failed\n");
        return NULL;
    }
    if (!SSL_set_fd(ssl, fd)) {
        fprintf(stderr, "SSL_set_fd failed\n");
        return NULL;
    }
    return ssl;
}

// Bidirectional SSL shutdown
int
ssl_shutdown(SSL *ssl) {
    int ret;
    ret = SSL_shutdown(ssl);
    if (ret < 0) {
        int ssl_err = SSL_get_error(ssl, ret);
        fprintf(stderr, "Server SSL_shutdown failed: ssl_err=%d\n", ssl_err);
        return -1;
    }
    printf("Server shut down a TLS session.\n");
    if (ret != 1) {
        ret = SSL_shutdown(ssl);
        if (ret != 1) {
            int ssl_err = SSL_get_error(ssl, ret);
            fprintf(stderr,
                    "Waiting for client shutdown using SSL_shutdown failed: "
                    "ssl_err=%d\n", ssl_err);
            return -1;
        }
    }
    printf("Server thinks a client shut down the TLS session.\n");
    return 0;
}

// Read from SSL socket
int
ssl_read(SSL *ssl)
{
    char buffer[BUFSIZE];
    int processed;

    processed = SSL_read(ssl, buffer, sizeof(buffer));
    if (processed > 0) {
        printf("%.*s", (int)processed, buffer);
    } else {
        int ssl_error = SSL_get_error(ssl, processed);
        if (ssl_error == SSL_ERROR_ZERO_RETURN) {
            printf("Server thinks a client closed a TLS session\n");
            return 0;
        }
        if (ssl_error != SSL_ERROR_WANT_READ &&
                ssl_error != SSL_ERROR_WANT_WRITE) {
            fprintf(stderr, "server read failed: ssl_error=%d:", ssl_error);
            ERR_print_errors_fp(stderr);
            fprintf(stderr, "\n");
            return -1;
        }
    }

    return processed;
}

// Wait for a connection request then print messages to stdout
// SSL implementation
int
tcp_ssl(int socket)
{
    struct sigaction sa;
	sa.sa_flags = 0;
	sa.sa_handler = &hdl;
	sigfillset(&sa.sa_mask);
	sigdelset(&sa.sa_mask, SIGINT);
	sigaction(SIGINT, &sa, NULL);
 
    int wstatus;
    int fd;
    int ret;
    fd_set read_set;

    SSL_CTX *ssl_ctx = ssl_ctx_new();
    if (!ssl_ctx) {
        return -1;
    }

    fd = accept(socket, NULL, 0);
    if (fd < 0) {
        fprintf(stderr, "accept failed\n");
        return -1;
    }
    printf("TCP connection accepted.\n");

    SSL *ssl = ssl_new(fd, ssl_ctx);
    if (!ssl) {
        return -1;
    }
    ret = SSL_accept(ssl);
    if (ret <= 0) {
        fprintf(stderr, "SSL_accept failed ssl_err=%d errno=%s: ",
                SSL_get_error(ssl, ret), strerror(errno));
        ERR_print_errors_fp(stderr);
        fprintf(stderr, "\n");
        return -1;
    }
    printf("TLS connection accepted.\n");

    FD_ZERO(&read_set);
    FD_SET(fd, &read_set);

    while(1) {
        int r = pselect(fd+1, &read_set, NULL, NULL, NULL, &sa.sa_mask);
        if (exit_request) {
            printf("\nReceived interrupt, exiting gracefully\n");
			break;
		}
        if (r < 0) {
            fprintf(stderr, "error in select\n");
            return -1;
        } else if (r > 0) {
            ret = ssl_read(ssl);
            if (ret < 0) {
                return -1;
            } else if (ret == 0) {
                break;
            }
        }
    }

    ret = ssl_shutdown(ssl);
    if (ret < 0) {
        return -1;
    }

    close(fd);
    printf("Server closed TCP socket.\n");

    SSL_free(ssl);
    SSL_CTX_free(ssl_ctx);
    return 0;
}


