/* 
 * tcpclient.c - A simple TCP client
 *
 * usage: tcpclient [OPTIONS] PORT
 * options: -t, --tls       use TLSv1.3
 *
 * requires: libssl-dev
 * build: gcc -g test/manual/tcpclient.c -lpthread -lssl -lcrypto -o tcpclient
 *
 */
#define _POSIX_C_SOURCE 1
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <errno.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <signal.h>
#include <strings.h>
#include <getopt.h>
#include <stdbool.h>

int socket_setup(int port);
int tcp_ssl(int socket);

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

    if(!tls) {
        printf("Currently supports -t only\n");
        exit (EXIT_FAILURE);
    }
    
    int socket = socket_setup(port);
    if (tls) {
        if (!tcp_ssl(socket)) {
            exit (EXIT_FAILURE);
        }
    } else {
        // tcp not yet supported
    }

    exit(EXIT_SUCCESS);
}

int
socket_setup(int port)
{
	// create socket
    int parent_fd;
	parent_fd = socket(AF_INET , SOCK_STREAM , 0);
	if (parent_fd < 0)	{
        fprintf(stderr, "Error opening socket\n");
        return -1;
	}

    // server properties
	struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = htonl(INADDR_ANY);
    server.sin_port = htons((unsigned short)port);

	// connect to remote server
	if (connect(parent_fd , (struct sockaddr *)&server , sizeof(server)) < 0)
	{
        fprintf(stderr, "Connection failed\n");
        return -1;
	}
    printf("TCP connected.\n");

    return parent_fd;
}

int
ssl_shutdown(SSL* ssl) {
    int ret;
    ret = SSL_shutdown(ssl);
    if (ret < 0) {
        int ssl_err = SSL_get_error(ssl, ret);
        fprintf(stderr, "Client SSL_shutdown failed: ssl_err=%d\n", ssl_err);
        return -1;
    }
    printf("Client shut down a TLS session.\n");
    if (ret != 1) {
        ret = SSL_shutdown(ssl);
        if (ret != 1) {
            int ssl_err = SSL_get_error(ssl, ret);
            fprintf(stderr,
                    "Waiting for server shutdown using SSL_shutdown failed: "
                    "ssl_err=%d\n", ssl_err);
            return -1;
        }
    }
    printf("Client thinks a server shut down the TLS session.\n");
    return 0;
}

int
tcp_ssl(int socket)
{
    const char message[] = "Hello from TCP client\n";
    int retval;
    int processed;

    SSL_CTX *ssl_ctx;
    SSL *ssl;
    ssl_ctx = SSL_CTX_new(TLS_client_method());
    if (!ssl_ctx) {
        fprintf(stderr, "SSL_CTX_new failed\n");
        return -1;
    }
	SSL_CTX_set_options(ssl_ctx, SSL_OP_SINGLE_DH_USE);
    ssl = SSL_new(ssl_ctx);
    if (!ssl) {
        fprintf(stderr, "SSL_new failed\n");
        return -1;
    }
    if (!SSL_set_fd(ssl, socket)) {
        fprintf(stderr, "SSL_set_fd failed\n");
        return -1;
    }
    if (SSL_connect(ssl) <= 0) {
        fprintf(stderr, "SSL_connect failed\n");
        return -1;
    }
    printf("SSL connected.\n");

    for (const char *start = message; start - message < sizeof(message);
            start += processed) {
        processed = SSL_write(ssl, start, sizeof(message) - (start - message));
        printf("Client SSL_write returned %d\n", processed);
        if (processed <= 0) {
            int ssl_err = SSL_get_error(ssl, processed);
            if (ssl_err != SSL_ERROR_WANT_READ && ssl_err != SSL_ERROR_WANT_WRITE) {
                fprintf(stderr, "client write failed: ssl_error=%d: ", ssl_err);
                ERR_print_errors_fp(stderr);
                fprintf(stderr, "\n");
                return -1;
            }
            processed = 0;
        }
    };
    printf("Client write finished.\n");

    int ret;
    ret = ssl_shutdown(ssl);
    if (ret < 0) {
        return -1;
    }
    printf("Client shut down TLS session.\n");

    if (shutdown(socket, SHUT_RDWR)) {
        perror("client shutdown failed");
        return -1;
    }
    printf("Client shut down TCP.\n");

    SSL_free(ssl);
    SSL_CTX_free(ssl_ctx);

    return 0;
}
