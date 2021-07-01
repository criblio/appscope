#define _GNU_SOURCE
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include "openssl/ssl.h"

// The openssl library itself:
// make contrib/openssl/libssl.a
//
// Dynamically:
// gcc -g -o tlstest -Icontrib/openssl/include test/manual/tlstest.c -Lcontrib/openssl -lssl -lcrypto && LD_LIBRARY_PATH=contrib/openssl ./tlstest
//
// Statically:
// gcc -g -o tlstest -Icontrib/openssl/include test/manual/tlstest.c contrib/openssl/libssl.a contrib/openssl/libcrypto.a -lpthread -ldl && ./tlstest
//
// To create tlstest.pem referenced below:
// SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt LD_LIBRARY_PATH=contrib/openssl contrib/openssl/apps/openssl s_client -showcerts -connect www.random.org:443 > tlstest.pem


// This was largely based on: https://wiki.openssl.org/index.php/SSL/TLS_Client
#define HOST_NAME "www.random.org"
#define HOST_PORT "443"
#define HOST_RESOURCE "/cgi-bin/randbyte?nbytes=32&format=h"

#define handleFailure() { printf("%s:%d\n", __FILE__, __LINE__); exit(1); }


int
createSocket(const char *host, const char *port)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;     // IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM; // For TCP
    hints.ai_protocol = IPPROTO_TCP; // For TCP

    // get a list of possible addresses for host and port
    if (getaddrinfo(host, port, &hints, &addr_list)) return -1;

    struct addrinfo* addr;
    int sock = -1;
    for (addr = addr_list; addr; addr = addr->ai_next) {
        // Create a socket
        sock = socket(addr->ai_family,
                             addr->ai_socktype,
                             addr->ai_protocol);
        if (sock == -1) continue;

        // Try to connect it
        if (!connect(sock, addr->ai_addr, addr->ai_addrlen)) {
            // sock connected successfully!  Stop looping.
            break;
        }

        // the latest connect attempt wasn't successful...
        close(sock);
        sock = -1;
    }

    if (addr_list) freeaddrinfo(addr_list);
    return sock;
}


int
main(void)
{
    printf("tlstest.c\n");

    SSL_CTX* ctx = NULL;
    SSL *ssl = NULL;

    const SSL_METHOD* method = TLS_method();
    if(!method) handleFailure();

    ctx = SSL_CTX_new(method);
    if(!ctx) handleFailure();

    // prohibits SSLv3 too
    // May be useful when we support Minimum TLS version, Maximum TLS version.
    //
    // long flags = SSL_CTX_get_options(ctx);
    // printf("option flags before tweaking = %lx\n", flags);
    // flags |= SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION;
    // printf("option flags after tweaking = %lx\n", flags);
    // SSL_CTX_set_options(ctx, flags);

    // enables partial write, to act like write
    long mode = SSL_CTX_get_mode(ctx);
    printf("mode flags before tweaking = %lx\n", mode);
    mode |= SSL_MODE_ENABLE_PARTIAL_WRITE | SSL_MODE_AUTO_RETRY;
    printf("mode flags after tweaking = %lx\n", mode);
    SSL_CTX_set_mode(ctx, mode);

    // This default seems ok for linux ... or does it?
    // https://serverfault.com/questions/62496/ssl-certificate-location-on-unix-linux
    long res = SSL_CTX_load_verify_locations(ctx, "/etc/ssl/certs/ca-certificates.crt",
                                                  "/etc/ssl/certs/");
//    long res = SSL_CTX_load_verify_locations(ctx, "./tlstest.pem",
//                                                  "/etc/ssl/certs/");
    if (res != 1) handleFailure();

    // May be useful for mutual auth.
    //SSL_CTX_use_certificate_file(ctx, SSL_CLIENT_CRT, SSL_FILETYPE_PEM)
    //SSL_CTX_use_PrivateKey_file(ctx, SSL_CLIENT_KEY, SSL_FILETYPE_PEM)
    //SSL_CTX_check_private_key(ctx);

    // May be useful for SNI (Server Name Indication)
    //SSL_set_tlsext_host_name(ssl, HOST_NAME);

    ssl = SSL_new(ctx);
    if (!ssl) handleFailure();

    int sock = createSocket(HOST_NAME, HOST_PORT);
    if (sock == -1) handleFailure();

    int setfd = SSL_set_fd(ssl, sock);
    if (!setfd) handleFailure();

    int connect = SSL_connect(ssl);
    if (connect != 1) handleFailure();

    /* Step 1: verify a server certificate was presented during the negotiation */
    X509* cert = SSL_get_peer_certificate(ssl);
    if(cert) { X509_free(cert); } /* Free immediately */
    if(NULL == cert) handleFailure();

    /* Step 2: verify the result of chain verification */
    /* Verification performed according to RFC 4158    */
    res = SSL_get_verify_result(ssl);
    printf("SSL_get_verify_result = %s\n", X509_verify_cert_error_string(res));
    if (!(X509_V_OK == res)) {
        handleFailure();
    }

    char *request = NULL;
    int req_size = asprintf(&request, "GET " HOST_RESOURCE " HTTP/1.1\r\n"
                                     "Host: " HOST_NAME "\r\n"
                                     "Connection: close\r\n\r\n");

    int byte_count = SSL_write(ssl, request, req_size);
    printf("%d of %d bytes written with SSL_write\n", byte_count, req_size);
    free(request);

    int len = 0;
    do {
        char buf[1537] = {};
        len = SSL_read(ssl, buf, sizeof(buf)-1);
        buf[len]='\0';
        printf("%s\n", buf);

    } while (len > 0);

    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    close(sock);

    return 0;
}

