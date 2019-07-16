#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "scopetypes.h"
#include "transport.h"

typedef struct operations_info_t {
    unsigned int udp_blocks;
    unsigned int udp_errors;
    unsigned int init_errors;
    unsigned int interpose_errors;
    char *errMsg[64];
} operations_info;

struct _transport_t
{
    cfg_transport_t type;
    union {
        struct {
            int sock;
            struct sockaddr_in saddr;
            operations_info ops;
        } udp;
        struct {
            char* path;
            int fd;
        } file;
    };

    // These fields are used to avoid infinite recursion since we call
    // write and sendto from write and sendto.
    //
    // We *could* remove them and use fields from g_fn from wrap.c instead.
    // However, I don't want to do this because it would create a dependency
    // from transport to wrap.  (A dep the other way is fine)
    ssize_t (*write)(int, const void *, size_t);
    ssize_t (*sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
};

transport_t*
transportCreateUdp(const char* host, int port)
{
    if (!host) return NULL;
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->type = CFG_UDP;

    // Create a UDP socket
    t->udp.sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (t->udp.sock == -1)     {
        transportDestroy(&t);
        return t;
    }

    // Set the socket to non blocking, and close on exec
    int flags = fcntl(t->udp.sock, F_GETFL, 0);
    if (fcntl(t->udp.sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        // TBD do something here too.
    }
    flags = fcntl(t->udp.sock, F_GETFD, 0);
    if (fcntl(t->udp.sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
        // TBD do something here.
    }

    // Create the address to send to
    memset(&t->udp.saddr, 0, sizeof(t->udp.saddr));
    t->udp.saddr.sin_family = AF_INET;
    t->udp.saddr.sin_port = htons(port);
    if (inet_aton(host, &t->udp.saddr.sin_addr) == 0) {
        close(t->udp.sock);
        free(t);
        return NULL;
    }
    return t;
}

transport_t*
transportCreateFile(const char* path)
{
    if (!path) return NULL;
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->type = CFG_FILE;
    t->file.path = strdup(path);
    if (!t->file.path) {
        transportDestroy(&t);
        return t;
    }

    t->file.fd = open(t->file.path, O_CREAT|O_RDWR|O_APPEND|O_CLOEXEC, 0666);
    if (t->file.fd == -1) {
        transportDestroy(&t);
        return t;
    }
    return t;
}

transport_t*
transportCreateUnix(const char* path)
{
    if (!path) return NULL;
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->type = CFG_UNIX;

    return t;
}

transport_t*
transportCreateSyslog(void)
{
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->type = CFG_SYSLOG;

    return t;
}

transport_t*
transportCreateShm()
{
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->type = CFG_SHM;

    return t;
}

void
transportDestroy(transport_t** transport)
{
    if (!transport || !*transport) return;

    transport_t* t = *transport;
    switch (t->type) {
        case CFG_UDP:
            if (t->udp.sock != -1) close(t->udp.sock);
            break;
        case CFG_UNIX:
            break;
        case CFG_FILE:
            if (t->file.path) free(t->file.path);
            if (t->file.fd != -1) close(t->file.fd);
            break;
        case CFG_SYSLOG:
            break;
        case CFG_SHM:
            break;
    }
    free(t);
    *transport = NULL;
}

int
transportSend(transport_t* t, const char* msg)
{
    if (!t || !msg) return -1;

    // Use these to avoid infinite recursion...
    if (!t->write) t->write = dlsym(RTLD_NEXT, "write");
    if (!t->sendto) t->sendto = dlsym(RTLD_NEXT, "sendto");
    if (!t->write || !t->sendto) return -1;

    switch (t->type) {
        case CFG_UDP:
            if (t->udp.sock != -1) {
                int rc = t->sendto(t->udp.sock, msg, strlen(msg), 0,
                                 (struct sockaddr *)&t->udp.saddr, sizeof(t->udp.saddr));
                if (rc < 0) {
                    switch (errno) {
                    case EWOULDBLOCK:
                        t->udp.ops.udp_blocks++;
                        break;
                    default:
                        t->udp.ops.udp_errors++;
                    }
                }
            }
            break;
        case CFG_FILE:
            if (t->file.fd != -1) {
                int bytes = t->write(t->file.fd, msg, strlen(msg));
                if (bytes < 0) {
                    // TBD do something here
                } else {
                    fsync(t->file.fd);
                }
            }
            break;
        case CFG_UNIX:
        case CFG_SYSLOG:
        case CFG_SHM:
            return -1;
            break;
    }
     return 0;
}
