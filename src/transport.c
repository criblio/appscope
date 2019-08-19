#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "dbg.h"
#include "scopetypes.h"
#include "transport.h"

struct _transport_t
{
    cfg_transport_t type;
    ssize_t (*write)(int, const void *, size_t);
    union {
        struct {
            int sock;
        } udp;
        struct {
            char* path;
            int fd;
        } file;
    };
};

transport_t*
transportCreateUdp(const char* host, const char* port)
{
    transport_t* t = NULL;
    struct addrinfo* addr_list = NULL;

    if (!host || !port) goto out;

    t = calloc(1, sizeof(transport_t));
    if (!t) goto out;

    t->type = CFG_UDP;
    t->udp.sock = -1;

    // Get some addresses to try
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;     // IPv4 or IPv6
    hints.ai_socktype = SOCK_DGRAM;  // For udp
    hints.ai_protocol = IPPROTO_UDP; // For udp
    if (getaddrinfo(host, port, &hints, &addr_list)) goto out;

    // Loop through the addresses until one works
    struct addrinfo* addr;
    for (addr = addr_list; addr; addr = addr->ai_next) {
        t->udp.sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
        if (t->udp.sock == -1) continue;
        if (connect(t->udp.sock, addr->ai_addr, addr->ai_addrlen) == -1) {
            // We could create a sock, but not connect.  Clean up.
            close(t->udp.sock);
            t->udp.sock = -1;
            continue;
        }
        break; // Success!
    }

    // If none worked, get out
    if (t->udp.sock == -1) goto out;

    // Set the socket to non blocking, and close on exec
    int flags = fcntl(t->udp.sock, F_GETFL, 0);
    if (fcntl(t->udp.sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        DBG("%d %s %s", t->udp.sock, host, port);
    }
    flags = fcntl(t->udp.sock, F_GETFD, 0);
    if (fcntl(t->udp.sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
        DBG("%d %s %s", t->udp.sock, host, port);
    }

out:
    if (addr_list) freeaddrinfo(addr_list);
    if (t && t->udp.sock == -1) transportDestroy(&t);
    return t;
}

transport_t*
transportCreateFile(const char* path)
{
    if (!path) return NULL;
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->write = dlsym(RTLD_NEXT, "write");
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
    } else {
        // Needed because umask affects open permissions
        if (fchmod(t->file.fd, 0666) == -1) {
            DBG("%s", path);
        }
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

    switch (t->type) {
        case CFG_UDP:
            if (t->udp.sock != -1) {
                int rc = send(t->udp.sock, msg, strlen(msg), 0);
                if (rc < 0) {
                    switch (errno) {
                    case EWOULDBLOCK:
                        DBG(NULL);
                        break;
                    default:
                        DBG(NULL);
                    }
                }
            }
            break;
        case CFG_FILE:
            if (!t->write) {
                DBG(NULL);
                break;
            }
            if (t->file.fd != -1) {
                int bytes = t->write(t->file.fd, msg, strlen(msg));
                if (bytes < 0) {
                    DBG("%d %d", t->file.fd, bytes);
                } else {
                    if (fsync(t->file.fd) == -1) {
                        DBG("%d", t->file.fd);
                    }
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
