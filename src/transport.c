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
    ssize_t (*send)(int, const void *, size_t, int);
    int (*open)(const char *, int, ...);
    int (*dup2)(int, int);
    int (*close)(int);
    int (*fcntl)(int, int, ...);
    size_t (*fwrite)(const void *, size_t, size_t, FILE *);
    int (*socket)(int, int, int);
    int (*connect)(int, const struct sockaddr *, socklen_t);
    int (*getaddrinfo)(const char *, const char *,
                       const struct addrinfo *,
                       struct addrinfo **);
    int (*fclose)(FILE*);
    FILE* (*fdopen)(int, const char *);
    union {
        struct {
            int sock;
            char* host;
            char* port;
        } net;
        struct {
            char* path;
            FILE* stream;
            int stdout;  // Flag to indicate that stream is stdout
            int stderr;  // Flag to indicate that stream is stderr
        } file;
    };
};

static transport_t*
newTransport()
{
    transport_t *t;

    t = calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    if ((t->send = dlsym(RTLD_NEXT, "send")) == NULL) goto out;
    if ((t->open = dlsym(RTLD_NEXT, "open")) == NULL) goto out;
    if ((t->dup2 = dlsym(RTLD_NEXT, "dup2")) == NULL) goto out;
    if ((t->close = dlsym(RTLD_NEXT, "close")) == NULL) goto out;
    if ((t->fcntl = dlsym(RTLD_NEXT, "fcntl")) == NULL) goto out;
    if ((t->fwrite = dlsym(RTLD_NEXT, "fwrite")) == NULL) goto out;
    if ((t->socket = dlsym(RTLD_NEXT, "socket")) == NULL) goto out;
    if ((t->connect = dlsym(RTLD_NEXT, "connect")) == NULL) goto out;
    if ((t->getaddrinfo = dlsym(RTLD_NEXT, "getaddrinfo")) == NULL) goto out;
    if ((t->fclose = dlsym(RTLD_NEXT, "fclose")) == NULL) goto out;
    if ((t->fdopen = dlsym(RTLD_NEXT, "fdopen")) == NULL) goto out;
    return t;

  out:
    DBG("send=%p open=%p dup2=%p close=%p "
        "fcntl=%p fwrite=%p socket=%p connect=%p "
        "getaddrinfo=%p fclose=%p fdopen=%p",
        t->send, t->open, t->dup2, t->close,
        t->fcntl, t->fwrite, t->socket, t->connect,
        t->getaddrinfo, t->fclose, t->fdopen);
    free(t);
    return NULL;
}

/*
 * Some apps require that a set of fds, usually low numbers, 0-20,
 * must exist. Therefore, we don't want to allow the kernel to
 * give us the next available fd. We need to place the fd in a
 * range that is likely not to affect an app. 
 *
 * We look for an available fd starting at a relatively high
 * range and work our way down until we find one we can get.
 * Then, we force the use of the availabale fd. 
 */
static int
placeDescriptor(int fd, transport_t *t)
{
    if (!t) return -1;

    int i, dupfd;

    for (i = DEFAULT_FD; i >= DEFAULT_MIN_FD; i--) {
        if ((t->fcntl(i, F_GETFD) == -1) && (errno == EBADF)) {
            // This fd is available
            if ((dupfd = t->dup2(fd, i)) == -1) continue;
            t->close(fd);
            return dupfd;
        }
    }
    DBG("%d", t->type);
    return -1;
}

int
transportConnection(transport_t *trans)
{
    if (!trans) return 0;

    return trans->net.sock;
}

int
transportNeedsConnection(transport_t *trans)
{
    if (!trans) return 0;
    return (trans->type == CFG_TCP || trans->type == CFG_UDP) && (trans->net.sock == -1);
}

int
transportDisconnect(transport_t *trans)
{
    if (!trans) return 0;
    if (trans->type == CFG_TCP || trans->type == CFG_UDP) {
        if (trans->net.sock != -1) trans->close(trans->net.sock);
        trans->net.sock = -1;
    }
    return 0;
}

int
transportConnect(transport_t *trans)
{
    // We're already connected.  Do nothing.
    if (!transportNeedsConnection(trans)) return 1;

    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;     // IPv4 or IPv6
    switch (trans->type) {
        case CFG_TCP:
            hints.ai_socktype = SOCK_STREAM; // For TCP
            hints.ai_protocol = IPPROTO_TCP; // For TCP
            break;
        case CFG_UDP:
            hints.ai_socktype = SOCK_DGRAM;  // For UDP
            hints.ai_protocol = IPPROTO_UDP; // For UDP
            break;
        default:
            DBG(NULL);
            return 1;
    }

    if (trans->getaddrinfo(trans->net.host,
                           trans->net.port,
                           &hints, &addr_list)) return 0;

    // Loop through the addresses until one works
    struct addrinfo* addr;
    for (addr = addr_list; addr; addr = addr->ai_next) {
        trans->net.sock = trans->socket(addr->ai_family,
                                        addr->ai_socktype,
                                        addr->ai_protocol);

        if (trans->net.sock == -1) continue;

        if (trans->connect(trans->net.sock,
                           addr->ai_addr,
                           addr->ai_addrlen) == -1) {

            // We could create a sock, but not connect.  Clean up.
            transportDisconnect(trans);
            continue;
        }

        break; // Success!
    }

    if (addr_list) freeaddrinfo(addr_list);

    return (trans->net.sock != -1);
}

transport_t *
transportCreateTCP(const char *host, const char *port)
{
    transport_t* trans = NULL;

    if (!host || !port) return trans;

    trans = newTransport();
    if (!trans) return trans;

    trans->type = CFG_TCP;
    trans->net.sock = -1;
    trans->net.host = strdup(host);
    trans->net.port = strdup(port);

    if (!trans->net.host || !trans->net.port) {
        DBG(NULL);
        transportDestroy(&trans);
        return trans;
    }

    if (!transportConnect(trans)) return trans;

    // Move this descriptor up out of the way
    trans->net.sock = placeDescriptor(trans->net.sock, trans);
    if (trans->net.sock == -1) return trans;

    // Set the socket to close on exec
    int flags = trans->fcntl(trans->net.sock, F_GETFD, 0);
    if (trans->fcntl(trans->net.sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
        DBG("%d %s %s", trans->net.sock, host, port);
    }

    return trans;
}

transport_t*
transportCreateUdp(const char* host, const char* port)
{
    transport_t* t = NULL;

    if (!host || !port) return t;

    t = newTransport();
    if (!t) return t;

    t->type = CFG_UDP;
    t->net.sock = -1;
    t->net.host = strdup(host);
    t->net.port = strdup(port);

    if (!t->net.host || !t->net.port) {
        DBG(NULL);
        transportDestroy(&t);
        return t;
    }

    if (!transportConnect(t))  return t;

    // Move this descriptor up out of the way
    t->net.sock = placeDescriptor(t->net.sock, t);
    if (t->net.sock == -1) return t;

    // Set the socket to non blocking, and close on exec
    int flags = t->fcntl(t->net.sock, F_GETFL, 0);
    if (t->fcntl(t->net.sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        DBG("%d %s %s", t->net.sock, host, port);
    }

    flags = t->fcntl(t->net.sock, F_GETFD, 0);
    if (t->fcntl(t->net.sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
        DBG("%d %s %s", t->net.sock, host, port);
    }

    return t;
}

transport_t*
transportCreateFile(const char* path, cfg_buffer_t buf_policy)
{
    transport_t *t;

    if (!path) return NULL;
    t = newTransport();
    if (!t) return NULL; 

    t->type = CFG_FILE;
    t->file.path = strdup(path);
    if (!t->file.path) {
        DBG("%s", path);
        transportDestroy(&t);
        return t;
    }

    // See if path is "stdout" or "stderr"
    t->file.stdout = !strcmp(path, "stdout");
    t->file.stderr = !strcmp(path, "stderr");

    // if stdout/stderr, set stream and skip everything else in the function.
    if (t->file.stdout) {
        t->file.stream = stdout;
        return t;
    } else if (t->file.stderr) {
        t->file.stream = stderr;
        return t;
    }

    int fd;
    fd = t->open(t->file.path, O_CREAT|O_WRONLY|O_APPEND|O_CLOEXEC, 0666);
    if (fd == -1) {
        DBG("%s", path);
        transportDestroy(&t);
        return t;
    }

    // Move this descriptor up out of the way
    if ((fd = placeDescriptor(fd, t)) == -1) {
        transportDestroy(&t);
        return t;
    }

    // Needed because umask affects open permissions
    if (fchmod(fd, 0666) == -1) {
        DBG("%d %s", fd, path);
    }

    // set close on exec
    int flags = t->fcntl(fd, F_GETFD, 0);
    if (t->fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
        DBG("%d %s", fd, path);
    }

    FILE* f;
    if (!(f = t->fdopen(fd, "a"))) {
        transportDestroy(&t);
        return t;
    }
    t->file.stream = f;

    // Fully buffer the output unless we're told not to.
    // I expect line buffering to be useful when we're debugging crashes or
    // or if many applications are configured to write to the same files.
    int buf_mode = _IOFBF;
    switch (buf_policy) {
        case CFG_BUFFER_FULLY:
            buf_mode = _IOFBF;
            break;
        case CFG_BUFFER_LINE:
            buf_mode = _IOLBF;
            break;
        default:
            DBG("%d", buf_policy);
    }
    if (setvbuf(t->file.stream, NULL, buf_mode, BUFSIZ)) {
        DBG(NULL);
    }

    return t;
}

transport_t*
transportCreateUnix(const char* path)
{
    if (!path) return NULL;
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    t->type = CFG_UNIX;

    return t;
}

transport_t*
transportCreateSyslog(void)
{
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    t->type = CFG_SYSLOG;

    return t;
}

transport_t*
transportCreateShm()
{
    transport_t* t = calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

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
        case CFG_TCP:
            transportDisconnect(t);
            if (t->net.host) free (t->net.host);
            if (t->net.port) free (t->net.port);
            break;
        case CFG_UNIX:
            break;
        case CFG_FILE:
            if (t->file.path) free(t->file.path);
            if (!t->file.stdout && !t->file.stderr) {
                // if stdout/stderr, we didn't open stream, so don't close it
                if (t->file.stream) t->fclose(t->file.stream);
            }
            break;
        case CFG_SYSLOG:
            break;
        case CFG_SHM:
            break;
        default:
            DBG("%d", t->type);
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
            if (t->net.sock != -1) {
                if (!t->send) {
                    DBG(NULL);
                    break;
                }
                int rc = t->send(t->net.sock, msg, strlen(msg), 0);

                if (rc < 0) {
                    switch (errno) {
                    case EBADF:
                        DBG(NULL);
                        return DEFAULT_BADFD;
                        break;
                    case EWOULDBLOCK:
                        DBG(NULL);
                        break;
                    default:
                        DBG(NULL);
                    }
                }
            }
            break;
        case CFG_TCP:
            if (t->net.sock != -1) {
                if (!t->send) {
                    DBG(NULL);
                    break;
                }
#ifdef __LINUX__
                int rc = t->send(t->net.sock, msg, strlen(msg), MSG_NOSIGNAL);
#else
                int rc = t->send(t->net.sock, msg, strlen(msg), 0);
#endif

                if (rc < 0) {
                    switch (errno) {
                    case EBADF:
                        DBG(NULL);
                        return DEFAULT_BADFD;
                        break;
                    default:
                        DBG(NULL);
                    }
                }
            }
            break;
        case CFG_FILE:
            if (t->file.stream) {
                size_t msg_size = strlen(msg);
                int bytes = t->fwrite(msg, 1, msg_size, t->file.stream);
                if (bytes != msg_size) {
                    if (errno == EBADF) {
                        DBG("%d %d", bytes, msg_size);
                        return DEFAULT_BADFD;
                    }
                    DBG("%d %d", bytes, msg_size);
                    return -1;
                }
            }
            break;
        case CFG_UNIX:
        case CFG_SYSLOG:
        case CFG_SHM:
            return -1;
        default:
            DBG("%d", t->type);
            return -1;
    }
     return 0;
}

int
transportFlush(transport_t* t)
{
    if (!t) return -1;

    switch (t->type) {
        case CFG_UDP:
        case CFG_TCP:
            break;
        case CFG_FILE:
            if (fflush(t->file.stream) == EOF) {
                DBG(NULL);
            }
            break;
        case CFG_UNIX:
        case CFG_SYSLOG:
        case CFG_SHM:
            return -1;
        default:
            DBG("%d", t->type);
            return -1;
    }
    return 0;
}

