#include "wrap.h"

#define DEBUG 0
#define EXPORT __attribute__((visibility("default")))

// Use these only if a config file is not accesible
#define PORT 8125
#define SERVER "127.0.0.1"

#define STATSD_READ "cribl.scope.calls.read.bytes:%d|c\n"
#define STATSD_WRITE "cribl.scope.calls.write.bytes:%d|c\n"
#define STATSD_VSYSLOG "cribl.scope.calls.vsyslog|c\n"
#define STATSD_SOCKET "cribl.scope.calls.socket|c\n"
#define STATSD_SEND "cribl.scope.calls.send.bytes:%d|c\n"
#define STATSD_SENDTO "cribl.scope.calls.sendto.bytes:%d|c\n"
#define STATSD_SENDMSG "cribl.scope.calls.sendmsg|c\n"
#define STATSD_RECV "cribl.scope.calls.recv.bytes:%d|c\n"
#define STATSD_RECVFROM "cribl.scope.calls.recvfrom.bytes:%d|c\n"
#define STATSD_RECVMSG "cribl.scope.calls.recvmsg|c\n"

static void (*real_vsyslog)(int, const char *, va_list);
static int (*real_socket)(int, int, int);
static ssize_t (*real_read)(int, void *, size_t);
static ssize_t (*real_write)(int, const void *, size_t);
static ssize_t (*real_send)(int, const void *, size_t, int);
static ssize_t (*real_sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
static ssize_t (*real_sendmsg)(int, const struct msghdr *, int);
static ssize_t (*real_recv)(int, void *, size_t, int);
static ssize_t (*real_recvfrom)(int sockfd, void *buf, size_t len, int flags,
                                struct sockaddr *src_addr, socklen_t *addrlen);
static ssize_t (*real_recvmsg)(int, struct msghdr *, int);
    
static int sock = 0;
static struct sockaddr_in saddr;
static operations_info ops;

static void initSocket(void)
{
    int flags;
    char server[sizeof(SERVER) + 1];

    // Create a UDP socket
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0)	{
        perror("socket");
    }

    // Set the socket to non blocking
    flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    // Create the address to send to
    strncpy(server, SERVER, sizeof(SERVER));
        
    memset(&saddr, 0, sizeof(saddr));
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(PORT);
    if (inet_aton(server, &saddr.sin_addr) == 0) {
        perror("inet_aton");
    }
}

static void postMetric(const char *metric)
{
    ssize_t rc;
    
    rc = real_sendto(sock, metric, strlen(metric), 0, 
                (struct sockaddr *)&saddr, sizeof(saddr));
    if (rc < 0) {
        switch (errno) {
        case EWOULDBLOCK:
            ops.udp_blocks++;
            break;
        default:
            ops.udp_errors++;
        }
    }
}

__attribute__((constructor)) void init(void)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "constructor\n", 12);
    real_vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    real_send = dlsym(RTLD_NEXT, "send");
    real_sendto = dlsym(RTLD_NEXT, "sendto");
    real_sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    real_socket = dlsym(RTLD_NEXT, "socket");
    real_read = dlsym(RTLD_NEXT, "read");
    real_write = dlsym(RTLD_NEXT, "write");
    real_recv = dlsym(RTLD_NEXT, "recv");
    real_recvfrom = dlsym(RTLD_NEXT, "recvfrom");
    real_recvmsg = dlsym(RTLD_NEXT, "recvmsg");

    initSocket();
}

EXPORT
ssize_t write(int fd, const void *buf, size_t count)
{
    if (real_write == NULL) {
        return -1;
    }

    // Don't init the socket from write; starts early
    // Delay posts until init is complete
    if (sock != 0) {
        char metric[strlen(STATSD_WRITE) + 16];
        
        snprintf(metric, sizeof(metric), STATSD_WRITE, (int)count);
        postMetric(metric);
    }

    return real_write(fd, buf, count);
}

EXPORT
ssize_t read(int fd, void *buf, size_t count)
{
    char metric[strlen(STATSD_READ) + 16];
    
    if (sock == 0) {
        initSocket();
    }
    
    if (real_read == NULL) {
        write(STDERR_FILENO, "ERROR: read\n", 12);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_READ, (int)count) <= 0) {
        write(STDERR_FILENO, "ERROR: read: string compose\n", 29);
    }
    
    postMetric(metric);
    return real_read(fd, buf, count);
}

EXPORT
void vsyslog(int priority, const char *format, va_list ap)
{
    char metric[strlen(STATSD_VSYSLOG)];
    
    if (sock == 0) {
        initSocket();
    }

    if (DEBUG > 0) write(STDOUT_FILENO, "vsyslog\n", 8);

    if (real_vsyslog == NULL) {
        write(STDERR_FILENO, "ERROR: vsyslog\n", 15);
        return;
    }

    if (snprintf(metric, sizeof(metric), STATSD_VSYSLOG) <= 0) {
        write(STDERR_FILENO, "ERROR: vsyslog: string compose\n", 29);
    }
    
    postMetric(metric);
    real_vsyslog(priority, format, ap);
    return;
}

EXPORT
int socket(int socket_family, int socket_type, int protocol)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "socket\n", 7);

    if (real_socket == NULL) {
        write(STDERR_FILENO, "ERROR: socket\n", 14);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_SOCKET)];

        if (snprintf(metric, sizeof(metric), STATSD_SOCKET) <= 0) {
            write(STDERR_FILENO, "ERROR: socket: string compose\n", 26);
        }
    
        postMetric(metric);
    }

    return real_socket(socket_family, socket_type, protocol);    
}

EXPORT
ssize_t send(int sockfd, const void *buf, size_t len, int flags)
{
    char metric[strlen(STATSD_SEND) + 16];

    if (sock == 0) {
        initSocket();
    }
    
    if (DEBUG > 0) write(STDOUT_FILENO, "send\n", 5);

    if (real_send == NULL) {
        write(STDERR_FILENO, "ERROR: send\n", 12);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_SEND, (int)len) <= 0) {
        write(STDERR_FILENO, "ERROR: send: string compose\n", 28);
    }
    
    postMetric(metric);

    return real_send(sockfd, buf, len, flags);
}

EXPORT
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "sendto\n", 7);

    if (real_sendto == NULL) {
        write(STDERR_FILENO, "ERROR: sendto\n", 14);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_SENDTO) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_SENDTO, (int)len) <= 0) {
            write(STDERR_FILENO, "ERROR: sendto: string compose\n", 30);
        }
    
        postMetric(metric);
    }

    return real_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

EXPORT
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "sendmsg\n", 8);

    if (real_sendmsg == NULL) {
        write(STDERR_FILENO, "ERROR: sendmsg\n", 14);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_SENDMSG) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_SENDMSG) <= 0) {
            write(STDERR_FILENO, "ERROR: sendmsg: string compose\n", 31);
        }
    
        postMetric(metric);
    }

    return real_sendmsg(sockfd, msg, flags);
}

EXPORT
ssize_t recv(int sockfd, void *buf, size_t len, int flags)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "recv\n", 5);

    if (real_recv == NULL) {
        write(STDERR_FILENO, "ERROR: recv\n", 12);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_RECV) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_RECV, (int)len) <= 0) {
            write(STDERR_FILENO, "ERROR: recv: string compose\n", 28);
        }
    
        postMetric(metric);
    }

    return real_recv(sockfd, buf, len, flags);
}

EXPORT
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "recvfrom\n", 9);

    if (real_recvfrom == NULL) {
        write(STDERR_FILENO, "ERROR: recvfrom\n", 16);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_RECVFROM) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_RECVFROM, (int)len) <= 0) {
            write(STDERR_FILENO, "ERROR: recvfrom: string compose\n", 32);
        }
    
        postMetric(metric);
    }

    return real_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
}

EXPORT
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    if (DEBUG > 0) write(STDOUT_FILENO, "recvmsg\n", 8);

    if (real_recvmsg == NULL) {
        write(STDERR_FILENO, "ERROR: recvmsg\n", 12);
        return -1;
    }

    if (sock != 0) {
        char metric[strlen(STATSD_RECVMSG) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_RECVMSG) <= 0) {
            write(STDERR_FILENO, "ERROR: recvmsg: string compose\n", 31);
        }
    
        postMetric(metric);
    }

    return real_recvmsg(sockfd, msg, flags);
}
