#include "wrap.h"

static int g_sock = 0;
static struct sockaddr_in g_saddr;
static operations_info g_ops;
static net_info *g_netinfo;
static int g_numNinfo;
static char g_hostname[MAX_HOSTNAME];
static char g_procname[MAX_PROCNAME];
static int g_openPorts = 0;
static int g_activeConnections = 0;
static interposed_funcs g_fn;

// These need to come from a config file
#define LOG_FILE 1  // eventually an enum for file, syslog, shared memory 
static bool g_log = TRUE;
static const char g_logFile[] = "/tmp/scope.log";
static unsigned int g_logOp = LOG_FILE;
static int g_logfd = -1;

static
void scopeLog(char *msg, int fd)
{
    size_t len;
    
    if ((g_log == FALSE) || (!msg)) {
        return;
    }

    if (g_logOp & LOG_FILE) {
        char buf[strlen(msg) + 128];
        
        if ((g_logfd == -1) && 
            (strlen(g_logFile) > 0)) {
                g_logfd = open(g_logFile, O_RDWR|O_APPEND);
        }

        len = sizeof(buf) - strlen(buf);
        snprintf(buf, sizeof(buf), "Scope: %s(%d): ", g_procname, fd);
        strncat(buf, msg, len);
        g_fn.write(g_logfd, buf, strlen(buf));
    }        
}

static
void initSocket(void)
{
    int flags;
    char server[sizeof(SERVER) + 1];

    // Create a UDP socket
    g_sock = g_fn.socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0)	{
        scopeLog("ERROR: initSocket:socket\n", -1);
    }

    // Set the socket to non blocking
    flags = fcntl(g_sock, F_GETFL, 0);
    fcntl(g_sock, F_SETFL, flags | O_NONBLOCK);

    // Create the address to send to
    strncpy(server, SERVER, sizeof(SERVER));
        
    memset(&g_saddr, 0, sizeof(g_saddr));
    g_saddr.sin_family = AF_INET;
    g_saddr.sin_port = htons(PORT);
    if (inet_aton(server, &g_saddr.sin_addr) == 0) {
        scopeLog("ERROR: initSocket:inet_aton\n", -1);
    }
}

static
void postMetric(const char *metric)
{
    ssize_t rc;

    if (g_fn.socket == 0) {
        initSocket();
    }

    scopeLog((char *)metric, -1);
    if (g_fn.sendto) {
        rc = g_fn.sendto(g_sock, metric, strlen(metric), 0, 
                         (struct sockaddr *)&g_saddr, sizeof(g_saddr));
        if (rc < 0) {
            scopeLog("ERROR: sendto\n", g_sock);
            switch (errno) {
            case EWOULDBLOCK:
                g_ops.udp_blocks++;
                break;
            default:
                g_ops.udp_errors++;
            }
        }
    }
}

static
void addSock(int fd, int type)
{
    if (g_netinfo) {
        if (g_netinfo[fd].fd == fd) {
            scopeLog("addSock: duplicate\n", fd);

            if (g_openPorts > 0) {
                atomicSub(&g_openPorts, 1);
            }
            
            scopeLog("decr\n", g_openPorts);
            return;
        }
        
        if ((fd > g_numNinfo) && (fd < MAX_FDS))  {
            // Need to realloc
            if ((g_netinfo = realloc(g_netinfo, sizeof(struct net_info_t) * fd)) == NULL) {
                scopeLog("ERROR: addSock:realloc\n", fd);
            }
            g_numNinfo = fd;
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].fd = fd;
        g_netinfo[fd].type = type;
        scopeLog("addSock\n", fd);
    }
}

static
int getProtocol(int type, char *proto, size_t len)
{
    if (!proto) {
        return -1;
    }
    
    if (type & SOCK_STREAM) {
        strncpy(proto, "TCP", len);
    } else if (type & SOCK_DGRAM) {
        strncpy(proto, "UDP", len);
    } else {
        strncpy(proto, "OTHER", len);
    }

    return 0;
}

static
void doOpenPorts(int fd)
{
    char proto[PROTOCOL_STR];
    char metric[strlen(STATSD_OPENPORTS) +
                sizeof(unsigned int) +
                strlen(g_hostname) +
                strlen(g_procname) +
                PROTOCOL_STR  +
                sizeof(unsigned int) +
                sizeof(unsigned int) +
                sizeof(unsigned int) + 1];
        
    getProtocol(g_netinfo[fd].type, proto, sizeof(proto));
    
    if (snprintf(metric, sizeof(metric), STATSD_OPENPORTS,
                 g_openPorts, g_procname, getpid(), fd, g_hostname, proto,
                 g_netinfo[fd].port) <= 0) {
        scopeLog("ERROR: doOpenPorts:snprintf\n", -1);
    }
    
    postMetric(metric);
}

__attribute__((constructor)) void init(void)
{
    g_fn.vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.socket = dlsym(RTLD_NEXT, "socket");
    g_fn.shutdown = dlsym(RTLD_NEXT, "shutdown");
    g_fn.listen = dlsym(RTLD_NEXT, "listen");
    g_fn.accept = dlsym(RTLD_NEXT, "accept");
    g_fn.accept4 = dlsym(RTLD_NEXT, "accept4");
    g_fn.bind = dlsym(RTLD_NEXT, "bind");
    g_fn.send = dlsym(RTLD_NEXT, "send");
    g_fn.sendto = dlsym(RTLD_NEXT, "sendto");
    g_fn.sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    g_fn.recv = dlsym(RTLD_NEXT, "recv");
    g_fn.recvfrom = dlsym(RTLD_NEXT, "recvfrom");
    g_fn.recvmsg = dlsym(RTLD_NEXT, "recvmsg");

#ifdef __MACOS__
    g_fn.close$NOCANCEL = dlsym(RTLD_NEXT, "close$NOCANCEL");
    g_fn.close_nocancel = dlsym(RTLD_NEXT, "close_nocancel");
    g_fn.guarded_close_np = dlsym(RTLD_NEXT, "guarded_close_np");
#endif // __MACOS__

    if ((g_netinfo = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc\n", -1);
    }

    g_numNinfo = NET_ENTRIES;
    if (gethostname(g_hostname, sizeof(g_hostname)) != 0) {
        scopeLog("ERROR: Constructor:gethostname\n", -1);
    }

    osGetProcname(g_procname, sizeof(g_procname));
        
    initSocket();
    scopeLog("Constructor\n", -1);
}

static
void doClose(int fd, char *func)
{
    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        scopeLog(func, fd);
        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(fd);
            scopeLog("decr port\n", fd);
        }

        if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(fd);
            scopeLog("decr connection\n", fd);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }
}

EXPORTON
int close(int fd)
{
    if (g_fn.close == NULL) {
        scopeLog("ERROR: close:NULL\n", fd);
        return -1;
    }

    doClose(fd, "close\n");
    return g_fn.close(fd);
}

#ifdef __MACOS__
EXPORTON
int close$NOCANCEL(int fd)
{
    if (g_fn.close$NOCANCEL == NULL) {
        scopeLog("ERROR: close$NOCANCEL:NULL\n", fd);
        return -1;
    }

    doClose(fd, "close$NOCANCEL\n");
    return g_fn.close$NOCANCEL(fd);
}


EXPORTON
int guarded_close_np(int fd, void *guard)
{
    if (g_fn.guarded_close_np == NULL) {
        scopeLog("ERROR: guarded_close_np:NULL\n", fd);
        return -1;
    }

    doClose(fd, "guarded_close_np\n");
    return g_fn.guarded_close_np(fd, guard);
}

EXPORTOFF
int close_nocancel(int fd)
{
    if (g_fn.close_nocancel == NULL) {
        scopeLog("ERROR: close_nocancel:NULL\n", fd);
        return -1;
    }

    doClose(fd, "close_nocancel\n");
    return g_fn.close_nocancel(fd);
}

#endif // __MACOS__

EXPORTOFF
ssize_t write(int fd, const void *buf, size_t count)
{
    if (g_fn.write == NULL) {
        scopeLog("ERROR: write:NULL\n", fd);
        return -1;
    }

    // Don't init the socket from write; starts early
    // Delay posts until init is complete
    if (g_sock != 0) {
        char metric[strlen(STATSD_WRITE) + 16];
        
        if (snprintf(metric, sizeof(metric), STATSD_WRITE, (int)count) <= 0) {
            scopeLog("ERROR: write: snprintf\n", fd);
        } else {
            postMetric(metric);
        }
    }

    return g_fn.write(fd, buf, count);
}

EXPORTOFF
ssize_t read(int fd, void *buf, size_t count)
{
    char metric[strlen(STATSD_READ) + 16];
    
    if (g_fn.read == NULL) {
        scopeLog("ERROR: read:NULL\n", fd);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_READ, (int)count) <= 0) {
        scopeLog("ERROR: read:snprintf\n", fd);
    } else {
        postMetric(metric);
    }
    
    return g_fn.read(fd, buf, count);
}

EXPORTOFF
void vsyslog(int priority, const char *format, va_list ap)
{
    char metric[strlen(STATSD_VSYSLOG)];
    
    if (g_fn.vsyslog == NULL) {
        scopeLog("ERROR: vsyslog:NULL\n", -1);
        return;
    }

    if (snprintf(metric, sizeof(metric), STATSD_VSYSLOG) <= 0) {
        scopeLog("ERROR: vsyslog:NULL\n", -1);
    } else {
        postMetric(metric);
    }
    
    g_fn.vsyslog(priority, format, ap);
    return;
}

EXPORTON
int socket(int socket_family, int socket_type, int protocol)
{
    int sd;
    
    if (g_fn.socket == NULL) {
        scopeLog("ERROR: socket:NULL\n", -1);
        return -1;
    }

    sd = g_fn.socket(socket_family, socket_type, protocol);
    if (sd != -1) {
        addSock(sd, socket_type);
        
        if (g_netinfo &&
            (g_netinfo[sd].fd == sd) &&
            ((socket_family == AF_INET) ||
             (socket_family == AF_INET6)) &&            
            (socket_type == SOCK_DGRAM)) {
            // Tracking number of open ports
            atomicAdd(&g_openPorts, 1);
            scopeLog("incr\n", g_openPorts);
            
            /*
             * State used in close()
             * We define that a UDP socket represents an open 
             * port when created and is open until the socket is closed
             *
             * a UDP socket is open we say the port is open
             * a UDP socket is closed we say the port is closed
             */
            g_netinfo[sd].listen = TRUE;
            doOpenPorts(sd);
        }
    }

    return sd;
}

EXPORTON
int shutdown(int sockfd, int how)
{
    if (g_fn.shutdown == NULL) {
        scopeLog("ERROR: shutdown:NULL\n", sockfd);
        return -1;
    }

    doClose(sockfd, "shutdown\n");
    return g_fn.shutdown(sockfd, how);
}

EXPORTON
int listen(int sockfd, int backlog)
{
    if (g_fn.listen == NULL) {
        scopeLog("ERROR: listen:NULL\n", -1);
        return -1;
    }

    // Tracking number of open ports
    atomicAdd(&g_openPorts, 1);
    scopeLog("incr\n", g_openPorts);
    scopeLog("listen\n", sockfd);

    if (g_netinfo && (g_netinfo[sockfd].fd == sockfd)) {
        doOpenPorts(sockfd);
    }
    
    return g_fn.listen(sockfd, backlog);
}

static
void doAccept(int sd, struct sockaddr *addr, char *func)
{
    scopeLog(func, sd);
    addSock(sd, SOCK_STREAM);
    g_netinfo[sd].listen = TRUE;
    g_netinfo[sd].accept = TRUE;
    atomicAdd(&g_openPorts, 1);
    scopeLog("incr\n", g_openPorts);

    if (addr->sa_family == AF_INET) {
        // Deal with IPV6 later
        g_netinfo[sd].port = ((struct sockaddr_in *)addr)->sin_port;
    }
    
    doOpenPorts(sd);
}

EXPORTON
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;
    
    if (g_fn.accept == NULL) {
        scopeLog("ERROR: accept:NULL\n", -1);
        return -1;
    }

    sd = g_fn.accept(sockfd, addr, addrlen);
    if (sd != -1) {
        doAccept(sd, addr, "accept\n");
    }

    return sd;
}

EXPORTON
int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags)
{
    int sd;
    
    if (g_fn.accept4 == NULL) {
        scopeLog("ERROR: accept:NULL\n", -1);
        return -1;
    }

    sd = g_fn.accept4(sockfd, addr, addrlen, flags);
    if (sd != -1) {
        doAccept(sd, addr, "accept4\n");
    }

    return sd;
}

EXPORTON
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    if (g_fn.bind == NULL) {
        scopeLog("ERROR: bind:NULL\n", -1);
        return -1;
    }

    if (g_netinfo && (g_netinfo[sockfd].fd == sockfd) &&
        (addr->sa_family == AF_INET)) {
        // Deal with IPV6 later
        g_netinfo[sockfd].port = ((struct sockaddr_in *)addr)->sin_port;
        scopeLog("bind\n", sockfd);
    }
    
    return g_fn.bind(sockfd, addr, addrlen);

}

EXPORTOFF
ssize_t send(int sockfd, const void *buf, size_t len, int flags)
{
    char metric[strlen(STATSD_SEND) + 16];

    if (g_fn.send == NULL) {
        scopeLog("ERROR: send:NULL\n", -1);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_SEND, (int)len) <= 0) {
        scopeLog("ERROR: send:snprintf\n", -1);
    } else {
        postMetric(metric);
    }

    return g_fn.send(sockfd, buf, len, flags);
}

EXPORTOFF
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
    if (g_fn.sendto == NULL) {
        scopeLog("ERROR: sendto:NULL\n", -1);
        return -1;
    }

    if (g_sock != 0) {
        char metric[strlen(STATSD_SENDTO) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_SENDTO, (int)len) <= 0) {
            scopeLog("ERROR: sendto:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }

    return g_fn.sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

EXPORTOFF
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    if (g_fn.sendmsg == NULL) {
        scopeLog("ERROR: sendmsg:NULL\n", -1);
        return -1;
    }

    if (g_sock != 0) {
        char metric[strlen(STATSD_SENDMSG) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_SENDMSG) <= 0) {
            scopeLog("ERROR: sendmsg:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }

    return g_fn.sendmsg(sockfd, msg, flags);
}

EXPORTOFF
ssize_t recv(int sockfd, void *buf, size_t len, int flags)
{
    if (g_fn.recv == NULL) {
        scopeLog("ERROR: recv:NULL\n", -1);
        return -1;
    }

    if (g_sock != 0) {
        char metric[strlen(STATSD_RECV) + 16];

        if (snprintf(metric, sizeof(metric), STATSD_RECV, (int)len) <= 0) {
            scopeLog("ERROR: recv:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }

    return g_fn.recv(sockfd, buf, len, flags);
}

EXPORTOFF
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen)
{
    if (g_fn.recvfrom == NULL) {
        scopeLog("ERROR: recvfrom:NULL\n", -1);
        return -1;
    }

    return g_fn.recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
}

EXPORTOFF
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    if (g_fn.recvmsg == NULL) {
        scopeLog("ERROR: recvmsg:NULL\n", -1);
        return -1;
    }

    return g_fn.recvmsg(sockfd, msg, flags);
}
