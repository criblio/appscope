#include "wrap.h"

static void scopeLog(char *msg, int fd)
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
        g_write(g_logfd, buf, strlen(buf));
    }        
}

static void initSocket(void)
{
    int flags;
    char server[sizeof(SERVER) + 1];

    // Create a UDP socket
    g_sock = g_socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_sock < 0)	{
        g_ops.init_errors++;
        strncpy((char *)g_ops.errMsg, "initSock:socket", sizeof(g_ops.errMsg));
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
        g_ops.init_errors++;
        strncpy((char *)g_ops.errMsg, "initSock:inet_aton", sizeof(g_ops.errMsg));
    }
}

static void postMetric(const char *metric)
{
    ssize_t rc;

    if (g_socket == 0) {
        initSocket();
    }

    if (g_sendto) {
        rc = g_sendto(g_sock, metric, strlen(metric), 0, 
                      (struct sockaddr *)&g_saddr, sizeof(g_saddr));
        if (rc < 0) {
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

static void addSock(int fd, int type) {
    if (g_netinfo) {
        if (g_netinfo[fd].fd == fd) {
            scopeLog("addSock: duplicate\n", fd);
            atomicSub(&g_openPorts, 1);
            return;
        }
        
        if ((fd > g_numNinfo) && (fd < MAX_FDS))  {
            // Need to realloc
            if ((g_netinfo = realloc(g_netinfo, sizeof(struct net_info_t) * fd)) == NULL) {
                g_ops.interpose_errors++;
                strncpy((char *)g_ops.errMsg, "addSock:realloc failed", sizeof(g_ops.errMsg));
            }
            g_numNinfo = fd;
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].fd = fd;
        g_netinfo[fd].type = type;
        scopeLog("addSocket\n", fd);
    }
}

static int getProtocol(int type, char *proto, size_t len) {
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

static void doOpenPorts(int fd) {
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
        g_ops.interpose_errors++;
        strncpy((char *)g_ops.errMsg, "close:snprintf", sizeof(g_ops.errMsg));
        
    }
    postMetric(metric);
}

__attribute__((constructor)) void init(void)
{
    if (DEBUG == 1) write(STDOUT_FILENO, "constructor\n", 12);
    g_vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    g_send = dlsym(RTLD_NEXT, "send");
    g_sendto = dlsym(RTLD_NEXT, "sendto");
    g_sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    g_socket = dlsym(RTLD_NEXT, "socket");
    g_read = dlsym(RTLD_NEXT, "read");
    g_write = dlsym(RTLD_NEXT, "write");
    g_recv = dlsym(RTLD_NEXT, "recv");
    g_recvfrom = dlsym(RTLD_NEXT, "recvfrom");
    g_recvmsg = dlsym(RTLD_NEXT, "recvmsg");
    g_listen = dlsym(RTLD_NEXT, "listen");
    g_close = dlsym(RTLD_NEXT, "close");
    g_close$NOCANCEL = dlsym(RTLD_NEXT, "close$NOCANCEL");
    g_close_nocancel = dlsym(RTLD_NEXT, "close_nocancel");
    g_shutdown = dlsym(RTLD_NEXT, "shutdown");
    g_bind = dlsym(RTLD_NEXT, "bind");
    g_guarded_close_np = dlsym(RTLD_NEXT, "guarded_close_np");

    if ((g_netinfo = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR:Constructor:Malloc\n", -1);
    }

    g_numNinfo = NET_ENTRIES;
    if (gethostname(g_hostname, sizeof(g_hostname)) != 0) {
        scopeLog("ERROR:Constructor:gethostname\n", -1);
    }

    osGetProcname(g_procname, sizeof(g_procname));
        
    initSocket();
}

EXPORTON
int close(int fd) {
    if (g_close == NULL) {
        scopeLog("ERROR: close:NULL\n", fd);
        return -1;
    }

    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(fd);
            scopeLog("close:sub port\n", fd);
        } else if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(fd);
        }
        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }
    
    return g_close(fd);
}

#ifdef __MACOS__
EXPORTON
int close$NOCANCEL(int fd) {
    if (g_close$NOCANCEL == NULL) {
        scopeLog("ERROR: close$NOCANCEL:NULL\n", fd);
        return -1;
    }
    
    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        scopeLog("close$NOCANCEL\n", fd);

        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(fd);
            scopeLog("close$NOCANCEL:port\n", fd);
        } else if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(fd);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }

    return g_close$NOCANCEL(fd);
}


EXPORTON
int guarded_close_np(int fd, void *guard) {
    if (g_guarded_close_np == NULL) {
        scopeLog("ERROR: guarded_close_np:NULL\n", fd);
        return -1;
    }

    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        scopeLog("guarded_close_np\n", fd);

        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(fd);
            scopeLog("guarded_close_np:port\n", fd);
        } else if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(fd);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }

    return g_guarded_close_np(fd, guard);
}

EXPORTOFF
int close_nocancel(int fd) {
    if (g_close_nocancel == NULL) {
        scopeLog("ERROR: close_nocancel:NULL\n", fd);
        return -1;
    }

    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        scopeLog("close_nocancel\n", fd);
        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(fd);
            scopeLog("close:sub port\n", fd);
        } else if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(fd);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }

    return g_close_nocancel(fd);
}

#endif // __MACOS__

EXPORTON
int shutdown(int sockfd, int how) {
    if (g_shutdown == NULL) {
        scopeLog("ERROR: shutdown:NULL\n", sockfd);
        return -1;
    }

    scopeLog("shutdown\n", sockfd);
    if (g_netinfo && (g_netinfo[sockfd].fd == sockfd)) {
        if (g_netinfo[sockfd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doOpenPorts(sockfd);
            scopeLog("shutdown:sub port\n", sockfd);
        } else if (g_netinfo[sockfd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_activeConnections, 1);
            //doActiveConns(sockfd);
        }

        memset(&g_netinfo[sockfd], 0, sizeof(struct net_info_t));
    }
    
    return g_shutdown(sockfd, how);
}

EXPORTOFF
ssize_t write(int fd, const void *buf, size_t count)
{
    if (g_write == NULL) {
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

    return g_write(fd, buf, count);
}

EXPORTOFF
ssize_t read(int fd, void *buf, size_t count)
{
    char metric[strlen(STATSD_READ) + 16];
    
    if (g_read == NULL) {
        scopeLog("ERROR: read:NULL\n", fd);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_READ, (int)count) <= 0) {
        scopeLog("ERROR: read:snprintf\n", fd);
    } else {
        postMetric(metric);
    }
    
    return g_read(fd, buf, count);
}

EXPORTOFF
void vsyslog(int priority, const char *format, va_list ap)
{
    char metric[strlen(STATSD_VSYSLOG)];
    
    if (g_vsyslog == NULL) {
        scopeLog("ERROR: vsyslog:NULL\n", -1);
        return;
    }

    if (snprintf(metric, sizeof(metric), STATSD_VSYSLOG) <= 0) {
        scopeLog("ERROR: vsyslog:NULL\n", -1);
    } else {
        postMetric(metric);
    }
    
    g_vsyslog(priority, format, ap);
    return;
}

EXPORTON
int socket(int socket_family, int socket_type, int protocol)
{
    int sd;
    
    if (g_socket == NULL) {
        scopeLog("ERROR: socket:NULL\n", -1);
        return -1;
    }

    sd = g_socket(socket_family, socket_type, protocol);
    if (sd != -1) {
        addSock(sd, socket_type);
        
        if (g_netinfo &&
            (g_netinfo[sd].fd == sd) &&
            ((socket_family == AF_INET) ||
             (socket_family == AF_INET6)) &&            
            (socket_type == SOCK_DGRAM)) {
            // Tracking number of open ports
            atomicAdd(&g_openPorts, 1);
            
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
int listen(int sockfd, int backlog)
{
    if (g_listen == NULL) {
        scopeLog("ERROR: listen:NULL\n", -1);
        return -1;
    }

    // Tracking number of open ports
    atomicAdd(&g_openPorts, 1);

    if (g_netinfo && (g_netinfo[sockfd].fd == sockfd)) {
        doOpenPorts(sockfd);
    }
    
    return g_listen(sockfd, backlog);
}

EXPORTON
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    if (g_bind == NULL) {
        scopeLog("ERROR: bind:NULL\n", -1);
        return -1;
    }

    if (g_netinfo && (g_netinfo[sockfd].fd == sockfd) &&
        (addr->sa_family == AF_INET)) {
        // Deal with IPV6 later
        g_netinfo[sockfd].port = ((struct sockaddr_in *)addr)->sin_port;
        doOpenPorts(sockfd);
    }
    
    return g_bind(sockfd, addr, addrlen);

}

EXPORTOFF
ssize_t send(int sockfd, const void *buf, size_t len, int flags)
{
    char metric[strlen(STATSD_SEND) + 16];

    if (g_send == NULL) {
        scopeLog("ERROR: send:NULL\n", -1);
        return -1;
    }

    if (snprintf(metric, sizeof(metric), STATSD_SEND, (int)len) <= 0) {
        scopeLog("ERROR: send:snprintf\n", -1);
    } else {
        postMetric(metric);
    }

    return g_send(sockfd, buf, len, flags);
}

EXPORTOFF
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
    if (g_sendto == NULL) {
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

    return g_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

EXPORTOFF
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    if (g_sendmsg == NULL) {
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

    return g_sendmsg(sockfd, msg, flags);
}

EXPORTOFF
ssize_t recv(int sockfd, void *buf, size_t len, int flags)
{
    if (g_recv == NULL) {
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

    return g_recv(sockfd, buf, len, flags);
}

EXPORTOFF
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen)
{
    if (g_recvfrom == NULL) {
        scopeLog("ERROR: recvfrom:NULL\n", -1);
        return -1;
    }

    return g_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
}

EXPORTOFF
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    if (g_recvmsg == NULL) {
        scopeLog("ERROR: recvmsg:NULL\n", -1);
        return -1;
    }

    return g_recvmsg(sockfd, msg, flags);
}
