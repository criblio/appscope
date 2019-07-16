#include "wrap.h"

interposed_funcs g_fn;

static int g_sock = 0;
static struct sockaddr_in g_saddr;
static operations_info g_ops;
static net_info *g_netinfo;
static int g_numNinfo;
static char g_hostname[MAX_HOSTNAME];
static char g_procname[MAX_PROCNAME];
static int g_openPorts = 0;
static int g_TCPConnections = 0;
static int g_activeConnections = 0;
static int g_netrx = 0;
static int g_nettx = 0;
static int g_dns = 0;

// These need to come from a config file
// Do we like the g_ or the cfg prefix?
#define LOG_FILE 1  // eventually an enum for file, syslog, shared memory 
static bool g_log = TRUE;
static const char g_logFile[] = "/tmp/scope.log";
static unsigned int g_logOp = LOG_FILE;
static int g_logfd = -1;
static bool g_logDataPath = FALSE;
static bool cfgNETRXTXPeriodic = TRUE;
static int cfgPeriod = 10;

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
void dataLog(char *msg, int fd)
{

    if (g_logDataPath == TRUE) {
        scopeLog(msg, fd);
    }
}

// DEBUG
EXPORTOFF
void dumpAddrs(int sd, enum control_type_t endp)
{
    in_port_t port;
    char ip[INET6_ADDRSTRLEN];                                                                                         
    char buf[1024];

    inet_ntop(AF_INET,                                                                                               
              &((struct sockaddr_in *)&g_netinfo[sd].localConn)->sin_addr,
              ip, sizeof(ip));
    port = GET_PORT(sd, g_netinfo[sd].localConn.ss_family, LOCAL);
    snprintf(buf, sizeof(buf), "%s:%d LOCAL: %s:%d\n", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd);

    inet_ntop(AF_INET,                                          
              &((struct sockaddr_in *)&g_netinfo[sd].remoteConn)->sin_addr,
              ip, sizeof(ip));
    port = GET_PORT(sd, g_netinfo[sd].remoteConn.ss_family, REMOTE);
    snprintf(buf, sizeof(buf), "%s:%d REMOTE:%s:%d\n", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd);
    
    if (GET_PORT(sd, g_netinfo[sd].localConn.ss_family, REMOTE) == DNS_PORT) {
        scopeLog("DNS\n", sd);
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
    flags = g_fn.fcntl(g_sock, F_GETFL, 0);
    g_fn.fcntl(g_sock, F_SETFL, flags | O_NONBLOCK);

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

    if (g_fn.sendto) {
        rc = g_fn.sendto(g_sock, metric, strlen(metric), 0, 
                         (struct sockaddr *)&g_saddr, sizeof(g_saddr));
        if (rc < 0) {
            scopeLog("ERROR: postMetric:sendto\n", g_sock);
            switch (errno) {
            case EWOULDBLOCK:
                g_ops.udp_blocks++;
                break;
            default:
                g_ops.udp_errors++;
            }
        }

        //DEBUG: config
        scopeLog((char *)metric, -1);
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
            
            if (g_TCPConnections > 0) {
                atomicSub(&g_TCPConnections, 1);
            }
            
            scopeLog("decr\n", g_openPorts);
            scopeLog("decr conn\n", g_TCPConnections);
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
    }
}

static
int getProtocol(int type, char *proto, size_t len)
{
    if (!proto) {
        return -1;
    }
    
    if (type == SOCK_STREAM) {
        strncpy(proto, "TCP", len);
    } else if (type == SOCK_DGRAM) {
        strncpy(proto, "UDP", len);
    } else if (type == SCOPE_UNIX) {
        // added, not a socket type, want to know if it's a UNIX socket
        strncpy(proto, "UNIX", len);
    } else if (type == SOCK_RAW) {
        strncpy(proto, "RAW", len);
    } else if (type == SOCK_RDM) {
        strncpy(proto, "RDM", len);
    } else if (type == SOCK_SEQPACKET) {
        strncpy(proto, "SEQPACKET", len);
    } else {
        strncpy(proto, "OTHER", len);
    }

    return 0;
}

static
void doProcMetric(enum metric_t type, long measurement)
{
    switch (type) {
    case PROC_CPU:
    {
        char metric[strlen(STATSD_PROCCPU) +
                    sizeof(long) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];
        if (snprintf(metric, sizeof(metric), STATSD_PROCCPU,
                     measurement,
                     g_procname,
                     getpid(),
                     g_hostname) <= 0) {
            scopeLog("ERROR: doProcMetric:CPU:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }
    break;

    case PROC_MEM:
    {
        char metric[strlen(STATSD_PROCMEM) +
                    sizeof(long) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];
        if (snprintf(metric, sizeof(metric), STATSD_PROCMEM,
                     measurement,
                     g_procname,
                     getpid(),
                     g_hostname) <= 0) {
            scopeLog("ERROR: doProcMetric:MEM:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }
    break;

    case PROC_THREAD:
    {
        char metric[strlen(STATSD_PROCTHREAD) +
                    sizeof(int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];
        if (snprintf(metric, sizeof(metric), STATSD_PROCTHREAD,
                     (int)measurement,
                     g_procname,
                     getpid(),
                     g_hostname) <= 0) {
            scopeLog("ERROR: doProcMetric:THREAD:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }
    break;

    case PROC_FD:
    {
        char metric[strlen(STATSD_PROCFD) +
                    sizeof(int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];
        if (snprintf(metric, sizeof(metric), STATSD_PROCFD,
                     (int)measurement,
                     g_procname,
                     getpid(),
                     g_hostname) <= 0) {
            scopeLog("ERROR: doProcMetric:FD:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }
    break;

    case PROC_CHILD:
    {
        char metric[strlen(STATSD_PROCCHILD) +
                    sizeof(int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];
        if (snprintf(metric, sizeof(metric), STATSD_PROCCHILD,
                     (int)measurement,
                     g_procname,
                     getpid(),
                     g_hostname) <= 0) {
            scopeLog("ERROR: doProcMetric:CHILD:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
    }
    break;

    default:
        scopeLog("ERROR: doProcMetric:metric type\n", -1);
    }
}

static
void doNetMetric(enum metric_t type, int fd, enum control_type_t source)
{
    pid_t pid = getpid();
    char proto[PROTOCOL_STR];
    in_port_t localPort, remotePort;
        
    if (g_netinfo == NULL) {
        return;
    }

    if ((source == EVENT_BASED) && (fd > 0)) {
        getProtocol(g_netinfo[fd].type, proto, sizeof(proto));
        localPort = GET_PORT(fd, g_netinfo[fd].localConn.ss_family, LOCAL);
        remotePort = GET_PORT(fd, g_netinfo[fd].remoteConn.ss_family, REMOTE);
    }
    
    switch (type) {
    case OPEN_PORTS:
    {
        char metric[strlen(STATSD_OPENPORTS) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    PROTOCOL_STR  +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) + 1];
            
        if (snprintf(metric, sizeof(metric), STATSD_OPENPORTS,
                     g_openPorts, g_procname, pid, fd, g_hostname, proto,
                     localPort) <= 0) {
            scopeLog("ERROR: doNetMetric:OPENPORTS:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case TCP_CONNECTIONS:
    {
        char metric[strlen(STATSD_TCPCONNS) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    PROTOCOL_STR  +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) + 1];
            
        if (snprintf(metric, sizeof(metric), STATSD_TCPCONNS,
                     g_TCPConnections, g_procname, pid, fd, g_hostname, proto,
                     localPort) <= 0) {
            scopeLog("ERROR: doNetMetric:TCPCONNS:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case ACTIVE_CONNECTIONS:
    {
        char metric[strlen(STATSD_ACTIVECONNS) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    PROTOCOL_STR  +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) + 1];
            
        if (snprintf(metric, sizeof(metric), STATSD_ACTIVECONNS,
                     g_activeConnections, g_procname, pid, fd, g_hostname, proto,
                     localPort) <= 0) {
            scopeLog("ERROR: doNetMetric:ACTIVECONNS:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case NETRX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];
        char metric[strlen(STATSD_NETRX) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    PROTOCOL_STR  +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) + 1];

        if ((cfgNETRXTXPeriodic == TRUE) && (source == EVENT_BASED)) {
            break;
        }

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if (g_netinfo[fd].type == SCOPE_UNIX) {
            strncpy(lip, " ", sizeof(lip));
            strncpy(rip, " ", sizeof(rip));
        } else {
            if (g_netinfo[fd].localConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&g_netinfo[fd].localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }
            } else if (g_netinfo[fd].localConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&g_netinfo[fd].localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }

            } else {
                strncpy(lip, " ", sizeof(lip));
            }

            if (g_netinfo[fd].remoteConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&g_netinfo[fd].remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else if (g_netinfo[fd].remoteConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&g_netinfo[fd].remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else {
                strncpy(rip, " ", sizeof(rip));
            }
        }
        
        if (snprintf(metric, sizeof(metric), STATSD_NETRX,
                     g_netrx, g_procname, pid,
                     fd, g_hostname, proto,
                     lip, localPort,
                     rip, remotePort, data) <= 0) {
            scopeLog("ERROR: doNetMetric:NETRX:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case NETTX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];
        char metric[strlen(STATSD_NETTX) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    PROTOCOL_STR  +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) +
                    sizeof(unsigned int) + 1];

        if ((cfgNETRXTXPeriodic == TRUE) && (source == EVENT_BASED)) {
            break;
        }

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if (g_netinfo[fd].type == SCOPE_UNIX) {
            strncpy(lip, " ", sizeof(lip));
            strncpy(rip, " ", sizeof(rip));
        } else {
            if (g_netinfo[fd].localConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&g_netinfo[fd].localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }
            } else if (g_netinfo[fd].localConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&g_netinfo[fd].localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }

            } else {
                strncpy(lip, " ", sizeof(lip));
            }

            if (g_netinfo[fd].remoteConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&g_netinfo[fd].remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else if (g_netinfo[fd].remoteConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&g_netinfo[fd].remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else {
                strncpy(rip, " ", sizeof(rip));
            }
        }
        
        if (snprintf(metric, sizeof(metric), STATSD_NETTX,
                     g_nettx, g_procname, pid,
                     fd, g_hostname, proto,
                     lip, localPort,
                     rip, remotePort, data) <= 0) {
            scopeLog("ERROR: doNetMetric:NETTX:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case NETRX_PROC:
    {
        char metric[strlen(STATSD_NETRX_PROC) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];

        if (snprintf(metric, sizeof(metric), STATSD_NETRX_PROC,
                     g_netrx, g_procname, pid,
                     g_hostname) <= 0) {
            scopeLog("ERROR: doNetMetric:NETRX_PROC:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case NETTX_PROC:
    {
        char metric[strlen(STATSD_NETTX_PROC) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) + 1];

        if (snprintf(metric, sizeof(metric), STATSD_NETTX_PROC,
                     g_nettx, g_procname, pid,
                     g_hostname) <= 0) {
            scopeLog("ERROR: doNetMetric:NETTX_PROC:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    case DNS:
    {
        char metric[strlen(STATSD_DNS) +
                    sizeof(unsigned int) +
                    strlen(g_hostname) +
                    strlen(g_procname) +
                    sizeof(unsigned int) +
                    strlen(g_netinfo[fd].dnsName) + 1];

        if (snprintf(metric, sizeof(metric), STATSD_DNS,
                     g_dns, g_procname, pid,
                     g_hostname,
                     g_netinfo[fd].dnsName) <= 0) {
            scopeLog("ERROR: doNetMetric:DNS:snprintf\n", -1);
        } else {
            postMetric(metric);
        }
        break;
    }

    default:
        scopeLog("ERROR: doNetMetric:metric type\n", -1);
    }
}

// Return process specific CPU usage in microseconds
static
long doGetProcCPU(pid_t pid) {
    struct rusage ruse;
    
    if (getrusage(RUSAGE_SELF, &ruse) != 0) {
        return (long)-1;
    }

    return (long)((ruse.ru_utime.tv_sec * (1024 * 1024)) + ruse.ru_utime.tv_usec) +
        ((ruse.ru_stime.tv_sec * (1024 * 1024)) + ruse.ru_stime.tv_usec);
}

// Return process specific memory usage in kilobytes
static
long doGetProcMem(pid_t pid) {
    struct rusage ruse;
    
    if (getrusage(RUSAGE_SELF, &ruse) != 0) {
        return (long)-1;
    }

    // macOS returns bytes, Linux returns kilobytes
#ifdef __MACOS__    
    return ruse.ru_maxrss / 1024;
#else
    return ruse.ru_maxrss;
#endif // __MACOS__        
}

static
void doSetConnection(int sd, const struct sockaddr *addr, socklen_t len, enum control_type_t endp)
{
    // Should we check for at least the size of sockaddr_in?
    if (g_netinfo && (g_netinfo[sd].fd == sd) &&
        addr && (len > 0)) {
        if (endp == LOCAL) {
            memcpy(&g_netinfo[sd].localConn, addr, len);
        } else {
            memcpy(&g_netinfo[sd].remoteConn, addr, len);
        }
    }
}

static
int doSetAddrs(int sockfd)
{
    if (g_netinfo[sockfd].localConn.ss_family == AF_UNSPEC) {
        struct sockaddr_storage addr;
        socklen_t addrlen = sizeof(struct sockaddr_storage);
        
        if (getsockname(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
            doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, LOCAL);
        }
    }
    
    if (g_netinfo[sockfd].remoteConn.ss_family == AF_UNSPEC) {
        struct sockaddr_storage addr;
        socklen_t addrlen = sizeof(struct sockaddr_storage);
        
        if (getpeername(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
            doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, LOCAL);
        }
    }

    return 0;
}

/*
 * We missed an accept
 * A socket was dup'd
 * We are a child proc that inherited a socket
 */
static
int doAddNewSock(int sockfd)
{
    struct sockaddr addr;
    socklen_t addrlen = sizeof(struct sockaddr);
        
    scopeLog("doAddNewSock: adding socket\n", sockfd);
    if (getsockname(sockfd, &addr, &addrlen) != -1) {
        if ((addr.sa_family == AF_INET) || (addr.sa_family == AF_INET6)) {
            int type;
            socklen_t len = sizeof(socklen_t);
                
            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type);
            } else {
                // Really can't add the socket at this point
                scopeLog("ERROR: doRecv:getsockopt\n", sockfd);
            }
        } else if (addr.sa_family == AF_UNIX) {
            // added, not a socket type, want to know if it's a UNIX socket
            addSock(sockfd, SCOPE_UNIX);
        } else {
            // is RAW a viable default?
            addSock(sockfd, SOCK_RAW);
        }
        doSetConnection(sockfd, &addr, addrlen, LOCAL);
    } else {
        addSock(sockfd, SOCK_RAW);
    }
    
    addrlen = sizeof(struct sockaddr);
    if (getpeername(sockfd, &addr, &addrlen) != -1) {
        doSetConnection(sockfd, &addr, addrlen, REMOTE);
    }

    return 0;
}

/*
 * Dereference a DNS packet and
 * extract the domain name.
 *
 * Example:
 * This converts "\003www\006google\003com"
 * in DNS format to www.google.com
 */

static
int getDNSName(int sd, void *pkt, int pktlen)
{
    dns_query *query;
    char *aname, *dname;

    query = (struct dns_query_t *)pkt;
    if ((dname = (char *)&query->name) == NULL) {
        return -1;
    }

    // TODO: add size check!
    aname = g_netinfo[sd].dnsName;
    dname++; // ignore the ETX
    while (*dname != '\0') {
        if (isprint(*dname) != 0) {
            *aname++ = *dname;
        } else {
            *aname++ = '.';
        }
        dname++;
    }
    *aname = '\0';

    return 0;
}


static
int doRecv(int sockfd, ssize_t rc)
{
    atomicAdd(&g_netrx, rc);
    if (g_netinfo && (g_netinfo[sockfd].fd != sockfd)) {
        doAddNewSock(sockfd);
    }

    doSetAddrs(sockfd);
    doNetMetric(NETRX, sockfd, EVENT_BASED);

    return 0;
}

static
int doSend(int sockfd, ssize_t rc)
{
    atomicAdd(&g_nettx, rc);
    if (g_netinfo && (g_netinfo[sockfd].fd != sockfd)) {
        doAddNewSock(sockfd);
    }

    doSetAddrs(sockfd);
    doNetMetric(NETTX, sockfd, EVENT_BASED);

    if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
        atomicAdd(&g_dns, 1);
        doNetMetric(DNS, sockfd, EVENT_BASED);
    }

    return 0;
}

static
void doAccept(int sd, struct sockaddr *addr, socklen_t addrlen, char *func)
{

    scopeLog(func, sd);
    addSock(sd, SOCK_STREAM);
    
    if (g_netinfo && (g_netinfo[sd].fd == sd)) {
        g_netinfo[sd].listen = TRUE;
        g_netinfo[sd].accept = TRUE;
        atomicAdd(&g_openPorts, 1);
        atomicAdd(&g_TCPConnections, 1);
        atomicAdd(&g_activeConnections, 1);
        scopeLog("incr conn\n", g_TCPConnections);
        scopeLog("incr\n", g_openPorts);
        doSetConnection(sd, addr, addrlen, REMOTE);
        doNetMetric(OPEN_PORTS, sd, EVENT_BASED);
        doNetMetric(TCP_CONNECTIONS, sd, EVENT_BASED);
        doNetMetric(ACTIVE_CONNECTIONS, sd, EVENT_BASED);
    }
}

static
void * periodic(void *arg)
{
    long cpu, mem;
    int nthread, nfds, children;
    pid_t pid = getpid();

    while (1) {
        cpu = doGetProcCPU(pid);
        doProcMetric(PROC_CPU, cpu);
        
        mem = doGetProcMem(pid);
        doProcMetric(PROC_MEM, mem);

        nthread = osGetNumThreads(pid);
        doProcMetric(PROC_THREAD, nthread);

        nfds = osGetNumFds(pid);
        doProcMetric(PROC_FD, nfds);

        children = osGetNumChildProcs(pid);
        doProcMetric(PROC_CHILD, children);

        doNetMetric(NETRX_PROC, -1, PERIODIC);
        doNetMetric(NETTX_PROC, -1, PERIODIC);
        
        // Needs to be defined in a config file
        sleep(cfgPeriod);
    }

    return NULL;
}

__attribute__((constructor)) void init(void)
{
    pthread_t periodicTID;
    
    g_fn.vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.fcntl = dlsym(RTLD_NEXT, "fcntl");
    g_fn.socket = dlsym(RTLD_NEXT, "socket");
    g_fn.shutdown = dlsym(RTLD_NEXT, "shutdown");
    g_fn.listen = dlsym(RTLD_NEXT, "listen");
    g_fn.accept = dlsym(RTLD_NEXT, "accept");
    g_fn.accept4 = dlsym(RTLD_NEXT, "accept4");
    g_fn.bind = dlsym(RTLD_NEXT, "bind");
    g_fn.connect = dlsym(RTLD_NEXT, "connect");    
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
    g_fn.accept$NOCANCEL = dlsym(RTLD_NEXT, "accept$NOCANCEL");
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

    if (pthread_create(&periodicTID, NULL, periodic, NULL) != 0) {
        scopeLog("ERROR: Constructor:pthread_create\n", -1);
    }

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
            doNetMetric(OPEN_PORTS, fd, EVENT_BASED);
            scopeLog("decr port\n", fd);
        }

        if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_TCPConnections, 1);
            scopeLog("decr conn\n", g_TCPConnections);
            doNetMetric(TCP_CONNECTIONS, fd, EVENT_BASED);
            scopeLog("decr connection\n", fd);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }
}

EXPORTON
int close(int fd)
{
    int rc;
    
    if (g_fn.close == NULL) {
        scopeLog("ERROR: close:NULL\n", fd);
        return -1;
    }

    rc = g_fn.close(fd);
    if (rc != -1) {
        doClose(fd, "close\n");
    }
    
    return rc;
}

#ifdef __MACOS__
EXPORTON
int close$NOCANCEL(int fd)
{
    int rc;
    
    if (g_fn.close$NOCANCEL == NULL) {
        scopeLog("ERROR: close$NOCANCEL:NULL\n", fd);
        return -1;
    }

    rc = g_fn.close$NOCANCEL(fd);
    if (rc != -1) {
        doClose(fd, "close$NOCANCEL\n");
    }
    
    return rc;
}


EXPORTON
int guarded_close_np(int fd, void *guard)
{
    int rc;
    
    if (g_fn.guarded_close_np == NULL) {
        scopeLog("ERROR: guarded_close_np:NULL\n", fd);
        return -1;
    }

    rc = g_fn.guarded_close_np(fd, guard);
    if (rc != -1) {
        doClose(fd, "guarded_close_np\n");
    }
    
    return rc;
}

EXPORTOFF
int close_nocancel(int fd)
{
    int rc;
    
    if (g_fn.close_nocancel == NULL) {
        scopeLog("ERROR: close_nocancel:NULL\n", fd);
        return -1;
    }

    rc = g_fn.close_nocancel(fd);
    if (rc != -1) {
        doClose(fd, "close_nocancel\n");
    }
    
    return rc;
}

EXPORTON
int accept$NOCANCEL(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;
    
    if (g_fn.accept$NOCANCEL == NULL) {
        scopeLog("ERROR: accept$NOCANCEL:NULL\n", -1);
        return -1;
    }

    sd = g_fn.accept$NOCANCEL(sockfd, addr, addrlen);
    if (sd != -1) {
        doAccept(sd, addr, *addrlen, "accept$NOCANCEL\n");
    }

    return sd;
}

#endif // __MACOS__

EXPORTON
ssize_t write(int fd, const void *buf, size_t count)
{
    ssize_t rc;
    
    if (g_fn.write == NULL) {
        scopeLog("ERROR: write:NULL\n", fd);
        return -1;
    }

    rc = g_fn.write(fd, buf, count);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd)) {
        // This is a network descriptor
        dataLog("write\n", fd);
        doSetAddrs(fd);
        doSend(fd, rc);
    }
    
    return rc;
}

EXPORTON
ssize_t read(int fd, void *buf, size_t count)
{
    ssize_t rc;
    
    if (g_fn.read == NULL) {
        scopeLog("ERROR: read:NULL\n", fd);
        return -1;
    }

    rc = g_fn.read(fd, buf, count);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd)) {
        // This is a network descriptor
        dataLog("read\n", fd);
        doSetAddrs(fd);
        doRecv(fd, rc);
    }
    
    return rc;
}

EXPORTON
int fcntl(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;

    if (g_fn.fcntl == NULL ) {
        scopeLog("ERROR: rcntl:NULL\n", fd);
        return -1;
    }

    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                    fArgs.arg[2], fArgs.arg[3]);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd) &&
        (cmd == F_DUPFD)) {
        // This is a network descriptor
        scopeLog("fcntl\n", rc);
        doAddNewSock(rc);
    }
    
    return rc;
}

EXPORTOFF
void vsyslog(int priority, const char *format, va_list ap)
{
    if (g_fn.vsyslog == NULL) {
        scopeLog("ERROR: vsyslog:NULL\n", -1);
        return;
    }

    scopeLog("vsyslog\n", -1);
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
        scopeLog("socket\n", sd);
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
            doNetMetric(OPEN_PORTS, sd, EVENT_BASED);
        }
    }

    return sd;
}

EXPORTON
int shutdown(int sockfd, int how)
{
    int rc;
    
    if (g_fn.shutdown == NULL) {
        scopeLog("ERROR: shutdown:NULL\n", sockfd);
        return -1;
    }

    rc = g_fn.shutdown(sockfd, how);
    if (rc != -1) {
        doClose(sockfd, "shutdown\n");
    }
    
    return rc;
}

EXPORTON
int listen(int sockfd, int backlog)
{
    int rc;
    
    if (g_fn.listen == NULL) {
        scopeLog("ERROR: listen:NULL\n", -1);
        return -1;
    }

    rc = g_fn.listen(sockfd, backlog);
    if (rc != -1) {
        scopeLog("listen\n", sockfd);
        
        // Tracking number of open ports
        atomicAdd(&g_openPorts, 1);
        scopeLog("incr\n", g_openPorts);
        
        if (g_netinfo && (g_netinfo[sockfd].fd == sockfd)) {
            g_netinfo[sockfd].listen = TRUE;
            g_netinfo[sockfd].accept = TRUE;
            doNetMetric(OPEN_PORTS, sockfd, EVENT_BASED);

            if (g_netinfo[sockfd].type == SOCK_STREAM) {
                atomicAdd(&g_TCPConnections, 1);
                g_netinfo[sockfd].accept = TRUE;                            
                scopeLog("incr conn\n", g_TCPConnections);
                doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED);
            }
        }
    }
    
    return rc;
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
        doAccept(sd, addr, *addrlen, "accept\n");
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
        doAccept(sd, addr, *addrlen, "accept4\n");
    }

    return sd;
}

EXPORTON
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;
    
    if (g_fn.bind == NULL) {
        scopeLog("ERROR: bind:NULL\n", -1);
        return -1;
    }

    rc = g_fn.bind(sockfd, addr, addrlen);
    if (rc != -1) { 
        doSetConnection(sockfd, addr, addrlen, LOCAL);
        scopeLog("bind\n", sockfd);
    }
    
    return rc;

}

EXPORTON
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)

{
    int rc;
    
    if (g_fn.connect == NULL) {
        scopeLog("ERROR: connect:NULL\n", -1);
        return -1;
    }

    rc = g_fn.connect(sockfd, addr, addrlen);
    if ((rc != -1) &&
        (g_netinfo) &&
        (g_netinfo[sockfd].fd == sockfd)) {
        doSetConnection(sockfd, addr, addrlen, REMOTE);
        g_netinfo[sockfd].accept = TRUE;
        atomicAdd(&g_activeConnections, 1);
        doNetMetric(ACTIVE_CONNECTIONS, sockfd, EVENT_BASED);

        if (g_netinfo[sockfd].type == SOCK_STREAM) {
            atomicAdd(&g_TCPConnections, 1);
            scopeLog("incr conn\n", g_TCPConnections);
            doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED);
        }
        scopeLog("connect\n", sockfd);
    }
    
    return rc;

}

EXPORTON
ssize_t send(int sockfd, const void *buf, size_t len, int flags)
{
    ssize_t rc;

    if (g_fn.send == NULL) {
        scopeLog("ERROR: send:NULL\n", -1);
        return -1;
    }

    rc = g_fn.send(sockfd, buf, len, flags);
    if (rc != -1) {
        dataLog("send\n", sockfd);
        doSend(sockfd, rc);
    }
    
    return rc;
}

EXPORTON
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;
    
    if (g_fn.sendto == NULL) {
        scopeLog("ERROR: sendto:NULL\n", -1);
        return -1;
    }

    rc = g_fn.sendto(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        dataLog("sendto\n", sockfd);
        doSetConnection(sockfd, dest_addr, addrlen, REMOTE);
        //dumpAddrs(sockfd, REMOTE);

        if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc);
    }

    return rc;
}

EXPORTON
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    ssize_t rc;
    
    if (g_fn.sendmsg == NULL) {
        scopeLog("ERROR: sendmsg:NULL\n", -1);
        return -1;
    }

    rc = g_fn.sendmsg(sockfd, msg, flags);
    if (rc != -1) {
        dataLog("sendmsg\n", sockfd);

        // For UDP connections the msg is a remote addr
        if (g_netinfo && msg && (g_netinfo[sockfd].type != SOCK_STREAM)) {
            if (msg->msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msg->msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, msg->msg_iov->iov_base, msg->msg_iov->iov_len);
        }

        doSend(sockfd, rc);
    }
    
    return rc;
}

EXPORTON
ssize_t recv(int sockfd, void *buf, size_t len, int flags)
{
    ssize_t rc;
    
    if (g_fn.recv == NULL) {
        scopeLog("ERROR: recv:NULL\n", -1);
        return -1;
    }

    dataLog("recv\n", sockfd);
    rc = g_fn.recv(sockfd, buf, len, flags);
    if (rc != -1) {
        doRecv(sockfd, rc);
    }
    
    return rc;
}

EXPORTON
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen)
{
    ssize_t rc;
    
    if (g_fn.recvfrom == NULL) {
        scopeLog("ERROR: recvfrom:NULL\n", -1);
        return -1;
    }

    dataLog("recvfrom\n", sockfd);
    rc = g_fn.recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    if (rc != -1) {
        atomicAdd(&g_netrx, rc);
        if (g_netinfo && (g_netinfo[sockfd].fd != sockfd)) {
            // We missed an accept...most likely
            // Or.. we are a child proc that inherited a socket
            int type;
            socklen_t len = sizeof(socklen_t);
                
            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type);
            } else {
                // Really can't add the socket at this point
                scopeLog("ERROR: recvfrom:getsockopt\n", sockfd);
            }
        }

        doSetConnection(sockfd, src_addr, *addrlen, REMOTE);
        doNetMetric(NETRX, sockfd, EVENT_BASED);
    }
    return rc;
}

EXPORTON
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    ssize_t rc;
    
    if (g_fn.recvmsg == NULL) {
        scopeLog("ERROR: recvmsg:NULL\n", -1);
        return -1;
    }

    rc = g_fn.recvmsg(sockfd, msg, flags);
    if (rc != -1) {
        dataLog("recvmsg\n", sockfd);

        // For UDP connections the msg is a remote addr
        if (msg && (g_netinfo[sockfd].type != SOCK_STREAM)) {
            if (msg->msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msg->msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }
        
        doRecv(sockfd, rc);
    }
    
    return rc;
}
