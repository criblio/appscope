#include "cfg.h"
#include "cfgutils.h"
#include "log.h"
#include "out.h"
#include "wrap.h"

interposed_funcs g_fn;

static net_info *g_netinfo;
static int g_numNinfo;
static char g_hostname[MAX_HOSTNAME];
static char g_procname[MAX_PROCNAME];

/*
 * We use static globals here as we
 * track the number of instances
 * and report the values as they occur.
 */
static int g_openPorts = 0;
static int g_TCPConnections = 0;
static int g_activeConnections = 0;

/*
 * We use static globals for this as we
 * accumulate the byte count and upload
 * in a periodic thread.
 */
static int g_netrx = 0;
static int g_nettx = 0;

typedef struct {
    unsigned interval;                   // in seconds
    time_t startTime; 
    bool once;
    pthread_t periodicTID;
} thread_timing_t;
static thread_timing_t g_thread = {0};

// These need to come from a config file
// Do we like the g_ or the cfg prefix?
static bool cfgNETRXTXPeriodic = TRUE;

static log_t* g_log = NULL;
static out_t* g_out = NULL;

// Forward declaration
static void * periodic(void *);
    
EXPORTON void
scopeLog(char* msg, int fd, cfg_log_level_t level)
{
    if (!g_log || !msg) return;

    char buf[strlen(msg) + 128];
    snprintf(buf, sizeof(buf), "Scope: %s(%d): %s", g_procname, fd, msg);
    logSend(g_log, buf, level);
}

// DEBUG
EXPORTOFF void
dumpAddrs(int sd, enum control_type_t endp)
{
    in_port_t port;
    char ip[INET6_ADDRSTRLEN];                                                                                         
    char buf[1024];

    inet_ntop(AF_INET,                                                                                               
              &((struct sockaddr_in *)&g_netinfo[sd].localConn)->sin_addr,
              ip, sizeof(ip));
    port = GET_PORT(sd, g_netinfo[sd].localConn.ss_family, LOCAL);
    snprintf(buf, sizeof(buf), "%s:%d LOCAL: %s:%d\n", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);

    inet_ntop(AF_INET,                                          
              &((struct sockaddr_in *)&g_netinfo[sd].remoteConn)->sin_addr,
              ip, sizeof(ip));
    port = GET_PORT(sd, g_netinfo[sd].remoteConn.ss_family, REMOTE);
    snprintf(buf, sizeof(buf), "%s:%d REMOTE:%s:%d\n", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);
    
    if (GET_PORT(sd, g_netinfo[sd].localConn.ss_family, REMOTE) == DNS_PORT) {
        scopeLog("DNS\n", sd, CFG_LOG_DEBUG);
    }
}

static void
addSock(int fd, int type)
{
    if (g_netinfo) {
        if (g_netinfo[fd].fd == fd) {
            scopeLog("addSock: duplicate\n", fd, CFG_LOG_DEBUG);

            if (g_openPorts > 0) {
                atomicSub(&g_openPorts, 1);
            }
            
            if (g_TCPConnections > 0) {
                atomicSub(&g_TCPConnections, 1);
            }
          
            return;
        }
        
        if ((fd > g_numNinfo) && (fd < MAX_FDS))  {
            // Need to realloc
            if ((g_netinfo = realloc(g_netinfo, sizeof(struct net_info_t) * fd)) == NULL) {
                scopeLog("ERROR: addSock:realloc\n", fd, CFG_LOG_ERROR);
            }
            g_numNinfo = fd;
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].fd = fd;
        g_netinfo[fd].type = type;
    }
}

static int
getProtocol(int type, char *proto, size_t len)
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

static void
doReset()
{
    g_openPorts = g_TCPConnections = g_activeConnections = g_netrx = g_nettx = g_thread.once = 0;
    g_thread.startTime = time(NULL) + g_thread.interval;
}

static void
doThread()
{
    /*
     * If we try to start the perioidic thread before the constructor
     * is executed and our config is not set, we are able to start the
     * thread too early. Some apps, most notably Chrome, check to 
     * ensure that no extra threads are created before it is fully 
     * initialized. This check is intended to ensure that we don't 
     * start the thread until after we have our config. 
     */
    if (!g_out) return;
    
    // Create one thread at most
    if (g_thread.once == TRUE) return;

    /*
     * g_thread.startTime is the start time, set in the constructor.
     * This is put in place to work around one of the Chrome sandbox limits.
     * Shouldn't hurt anything else.  
     */
    if (time(NULL) >= g_thread.startTime) {
        g_thread.once = TRUE;
        if (pthread_create(&g_thread.periodicTID, NULL, periodic, NULL) != 0) {
            scopeLog("ERROR: doThread:pthread_create\n", -1, CFG_LOG_ERROR);
        }
    }
}

static void
doProcMetric(enum metric_t type, long long measurement)
{
    pid_t pid = getpid();
    switch (type) {
    case PROC_CPU:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "microsecond",         1),
            FIELDEND
        };
        event_t e = {"proc.cpu", measurement, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doProcMetric:CPU:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case PROC_MEM:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "kibibyte",            1),
            FIELDEND
        };
        event_t e = {"proc.mem", measurement, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doProcMetric:MEM:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case PROC_THREAD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "thread",              1),
            FIELDEND
        };
        event_t e = {"proc.thread", measurement, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doProcMetric:THREAD:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case PROC_FD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "file",                1),
            FIELDEND
        };
        event_t e = {"proc.fd", measurement, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doProcMetric:FD:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case PROC_CHILD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "process",             1),
            FIELDEND
        };
        event_t e = {"proc.child", measurement, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doProcMetric:CHILD:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    default:
        scopeLog("ERROR: doProcMetric:metric type\n", -1, CFG_LOG_ERROR);
    }
}

static void
doNetMetric(enum metric_t type, int fd, enum control_type_t source)
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
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "instance",            1),
            FIELDEND
        };
        event_t e = {"net.port", g_openPorts, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:OPENPORTS:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case TCP_CONNECTIONS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "session",             1),
            FIELDEND
        };
        event_t e = {"net.tcp", g_TCPConnections, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:TCPCONNS:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case ACTIVE_CONNECTIONS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "connection",          1),
            FIELDEND
        };
        event_t e = {"net.conn", g_activeConnections, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:ACTIVECONNS:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        atomicSet(&g_activeConnections, 0);
        break;
    }

    case NETRX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

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
        
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("proto",            proto,                 1),
            STRFIELD("localip",          lip,                   5),
            NUMFIELD("localp",           localPort,             5),
            STRFIELD("remoteip",         rip,                   5),
            NUMFIELD("remotep",          remotePort,            5),
            STRFIELD("data",             data,                  9),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.rx", g_netrx, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETRX:outSendEvent\n", -1, CFG_LOG_ERROR);
        }

        /*
         * This creates DELTA behavior by uploading the number of bytes
         * since the last time the metric was uploaded.
         */
        atomicSet(&g_netrx, 0);
        break;
    }

    case NETTX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

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

        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("proto",            proto,                 1),
            STRFIELD("localip",          lip,                   5),
            NUMFIELD("localp",           localPort,             5),
            STRFIELD("remoteip",         rip,                   5),
            NUMFIELD("remotep",          remotePort,            5),
            STRFIELD("data",             data,                  9),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.tx", g_nettx, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETTX:outSendEvent\n", -1, CFG_LOG_ERROR);
        }

        atomicSet(&g_nettx, 0);
        break;
    }

    case NETRX_PROC:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.rx", g_netrx, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETRX_PROC:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        atomicSet(&g_netrx, 0);
        break;
    }

    case NETTX_PROC:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.tx", g_nettx, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETTX_PROC:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        atomicSet(&g_nettx, 0);
        break;
    }

    case DNS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("domain",           g_netinfo[fd].dnsName, 6),
            STRFIELD("unit",             "request",             1),
            FIELDEND
        };

        // Increment the DNS counter by one for each event
        event_t e = {"net.dns", 1, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:DNS:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
        break;
    }

    default:
        scopeLog("ERROR: doNetMetric:metric type\n", -1, CFG_LOG_ERROR);
    }
}

// Return process specific CPU usage in microseconds
static long long
doGetProcCPU() {
    struct rusage ruse;
    
    if (getrusage(RUSAGE_SELF, &ruse) != 0) {
        return (long long)-1;
    }

    return
        (((long long)ruse.ru_utime.tv_sec + (long long)ruse.ru_stime.tv_sec) * 1000 * 1000) +
        ((long long)ruse.ru_utime.tv_usec + (long long)ruse.ru_stime.tv_usec);
}

// Return process specific memory usage in kilobytes
static long
doGetProcMem() {
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

static void
doSetConnection(int sd, const struct sockaddr *addr, socklen_t len, enum control_type_t endp)
{
    if (!addr || (len <= 0)) {
        return;
    }
    
    // Should we check for at least the size of sockaddr_in?
    if (g_netinfo && (g_netinfo[sd].fd == sd) &&
        addr && (len > 0)) {
        if (endp == LOCAL) {
            memmove(&g_netinfo[sd].localConn, addr, len);
        } else {
            memmove(&g_netinfo[sd].remoteConn, addr, len);
        }
    }
}

static int
doSetAddrs(int sockfd)
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
static int
doAddNewSock(int sockfd)
{
    struct sockaddr addr;
    socklen_t addrlen = sizeof(struct sockaddr);
        
    scopeLog("doAddNewSock: adding socket\n", sockfd, CFG_LOG_DEBUG);
    if (getsockname(sockfd, &addr, &addrlen) != -1) {
        if ((addr.sa_family == AF_INET) || (addr.sa_family == AF_INET6)) {
            int type;
            socklen_t len = sizeof(socklen_t);
                
            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type);
            } else {
                // Really can't add the socket at this point
                scopeLog("ERROR: doRecv:getsockopt\n", sockfd, CFG_LOG_ERROR);
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
 *
 * name format:
 * octet of len followed by a label of len octets
 * len is <=63 and total len octets + labels <= 255
 */

static int
getDNSName(int sd, void *pkt, int pktlen)
{
    int llen;
    dns_query *query;
    struct question *q;
    char *aname, *dname;

    if (g_netinfo && (g_netinfo[sd].type == SOCK_STREAM)) {
        return 0;
    }
    
    query = (struct dns_query_t *)pkt;
    if ((dname = (char *)&query->name) == NULL) {
        return -1;
    }

    q = (struct question *)(pkt + sizeof(struct dns_header) + strlen(dname));
    if (q->qclass != 1) {
        return 0;
    }

    aname = g_netinfo[sd].dnsName;

    while (*dname != '\0') {
        // handle one label
        for (llen = (int)*dname++; llen > 0; llen--) {
            *aname++ = *dname++;
        }
        
        *aname++ = '.';
    }

    aname--;
    *aname = '\0';
    return 0;
}


static int
doRecv(int sockfd, ssize_t rc)
{
    atomicAdd(&g_netrx, rc);
    if (g_netinfo && (g_netinfo[sockfd].fd != sockfd)) {
        doAddNewSock(sockfd);
    }

    doSetAddrs(sockfd);
    doNetMetric(NETRX, sockfd, EVENT_BASED);

    return 0;
}

static int
doSend(int sockfd, ssize_t rc)
{
    atomicAdd(&g_nettx, rc);
    if (g_netinfo && (g_netinfo[sockfd].fd != sockfd)) {
        doAddNewSock(sockfd);
    }

    doSetAddrs(sockfd);
    doNetMetric(NETTX, sockfd, EVENT_BASED);

    if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
        doNetMetric(DNS, sockfd, EVENT_BASED);
    }

    return 0;
}

static void
doAccept(int sd, struct sockaddr *addr, socklen_t addrlen, char *func)
{

    scopeLog(func, sd, CFG_LOG_DEBUG);
    addSock(sd, SOCK_STREAM);
    
    if (g_netinfo && (g_netinfo[sd].fd == sd)) {
        g_netinfo[sd].listen = TRUE;
        g_netinfo[sd].accept = TRUE;
        atomicAdd(&g_openPorts, 1);
        atomicAdd(&g_TCPConnections, 1);
        atomicAdd(&g_activeConnections, 1);
        doSetConnection(sd, addr, addrlen, REMOTE);
        doNetMetric(OPEN_PORTS, sd, EVENT_BASED);
        doNetMetric(TCP_CONNECTIONS, sd, EVENT_BASED);
        doNetMetric(ACTIVE_CONNECTIONS, sd, EVENT_BASED);
    }
}

static void *
periodic(void *arg)
{
    long mem;
    int nthread, nfds, children;
    pid_t pid = getpid();
    long long cpu, cpuState = 0;

    while (1) {
        // We report CPU time for this period.
        cpu = doGetProcCPU();
        doProcMetric(PROC_CPU, cpu - cpuState);
        cpuState = cpu;
        
        mem = doGetProcMem();
        doProcMetric(PROC_MEM, mem);

        nthread = osGetNumThreads(pid);
        doProcMetric(PROC_THREAD, nthread);

        nfds = osGetNumFds(pid);
        doProcMetric(PROC_FD, nfds);

        children = osGetNumChildProcs(pid);
        doProcMetric(PROC_CHILD, children);

        doNetMetric(NETRX_PROC, -1, PERIODIC);
        doNetMetric(NETTX_PROC, -1, PERIODIC);
        
        // From the config file
        sleep(g_thread.interval);
    }

    return NULL;
}

__attribute__((constructor)) void
init(void)
{
    g_fn.vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    g_fn.fork = dlsym(RTLD_NEXT, "fork");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.fcntl = dlsym(RTLD_NEXT, "fcntl");
    g_fn.fcntl64 = dlsym(RTLD_NEXT, "fcntl64");
    g_fn.dup = dlsym(RTLD_NEXT, "dup");
    g_fn.dup2 = dlsym(RTLD_NEXT, "dup2");
    g_fn.dup3 = dlsym(RTLD_NEXT, "dup3");
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
    g_fn.__sendto_nocancel = dlsym(RTLD_NEXT, "__sendto_nocancel");
    g_fn.DNSServiceQueryRecord = dlsym(RTLD_NEXT, "DNSServiceQueryRecord");
#endif // __MACOS__

    if ((g_netinfo = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc\n", -1, CFG_LOG_ERROR);
    }

    g_numNinfo = NET_ENTRIES;
    if (gethostname(g_hostname, sizeof(g_hostname)) != 0) {
        scopeLog("ERROR: Constructor:gethostname\n", -1, CFG_LOG_ERROR);
    }

    osGetProcname(g_procname, sizeof(g_procname));

    {
        char* path = cfgPath(CFG_FILE_NAME);
        config_t* cfg = cfgRead(path);
        g_thread.interval = cfgOutPeriod(cfg);
        g_thread.startTime = time(NULL) + g_thread.interval;
        log_t* log = initLog(cfg);
        g_out = initOut(cfg, log);
        g_log = log; // Set after initOut to avoid infinite loop with socket
        cfgDestroy(&cfg);
        if (path) free(path);
    }

    scopeLog("Constructor\n", -1, CFG_LOG_INFO);
}

static void
doClose(int fd, char *func)
{
    if (g_netinfo && (g_netinfo[fd].fd == fd)) {
        scopeLog(func, fd, CFG_LOG_DEBUG);
        if (g_netinfo[fd].listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_openPorts, 1);
            doNetMetric(OPEN_PORTS, fd, EVENT_BASED);
        }

        if (g_netinfo[fd].accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_TCPConnections, 1);
            doNetMetric(TCP_CONNECTIONS, fd, EVENT_BASED);
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
    }
}

EXPORTON int
close(int fd)
{
    int rc;

    WRAP_CHECK_ONLY(close);
    doThread(); // Will do nothing if a thread already exists

    rc = g_fn.close(fd);
    if (rc != -1) {
        doClose(fd, "close\n");
    }
    
    return rc;
}

#ifdef __MACOS__
EXPORTON int
close$NOCANCEL(int fd)
{
    int rc;

    WRAP_CHECK(close$NOCANCEL);
    doThread();
    rc = g_fn.close$NOCANCEL(fd);
    if (rc != -1) {
        doClose(fd, "close$NOCANCEL\n");
    }
    
    return rc;
}


EXPORTON int
guarded_close_np(int fd, void *guard)
{
    int rc;

    WRAP_CHECK(guarded_close_np);
    doThread();
    rc = g_fn.guarded_close_np(fd, guard);
    if (rc != -1) {
        doClose(fd, "guarded_close_np\n");
    }
    
    return rc;
}

EXPORTOFF int
close_nocancel(int fd)
{
    int rc;

    WRAP_CHECK(close_nocancel);
    rc = g_fn.close_nocancel(fd);
    if (rc != -1) {
        doClose(fd, "close_nocancel\n");
    }
    
    return rc;
}

EXPORTON int
accept$NOCANCEL(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept$NOCANCEL);
    doThread();
    sd = g_fn.accept$NOCANCEL(sockfd, addr, addrlen);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept$NOCANCEL\n");
    }

    return sd;
}

EXPORTON ssize_t
__sendto_nocancel(int sockfd, const void *buf, size_t len, int flags,
                  const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;

    WRAP_CHECK(__sendto_nocancel);
    doThread();
    rc = g_fn.__sendto_nocancel(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog("__sendto_nocancel\n", sockfd, CFG_LOG_TRACE);

        doSetAddrs(sockfd);

        if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, (void *)buf, len);
        }
        
        doSend(sockfd, rc);
    }

    return rc;
}

EXPORTON uint32_t
DNSServiceQueryRecord(void *sdRef, uint32_t flags, uint32_t interfaceIndex,
                      const char *fullname, uint16_t rrtype, uint16_t rrclass,
                      void *callback, void *context)
{
    uint32_t rc;

    WRAP_CHECK(DNSServiceQueryRecord);
    rc = g_fn.DNSServiceQueryRecord(sdRef, flags, interfaceIndex, fullname,
                                    rrtype, rrclass, callback, context);
    if (rc == 0) {
        scopeLog("DNSServiceQueryRecord\n", -1, CFG_LOG_DEBUG);

        event_field_t fields[] = {
            STRFIELD("proc",             g_procname,            2),
            NUMFIELD("pid",              getpid(),              7),
            STRFIELD("host",             g_hostname,            2),
            STRFIELD("domain",           fullname,              6),
            STRFIELD("unit",             "request",             1),
            FIELDEND
        };

        event_t e = {"net.dns", 1, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: DNSServiceQueryRecord:DNS:outSendEvent\n", -1, CFG_LOG_ERROR);
        }
    }

    return rc;
}

#endif // __MACOS__

EXPORTON ssize_t
write(int fd, const void *buf, size_t count)
{
    ssize_t rc;

    WRAP_CHECK(write);
    doThread();
    rc = g_fn.write(fd, buf, count);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd)) {
        // This is a network descriptor
        scopeLog("write\n", fd, CFG_LOG_TRACE);
        doSetAddrs(fd);
        doSend(fd, rc);
    }
    
    return rc;
}

EXPORTON ssize_t
read(int fd, void *buf, size_t count)
{
    ssize_t rc;

    WRAP_CHECK_ONLY(read);
    doThread();
    rc = g_fn.read(fd, buf, count);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd)) {
        // This is a network descriptor
        scopeLog("read\n", fd, CFG_LOG_TRACE);
        doSetAddrs(fd);
        doRecv(fd, rc);
    }
    
    return rc;
}

EXPORTON int
fcntl(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                    fArgs.arg[2], fArgs.arg[3]);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd) &&
        (cmd == F_DUPFD)) {
        // This is a network descriptor
        scopeLog("fcntl\n", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }
    
    return rc;
}

EXPORTON int
fcntl64(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl64);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl64(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                      fArgs.arg[2], fArgs.arg[3]);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd) &&
        (cmd == F_DUPFD)) {
        // This is a network descriptor
        scopeLog("fcntl\n", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }
    
    return rc;
}

EXPORTON int
dup(int fd)
{
    int rc;

    WRAP_CHECK(dup);
    doThread();
    rc = g_fn.dup(fd);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[fd].fd == fd)) {
        // This is a network descriptor
        scopeLog("dup\n", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }

    return rc;
}

EXPORTON int
dup2(int oldfd, int newfd)
{
    int rc;

    WRAP_CHECK(dup2);
    doThread();
    rc = g_fn.dup2(oldfd, newfd);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[oldfd].fd == oldfd)) {
        // This is a network descriptor
        scopeLog("dup2\n", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }

    return rc;
}

EXPORTON int
dup3(int oldfd, int newfd, int flags)
{
    int rc;

    WRAP_CHECK(dup3);
    doThread();
    rc = g_fn.dup3(oldfd, newfd, flags);
    if ((rc != -1) && (g_netinfo) && (g_netinfo[oldfd].fd == oldfd)) {
        // This is a network descriptor
        scopeLog("dup3\n", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }

    return rc;
}

EXPORTOFF void
vsyslog(int priority, const char *format, va_list ap)
{
    WRAP_CHECK_VOID(vsyslog);
    scopeLog("vsyslog\n", -1, CFG_LOG_DEBUG);
    g_fn.vsyslog(priority, format, ap);
    return;
}

EXPORTON pid_t
fork()
{
    pid_t rc;

    WRAP_CHECK(fork);
    doThread();
    scopeLog("fork\n", -1, CFG_LOG_DEBUG);
    rc = g_fn.fork();
    if (rc == 0) {
        // We are the child proc
        doReset();
    }
    
    return rc;
}

EXPORTON int
socket(int socket_family, int socket_type, int protocol)
{
    int sd;

    WRAP_CHECK(socket);
    doThread();
    sd = g_fn.socket(socket_family, socket_type, protocol);
    if (sd != -1) {
        scopeLog("socket\n", sd, CFG_LOG_DEBUG);
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
            doNetMetric(OPEN_PORTS, sd, EVENT_BASED);
        }
    }

    return sd;
}

EXPORTON int
shutdown(int sockfd, int how)
{
    int rc;

    WRAP_CHECK(shutdown);
    doThread();
    rc = g_fn.shutdown(sockfd, how);
    if (rc != -1) {
        doClose(sockfd, "shutdown\n");
    }
    
    return rc;
}

EXPORTON int
listen(int sockfd, int backlog)
{
    int rc;

    WRAP_CHECK(listen);
    doThread();
    rc = g_fn.listen(sockfd, backlog);
    if (rc != -1) {
        scopeLog("listen\n", sockfd, CFG_LOG_DEBUG);
        
        // Tracking number of open ports
        atomicAdd(&g_openPorts, 1);
        
        if (g_netinfo && (g_netinfo[sockfd].fd == sockfd)) {
            g_netinfo[sockfd].listen = TRUE;
            g_netinfo[sockfd].accept = TRUE;
            doNetMetric(OPEN_PORTS, sockfd, EVENT_BASED);

            if (g_netinfo[sockfd].type == SOCK_STREAM) {
                atomicAdd(&g_TCPConnections, 1);
                g_netinfo[sockfd].accept = TRUE;                            
                doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED);
            }
        }
    }
    
    return rc;
}

EXPORTON int
accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept);
    doThread();
    sd = g_fn.accept(sockfd, addr, addrlen);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept\n");
    }

    return sd;
}

EXPORTON int
accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags)
{
    int sd;

    WRAP_CHECK(accept4);
    doThread();
    sd = g_fn.accept4(sockfd, addr, addrlen, flags);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept4\n");
    }

    return sd;
}

EXPORTON int
bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;

    WRAP_CHECK(bind);
    doThread();
    rc = g_fn.bind(sockfd, addr, addrlen);
    if (rc != -1) { 
        doSetConnection(sockfd, addr, addrlen, LOCAL);
        scopeLog("bind\n", sockfd, CFG_LOG_DEBUG);
    }
    
    return rc;

}

EXPORTON int
connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;

    WRAP_CHECK(connect);
    doThread();
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
            doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED);
        }
        scopeLog("connect\n", sockfd, CFG_LOG_DEBUG);
    }
    
    return rc;

}

EXPORTON ssize_t
send(int sockfd, const void *buf, size_t len, int flags)
{
    ssize_t rc;

    WRAP_CHECK(send);
    doThread();
    rc = g_fn.send(sockfd, buf, len, flags);
    if (rc != -1) {
        scopeLog("send\n", sockfd, CFG_LOG_TRACE);
        if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc);
    }
    
    return rc;
}

EXPORTON ssize_t
sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;

    WRAP_CHECK(sendto);
    doThread();
    rc = g_fn.sendto(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog("sendto\n", sockfd, CFG_LOG_TRACE);
        doSetConnection(sockfd, dest_addr, addrlen, REMOTE);

        if (g_netinfo && GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc);
    }

    return rc;
}

EXPORTON ssize_t
sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    ssize_t rc;

    WRAP_CHECK(sendmsg);
    doThread();
    rc = g_fn.sendmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog("sendmsg\n", sockfd, CFG_LOG_TRACE);

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

EXPORTON ssize_t
recv(int sockfd, void *buf, size_t len, int flags)
{
    ssize_t rc;

    WRAP_CHECK(recv);
    doThread();
    scopeLog("recv\n", sockfd, CFG_LOG_TRACE);
    rc = g_fn.recv(sockfd, buf, len, flags);
    if (rc != -1) {
        doRecv(sockfd, rc);
    }
    
    return rc;
}

EXPORTON ssize_t
recvfrom(int sockfd, void *buf, size_t len, int flags,
         struct sockaddr *src_addr, socklen_t *addrlen)
{
    ssize_t rc;

    WRAP_CHECK(recvfrom);
    doThread();
    scopeLog("recvfrom\n", sockfd, CFG_LOG_TRACE);
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
                scopeLog("ERROR: recvfrom:getsockopt\n", sockfd, CFG_LOG_ERROR);
            }
        }

        if (src_addr && addrlen) {
            doSetConnection(sockfd, src_addr, *addrlen, REMOTE);
        }

        doNetMetric(NETRX, sockfd, EVENT_BASED);
    }
    return rc;
}

EXPORTON ssize_t
recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    ssize_t rc;

    WRAP_CHECK(recvmsg);
    doThread();
    rc = g_fn.recvmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog("recvmsg\n", sockfd, CFG_LOG_TRACE);

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
