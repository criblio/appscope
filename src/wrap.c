#define _GNU_SOURCE
#include <dlfcn.h>
#include "cfg.h"
#include "scopetypes.h"
#include "cfgutils.h"
#include "dbg.h"
#include "log.h"
#include "out.h"
#include "wrap.h"

interposed_funcs g_fn;
rtconfig g_cfg = {0};
static net_info *g_netinfo;
static fs_info *g_fsinfo;
static metric_counters g_ctrs = {0};
static thread_timing g_thread = {0};
static log_t* g_log = NULL;
static out_t* g_out = NULL;

// Forward declaration
static void * periodic(void *);
    
EXPORTOFF void
reinitLogDescriptor()
{
    char* path = cfgPath(CFG_FILE_NAME);
    config_t* cfg = cfgRead(path);
    cfgProcessEnvironment(cfg);

    logDestroy(&g_log);
    g_log = initLog(cfg);

    if (path) free(path);
    cfgDestroy(&cfg);
}

EXPORTOFF void
reinitMetricDescriptor()
{
    char* path = cfgPath(CFG_FILE_NAME);
    config_t* cfg = cfgRead(path);
    cfgProcessEnvironment(cfg);

    outDestroy(&g_out);
    g_out = initOut(cfg, NULL);

    if (path) free(path);
    cfgDestroy(&cfg);
}

EXPORTON void
scopeLog(const char* msg, int fd, cfg_log_level_t level)
{
    if (!g_log || !msg || !g_cfg.procname[0]) return;

    char buf[strlen(msg) + 128];
    snprintf(buf, sizeof(buf), "Scope: %s(%d): %s\n", g_cfg.procname, fd, msg);
    if (logSend(g_log, buf, level) == DEFAULT_BADFD) {
        // We lost our fd, re-open
        reinitLogDescriptor();
    }
}

EXPORTOFF void
sendEvent(out_t* out, event_t* e)
{
    int rc;

    rc = outSendEvent(g_out, e);
    if (rc == DEFAULT_BADFD) {
        // We lost our fd, re-open
        reinitMetricDescriptor();
    } else if (rc == -1) {
        scopeLog("ERROR: doProcMetric:CPU:outSendEvent", -1, CFG_LOG_ERROR);
    }
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
    snprintf(buf, sizeof(buf), "%s:%d LOCAL: %s:%d", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);

    inet_ntop(AF_INET,                                          
              &((struct sockaddr_in *)&g_netinfo[sd].remoteConn)->sin_addr,
              ip, sizeof(ip));
    port = GET_PORT(sd, g_netinfo[sd].remoteConn.ss_family, REMOTE);
    snprintf(buf, sizeof(buf), "%s:%d REMOTE:%s:%d", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);
    
    if (GET_PORT(sd, g_netinfo[sd].localConn.ss_family, REMOTE) == DNS_PORT) {
        scopeLog("DNS", sd, CFG_LOG_DEBUG);
    }
}

// Return the time delta from start to now in nanoseconds
EXPORTON uint64_t
getDuration(uint64_t start)
{
    /*
     * The clock frequency is in Mhz.
     * In order to get NS resolution we
     * multiply the difference by 1000.
     *
     * If the counter rolls over we adjust
     * by using the max value of the counter.
     * A roll over is rare. But, we should handle it.  
     */
    uint64_t now = getTime();
    if (start < now) {
        return ((now - start) * 1000) / g_cfg.freq;
    } else {
        return (((ULONG_MAX - start) + now) * 1000) / g_cfg.freq;
    }
    
}

static bool
checkNetEntry(int fd)
{
    if (g_netinfo && (fd > 0) && (fd <= g_cfg.numNinfo)) {
        return TRUE;
    }

    return FALSE;
}

static bool
checkFSEntry(int fd)
{
    if (g_fsinfo && (fd > 0) && (fd <= g_cfg.numFSInfo)) {
        return TRUE;
    }

    return FALSE;
}

static net_info *
getNetEntry(int fd)
{
    if (g_netinfo && (fd > 0) && (fd <= g_cfg.numNinfo) &&
        (g_netinfo[fd].fd == fd)) {
        return &g_netinfo[fd];
    }
    return NULL;    
}

static fs_info *
getFSEntry(int fd)
{
    if (g_fsinfo && (fd > 0) && (fd <= g_cfg.numFSInfo) &&
        (g_fsinfo[fd].fd == fd)) {
        return &g_fsinfo[fd];
    }
    return NULL;    
}

static void
addSock(int fd, int type)
{
    if (checkNetEntry(fd) == TRUE) {
        if (g_netinfo[fd].fd == fd) {
            scopeLog("addSock: duplicate", fd, CFG_LOG_DEBUG);

            if (g_ctrs.openPorts > 0) {
                atomicSub(&g_ctrs.openPorts, 1);
            }
            
            if (g_ctrs.TCPConnections > 0) {
                atomicSub(&g_ctrs.TCPConnections, 1);
            }

            g_netinfo[fd].type = type;
#ifdef __LINUX__
            // Clear these bits so comparisons of type will work
            g_netinfo[fd].type &= ~SOCK_CLOEXEC;
            g_netinfo[fd].type &= ~SOCK_NONBLOCK;
#endif // __LINUX__

            return;
        }
        
        if ((fd > g_cfg.numNinfo) && (fd < MAX_FDS))  {
            int increase;
            net_info *temp;

            if (fd < (MAX_FDS / 2)) {
                increase = MAX_FDS / 2;
            } else {
                increase = MAX_FDS;
            }

            // Need to realloc
            if ((temp = realloc(g_netinfo, sizeof(struct net_info_t) * increase)) == NULL) {
                scopeLog("ERROR: addSock:realloc", fd, CFG_LOG_ERROR);
                DBG("re-alloc on Net table failed");
            } else {
                memset(&temp[g_cfg.numNinfo], 0, sizeof(struct net_info_t) * (increase - g_cfg.numNinfo));
                g_cfg.numNinfo = increase;
                g_netinfo = temp;
            }
        }

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].fd = fd;
        g_netinfo[fd].type = type;
#ifdef __LINUX__
        // Clear these bits so comparisons of type will work
        g_netinfo[fd].type &= ~SOCK_CLOEXEC;
        g_netinfo[fd].type &= ~SOCK_NONBLOCK;
#endif // __LINUX__
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
            scopeLog("ERROR: doThread:pthread_create", -1, CFG_LOG_ERROR);
        }
    }
}

static void
doDNSMetricName(const char *domain)
{
    if (!domain) return;

    event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              getpid(),              7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("domain",           domain,                2),
            STRFIELD("unit",             "request",             1),
            FIELDEND
    };

    event_t e = {"net.dns", 1, DELTA, fields};
    if (outSendEvent(g_out, &e)) {
        scopeLog("ERROR: doDNSMetricName:DNS:outSendEvent", -1, CFG_LOG_ERROR);
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
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "microsecond",         1),
            FIELDEND
        };
        event_t e = {"proc.cpu", measurement, DELTA, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_MEM:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "kibibyte",            1),
            FIELDEND
        };
        event_t e = {"proc.mem", measurement, DELTA, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_THREAD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "thread",              1),
            FIELDEND
        };
        event_t e = {"proc.thread", measurement, CURRENT, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_FD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "file",                1),
            FIELDEND
        };
        event_t e = {"proc.fd", measurement, CURRENT, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_CHILD:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "process",             1),
            FIELDEND
        };
        event_t e = {"proc.child", measurement, CURRENT, fields};
        sendEvent(g_out, &e);
        break;
    }

    default:
        scopeLog("ERROR: doProcMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

static void
doFSMetric(enum metric_t type, int fd, enum control_type_t source,
           const char *op, ssize_t size, const char *pathname)
{
    pid_t pid = getpid();
    fs_info *fs;
    
    if ((fs = getFSEntry(fd)) == NULL) {
        return;
    }


    switch (type) {
    case FS_DURATION:
    {
        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            g_fsinfo[fd].action |= EVENT_FS;
            atomicAdd(&g_fsinfo[fd].numDuration, 1);
            atomicAdd(&g_fsinfo[fd].totalDuration, size);
        }

        // Only report if enabled
        if ((g_cfg.verbosity != CFG_FS_EVENTS_VERBOSITY) &&
            (g_cfg.verbosity != CFG_NET_FS_EVENTS_VERBOSITY) &&
            (source == EVENT_BASED)) {
            return;
        }

        uint64_t d = 0ULL;
        if (g_fsinfo[fd].numDuration >= 1) {
            // factor of 1000 converts us to ms.
            d = g_fsinfo[fd].totalDuration / ( 1000 * g_fsinfo[fd].numDuration);
        }

        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("op",               op,                    2),
            STRFIELD("file",             g_fsinfo[fd].path,     2),
            NUMFIELD("numops",        g_fsinfo[fd].numDuration, 8),
            STRFIELD("unit",             "millisecond",         1),
            FIELDEND
        };
        event_t e = {"fs.duration", d, HISTOGRAM, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doFSMetric:FS_DURATION:outSendEvent", fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        //g_fsinfo[fd].action &= ~EVENT_FS;
        atomicSet(&g_fsinfo[fd].numDuration, 0);
        atomicSet(&g_fsinfo[fd].totalDuration, 0);
        break;        
    }

    case FS_READ:
    case FS_WRITE:
    {
        const char* metric = "UNKNOWN";
        int* numops = NULL;
        int* sizebytes = NULL;
        const char* err_str = "UNKNOWN";
        switch (type) {
            case FS_READ:
                metric = "fs.read";
                numops = &g_fsinfo[fd].numRead;
                sizebytes = &g_fsinfo[fd].readBytes;
                err_str = "ERROR: doFSMetric:FS_READ:outSendEvent";
                break;
            case FS_WRITE:
                metric = "fs.write";
                numops = &g_fsinfo[fd].numWrite;
                sizebytes = &g_fsinfo[fd].writeBytes;
                err_str = "ERROR: doFSMetric:FS_WRITE:outSendEvent";
                break;
            default:
                DBG(NULL);
                return;
	    }

        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            g_fsinfo[fd].action |= EVENT_FS;
            atomicAdd(numops, 1);
            atomicAdd(sizebytes, size);
        }

        // Only report if enabled
        if ((g_cfg.verbosity != CFG_FS_EVENTS_VERBOSITY) &&
            (g_cfg.verbosity != CFG_NET_FS_EVENTS_VERBOSITY) &&
            (source == EVENT_BASED)) {
            return;
        }

        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("op",               op,                    2),
            STRFIELD("file",             g_fsinfo[fd].path,     2),
            NUMFIELD("numops",           *numops,               8),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {metric, *sizebytes, HISTOGRAM, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog(err_str, fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        //g_fsinfo[fd].action &= ~EVENT_FS;
        atomicSet(numops, 0);
        atomicSet(sizebytes, 0);

        break;
    }
    
    case FS_OPEN:
    case FS_CLOSE:
    case FS_SEEK:
    case FS_STAT:
    {
        const char* metric = "UNKNOWN";
        int* numops = NULL;
        const char* err_str = "UNKNOWN";

        switch (type) {
            case FS_OPEN:
                metric = "fs.op.open";
                numops = &g_fsinfo[fd].numOpen;
                err_str = "ERROR: doFSMetric:FS_OPEN:outSendEvent";
                break;
            case FS_CLOSE:
                metric = "fs.op.close";
                numops = &g_fsinfo[fd].numClose;
                err_str = "ERROR: doFSMetric:FS_CLOSE:outSendEvent";
                break;
            case FS_SEEK:
                metric = "fs.op.seek";
                numops = &g_fsinfo[fd].numSeek;
                err_str = "ERROR: doFSMetric:FS_SEEK:outSendEvent";
                break;
            case FS_STAT:
                metric = "fs.op.stat";
                numops = &g_fsinfo[fd].numStat;
                err_str = "ERROR: doFSMetric:FS_STAT:outSendEvent";
                break;
            default:
                DBG(NULL);
                return;
        }

        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            g_fsinfo[fd].action |= EVENT_FS;
            atomicAdd(numops, 1);
        }

        // Only report if enabled
        if (((type == FS_STAT) || (type == FS_SEEK)) &&
            (g_cfg.verbosity != CFG_FS_EVENTS_VERBOSITY) &&
            (g_cfg.verbosity != CFG_NET_FS_EVENTS_VERBOSITY) &&
            (source == EVENT_BASED)) {
            return;
        }

        // This stat func passes a path and not an fd
        if ((fd == -1) && (pathname)) {
            strncpy(g_fsinfo[fd].path, pathname, sizeof(g_fsinfo[fd].path));
        }
        
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("op",               op,                    2),
            STRFIELD("file",             g_fsinfo[fd].path,     2),
            STRFIELD("unit",             "operation",           1),
            FIELDEND
        };

        event_t e = {metric, *numops, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog(err_str, fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        //g_fsinfo[fd].action &= ~EVENT_FS;
        atomicSet(numops, 0);

        break;
    }
    
    default:
        DBG(NULL);
        scopeLog("ERROR: doFSMetric:metric type", fd, CFG_LOG_ERROR);
    }
}


static void
doTotal(enum metric_t type, pid_t pid)
{
    const char* metric = "UNKNOWN";
    int* sizebytes = NULL;
    const char* err_str = "UNKNOWN";
    switch (type) {
        case TOT_READ:
            metric = "fs.read.total";
            sizebytes = &g_ctrs.readBytes;
            err_str = "ERROR: doTotal:TOT_READ:outSendEvent";
            break;
        case TOT_WRITE:
            metric = "fs.write.total";
            sizebytes = &g_ctrs.writeBytes;
            err_str = "ERROR: doTotal:TOT_WRITE:outSendEvent";
            break;
        case TOT_RX:
            metric = "net.rx.total";
            sizebytes = &g_ctrs.netrxBytes;
            err_str = "ERROR: doTotal:TOT_READ:outSendEvent";
            break;
        case TOT_TX:
            metric = "net.tx.total";
            sizebytes = &g_ctrs.nettxBytes;
            err_str = "ERROR: doTotal:TOT_READ:outSendEvent";
            break;
        default:
            DBG(NULL);
            return;
	}

    event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
    };
    event_t e = {metric, *sizebytes, DELTA, fields};
    if (outSendEvent(g_out, &e)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report
    atomicSet(sizebytes, 0);
}


static void
doNetMetric(enum metric_t type, int fd, enum control_type_t source, ssize_t size)
{
    pid_t pid = getpid();
    char proto[PROTOCOL_STR];
    in_port_t localPort, remotePort;
        
    if (getNetEntry(fd) == NULL) {
        return;
    }

    getProtocol(g_netinfo[fd].type, proto, sizeof(proto));
    localPort = GET_PORT(fd, g_netinfo[fd].localConn.ss_family, LOCAL);
    remotePort = GET_PORT(fd, g_netinfo[fd].remoteConn.ss_family, REMOTE);
    
    switch (type) {
    case OPEN_PORTS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "instance",            1),
            FIELDEND
        };
        event_t e = {"net.port", g_ctrs.openPorts, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:OPENPORTS:outSendEvent", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case TCP_CONNECTIONS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "session",             1),
            FIELDEND
        };
        event_t e = {"net.tcp", g_ctrs.TCPConnections, CURRENT, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:TCPCONNS:outSendEvent", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case ACTIVE_CONNECTIONS:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "connection",          1),
            FIELDEND
        };
        event_t e = {"net.conn", g_ctrs.activeConnections, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:ACTIVECONNS:outSendEvent", -1, CFG_LOG_ERROR);
        }
        atomicSet(&g_ctrs.activeConnections, 0);
        break;
    }

    case CONNECTION_DURATION:
    {
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            NUMFIELD("port",             localPort,             5),
            STRFIELD("unit",             "millisecond",         1),
            FIELDEND
        };
        event_t e = {"net.conn_duration", g_netinfo[fd].duration, DELTA_MS, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:CONNECTION_DURATION:outSendEvent", fd, CFG_LOG_ERROR);
        }
        break;
    }

    case NETRX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        if (source == EVENT_BASED) {
            g_netinfo[fd].action |= EVENT_RX;
            atomicAdd(&g_netinfo[fd].numRX, 1);
            atomicAdd(&g_netinfo[fd].rxBytes, size);
            atomicAdd(&g_ctrs.netrxBytes, size);
        }

        if ((g_cfg.verbosity != CFG_NET_EVENTS_VERBOSITY) &&
            (g_cfg.verbosity != CFG_NET_FS_EVENTS_VERBOSITY) &&
            (source == EVENT_BASED)) {
            return;
        }

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if ((g_netinfo[fd].type == SCOPE_UNIX) ||
            (g_netinfo[fd].localConn.ss_family == AF_LOCAL) ||
            (g_netinfo[fd].localConn.ss_family == AF_NETLINK)) {
            strncpy(lip, "UNIX", sizeof(lip));
            strncpy(rip, "UNIX", sizeof(rip));
            localPort = remotePort = 0;
            if (g_netinfo[fd].localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }
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
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            STRFIELD("localip",          lip,                   5),
            NUMFIELD("localp",           localPort,             5),
            STRFIELD("remoteip",         rip,                   5),
            NUMFIELD("remotep",          remotePort,            5),
            STRFIELD("data",             data,                  1),
            NUMFIELD("numops",           g_netinfo[fd].numRX,   8),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.rx", g_netinfo[fd].rxBytes, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETRX:outSendEvent", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        //g_netinfo[fd].action &= ~EVENT_RX;
        atomicSet(&g_netinfo[fd].numRX, 0);
        atomicSet(&g_netinfo[fd].rxBytes, 0);
        //atomicSet(&g_ctrs.netrx, 0);

        break;
    }

    case NETTX:
    {
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        if (source == EVENT_BASED) {
            g_netinfo[fd].action |= EVENT_TX;
            atomicAdd(&g_netinfo[fd].numTX, 1);
            atomicAdd(&g_netinfo[fd].txBytes, size);
            atomicAdd(&g_ctrs.nettxBytes, size);
        }

        if ((g_cfg.verbosity != CFG_NET_EVENTS_VERBOSITY) &&
            (g_cfg.verbosity != CFG_NET_FS_EVENTS_VERBOSITY) &&
            (source == EVENT_BASED)) {
            return;
        }

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if ((g_netinfo[fd].type == SCOPE_UNIX) ||
            (g_netinfo[fd].localConn.ss_family == AF_LOCAL) ||
            (g_netinfo[fd].localConn.ss_family == AF_NETLINK)) {
            strncpy(lip, "UNIX", sizeof(lip));
            strncpy(rip, "UNIX", sizeof(rip));
            localPort = remotePort = 0;
            if (g_netinfo[fd].localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }
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
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            NUMFIELD("fd",               fd,                    7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("proto",            proto,                 1),
            STRFIELD("localip",          lip,                   5),
            NUMFIELD("localp",           localPort,             5),
            STRFIELD("remoteip",         rip,                   5),
            NUMFIELD("remotep",          remotePort,            5),
            STRFIELD("data",             data,                  1),
            NUMFIELD("numops",           g_netinfo[fd].numTX,   8),
            STRFIELD("unit",             "byte",                1),
            FIELDEND
        };
        event_t e = {"net.tx", g_netinfo[fd].txBytes, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:NETTX:outSendEvent", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        //g_netinfo[fd].action &= ~EVENT_TX;
        atomicSet(&g_netinfo[fd].numTX, 0);
        atomicSet(&g_netinfo[fd].txBytes, 0);
        //atomicSet(&g_ctrs.nettx, 0);

        break;
    }

    case DNS:
    {
        if (g_netinfo[fd].dnsSend == FALSE) {
            break;
        }

        // For next time
        g_netinfo[fd].dnsSend = FALSE;
        
        event_field_t fields[] = {
            STRFIELD("proc",             g_cfg.procname,        2),
            NUMFIELD("pid",              pid,                   7),
            STRFIELD("host",             g_cfg.hostname,        2),
            STRFIELD("domain",           g_netinfo[fd].dnsName, 2),
            STRFIELD("unit",             "request",             1),
            FIELDEND
        };

        // Increment the DNS counter by one for each event
        event_t e = {"net.dns", 1, DELTA, fields};
        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:DNS:outSendEvent", -1, CFG_LOG_ERROR);
        }
        break;
    }

    default:
        scopeLog("ERROR: doNetMetric:metric type", -1, CFG_LOG_ERROR);
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

static void
doSetConnection(int sd, const struct sockaddr *addr, socklen_t len, enum control_type_t endp)
{
    if (!addr || (len <= 0)) {
        return;
    }
    
    // Should we check for at least the size of sockaddr_in?
    if ((getNetEntry(sd) != NULL) && addr && (len > 0)) {
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
    struct sockaddr_storage addr;
    socklen_t addrlen = sizeof(struct sockaddr_storage);
    
    if (getNetEntry(sockfd) == NULL) {
        return -1;
    }
    
    if (getsockname(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
        doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, LOCAL);
    }
    
    if (getpeername(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
        doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, REMOTE);
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
        
    scopeLog("doAddNewSock: adding socket", sockfd, CFG_LOG_DEBUG);
    if (getsockname(sockfd, &addr, &addrlen) != -1) {
        if ((addr.sa_family == AF_INET) || (addr.sa_family == AF_INET6)) {
            int type;
            socklen_t len = sizeof(socklen_t);
                
            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type);
            } else {
                // Really can't add the socket at this point
                scopeLog("ERROR: doRecv:getsockopt", sockfd, CFG_LOG_ERROR);
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
    char dnsName[MAX_HOSTNAME];

    if (getNetEntry(sd) == NULL) {
        return -1;
    }
    
    query = (struct dns_query_t *)pkt;
    if ((dname = (char *)&query->name) == NULL) {
        return -1;
    }

/*    
      An opcode appears to be represented in a query packet 
      in what we define as a queston type; q->qtype. 
      Based on the table below we want to only handle a type of 0.
      OpCode 	Name 	Reference 
      0	Query	[RFC1035]
      1	IQuery (Inverse Query, OBSOLETE)	[RFC3425]
      2	Status	[RFC1035]
      3	Unassigned	
      4	Notify	[RFC1996]
      5	Update	[RFC2136]
      6	DNS Stateful Operations (DSO)	[RFC8490]
      7-15	Unassigned	

      Note that these types are a subset of QTYPEs.
      The type appears to be represented in a query packet
      in what we define as a question class; q->qclass. 
      We think a class of 1-16 should be valid.
      NOTE: We have not seen/tested all of these class
      types. We have seen a 1 and a 12. 
      TYPE            value and meaning
      A               1 a host address
      NS              2 an authoritative name server
      MD              3 a mail destination (Obsolete - use MX)
      MF              4 a mail forwarder (Obsolete - use MX)
      CNAME           5 the canonical name for an alias
      SOA             6 marks the start of a zone of authority
      MB              7 a mailbox domain name (EXPERIMENTAL)
      MG              8 a mail group member (EXPERIMENTAL)
      MR              9 a mail rename domain name (EXPERIMENTAL)
      NULL            10 a null RR (EXPERIMENTAL)
      WKS             11 a well known service description
      PTR             12 a domain name pointer
      HINFO           13 host information
      MINFO           14 mailbox or mail list information
      MX              15 mail exchange
      TXT             16 text strings

      *** or on Linux ***
      The packet could be sent to a local name server hosted by 
      systemd. If it is, the remote IP should be 127.0.0.53. 
      We pick up these queries by interposing the various
      gethostbyname functions, including getaddinfo. 
      For reference, the format of the string sent to a local name server is 
      of the form:
      TTP/1.1\r\nHost: wttr.in\r\nUser-Agent: curl/7.64.0\r\nAccept: @/@\r\n\r\nert.` 

      *** or on macOS ***
      macOS provides a DNS service, a daemon process that acts as a
      local name server. We interpose the function DNSServiceQueryRecord
      in order to dig out the domain name. The DNS metric is created
      directly from that function interposition. 
*/
    q = (struct question *)(pkt + sizeof(struct dns_header) + strlen(dname));
    if ((q->qtype != 0) || ((q->qclass < 1) || (q->qclass > 16))) {
        return 0;
    }

    // We think we have a direct DNS request
    aname = dnsName;

    while (*dname != '\0') {
        // handle one label
        for (llen = (int)*dname++; llen > 0; llen--) {
            *aname++ = *dname++;
        }
        
        *aname++ = '.';
    }

    aname--;
    *aname = '\0';

    if (strncmp(aname, g_netinfo[sd].dnsName, strlen(aname)) == 0) {
        // Already sent this from an interposed function
        g_netinfo[sd].dnsSend = FALSE;
    } else {
        strncpy(g_netinfo[sd].dnsName, aname, strlen(aname));
        g_netinfo[sd].dnsSend = TRUE;
    }
    
    return 0;
}

static int
doRecv(int sockfd, ssize_t rc)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (g_netinfo[sockfd].fd != sockfd) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doNetMetric(NETRX, sockfd, EVENT_BASED, rc);
    }
    return 0;
}

static int
doSend(int sockfd, ssize_t rc)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (g_netinfo[sockfd].fd != sockfd) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doNetMetric(NETTX, sockfd, EVENT_BASED, rc);

        if (GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            doNetMetric(DNS, sockfd, EVENT_BASED, 0);
        }
    }
    return 0;
}

static void
doAccept(int sd, struct sockaddr *addr, socklen_t addrlen, char *func)
{

    scopeLog(func, sd, CFG_LOG_DEBUG);
    addSock(sd, SOCK_STREAM);
    
    if (getNetEntry(sd) != NULL) {
        g_netinfo[sd].listen = TRUE;
        g_netinfo[sd].accept = TRUE;
        atomicAdd(&g_ctrs.openPorts, 1);
        atomicAdd(&g_ctrs.TCPConnections, 1);
        atomicAdd(&g_ctrs.activeConnections, 1);
        doSetConnection(sd, addr, addrlen, REMOTE);
        doNetMetric(OPEN_PORTS, sd, EVENT_BASED, 0);
        doNetMetric(TCP_CONNECTIONS, sd, EVENT_BASED, 0);
        doNetMetric(ACTIVE_CONNECTIONS, sd, EVENT_BASED, 0);
        g_netinfo[sd].startTime = getTime();
    }
}

static void
doReset()
{
    g_thread.once = 0;
    g_thread.startTime = time(NULL) + g_thread.interval;
    memset(&g_ctrs, 0, sizeof(struct metric_counters_t));
}

static void
reportFD(pid_t pid, int fd)
{
    struct net_info_t *ninfo = getNetEntry(fd);
    if (ninfo) {
        if (ninfo->action & EVENT_TX) {
            doNetMetric(NETTX, fd, PERIODIC, 0);
        }
        if (ninfo->action & EVENT_RX) {
            doNetMetric(NETRX, fd, PERIODIC, 0);
        }
        ninfo->action = 0;
    }

    struct fs_info_t *finfo = getFSEntry(fd);
    if (finfo) {
        if (finfo->action & EVENT_FS) {
            if (finfo->totalDuration > 0) {
                doFSMetric(FS_DURATION, fd, PERIODIC, "read/write", 0, NULL);
            }
            if (finfo->readBytes > 0) {
                doFSMetric(FS_READ, fd, PERIODIC, "read", 0, NULL);
            }
            if (finfo->writeBytes > 0) {
                doFSMetric(FS_WRITE, fd, PERIODIC, "write", 0, NULL);
            }
            if (finfo->numSeek > 0) {
                doFSMetric(FS_SEEK, fd, PERIODIC, "seek", 0, NULL);
            }
            if (finfo->numStat > 0) {
                doFSMetric(FS_STAT, fd, PERIODIC, "stat", 0, NULL);
            }
        }
        finfo->action = 0;
    }
}

static void *
periodic(void *arg)
{
    long mem;
    int i, nthread, nfds, children;
    pid_t pid = getpid();
    long long cpu, cpuState = 0;

    while (1) {
        // We report CPU time for this period.
        cpu = doGetProcCPU();
        doProcMetric(PROC_CPU, cpu - cpuState);
        cpuState = cpu;
        
        mem = osGetProcMemory(pid);
        doProcMetric(PROC_MEM, mem);

        nthread = osGetNumThreads(pid);
        doProcMetric(PROC_THREAD, nthread);

        nfds = osGetNumFds(pid);
        doProcMetric(PROC_FD, nfds);

        children = osGetNumChildProcs(pid);
        doProcMetric(PROC_CHILD, children);

        // report totals (not by file descriptor/socket descriptor)
        if (g_ctrs.readBytes > 0)  doTotal(TOT_READ, pid);
        if (g_ctrs.writeBytes > 0) doTotal(TOT_WRITE, pid);
        if (g_ctrs.netrxBytes > 0) doTotal(TOT_RX, pid);
        if (g_ctrs.nettxBytes > 0) doTotal(TOT_TX, pid);

        // report net and file by descriptor
        for (i = 0; i < MAX(g_cfg.numNinfo, g_cfg.numFSInfo); i++) {
            reportFD(pid, i);
        }
        
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
    g_fn.open = dlsym(RTLD_NEXT, "open");
    g_fn.openat = dlsym(RTLD_NEXT, "openat");
    g_fn.fopen = dlsym(RTLD_NEXT, "fopen");
    g_fn.freopen = dlsym(RTLD_NEXT, "freopen");
    g_fn.creat = dlsym(RTLD_NEXT, "creat");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.fclose = dlsym(RTLD_NEXT, "fclose");
    g_fn.fcloseall = dlsym(RTLD_NEXT, "fcloseall");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.pread = dlsym(RTLD_NEXT, "pread");
    g_fn.readv = dlsym(RTLD_NEXT, "readv");
    g_fn.fread = dlsym(RTLD_NEXT, "fread");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.pwrite = dlsym(RTLD_NEXT, "pwrite");
    g_fn.writev = dlsym(RTLD_NEXT, "writev");
    g_fn.fwrite = dlsym(RTLD_NEXT, "fwrite");
    g_fn.lseek = dlsym(RTLD_NEXT, "lseek");
    g_fn.fseeko = dlsym(RTLD_NEXT, "fseeko");
    g_fn.ftell = dlsym(RTLD_NEXT, "ftell");
    g_fn.ftello = dlsym(RTLD_NEXT, "ftello");
    g_fn.fgetpos = dlsym(RTLD_NEXT, "fgetpos");
    g_fn.fsetpos = dlsym(RTLD_NEXT, "fsetpos");
    g_fn.stat = dlsym(RTLD_NEXT, "stat");
    g_fn.lstat = dlsym(RTLD_NEXT, "lstat");
    g_fn.fstat = dlsym(RTLD_NEXT, "fstat");
    g_fn.statfs = dlsym(RTLD_NEXT, "statfs");
    g_fn.fstatfs = dlsym(RTLD_NEXT, "fstatfs");
    g_fn.statvfs = dlsym(RTLD_NEXT, "statvfs");
    g_fn.fstatvfs = dlsym(RTLD_NEXT, "fstatvfs");
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
    g_fn.gethostbyname = dlsym(RTLD_NEXT, "gethostbyname");
    g_fn.gethostbyname2 = dlsym(RTLD_NEXT, "gethostbyname2");
    g_fn.getaddrinfo = dlsym(RTLD_NEXT, "getaddrinfo");

#ifdef __MACOS__
    g_fn.close$NOCANCEL = dlsym(RTLD_NEXT, "close$NOCANCEL");
    g_fn.close_nocancel = dlsym(RTLD_NEXT, "close_nocancel");
    g_fn.guarded_close_np = dlsym(RTLD_NEXT, "guarded_close_np");
    g_fn.accept$NOCANCEL = dlsym(RTLD_NEXT, "accept$NOCANCEL");
    g_fn.__sendto_nocancel = dlsym(RTLD_NEXT, "__sendto_nocancel");
    g_fn.DNSServiceQueryRecord = dlsym(RTLD_NEXT, "DNSServiceQueryRecord");
    g_fn.fstatat = dlsym(RTLD_NEXT, "fstatat");
#endif // __MACOS__

#ifdef __LINUX__
    g_fn.open64 = dlsym(RTLD_NEXT, "open64");
    g_fn.openat64 = dlsym(RTLD_NEXT, "openat64");
    g_fn.fopen64 = dlsym(RTLD_NEXT, "fopen64");
    g_fn.freopen64 = dlsym(RTLD_NEXT, "freopen64");
    g_fn.creat64 = dlsym(RTLD_NEXT, "creat64");
    g_fn.pread64 = dlsym(RTLD_NEXT, "pread64");
    g_fn.preadv = dlsym(RTLD_NEXT, "preadv");
    g_fn.preadv2 = dlsym(RTLD_NEXT, "preadv2");
    g_fn.preadv64v2 = dlsym(RTLD_NEXT, "preadv64v2");
    g_fn.pwrite64 = dlsym(RTLD_NEXT, "pwrite64");
    g_fn.pwritev = dlsym(RTLD_NEXT, "pwritev");
    g_fn.pwritev2 = dlsym(RTLD_NEXT, "pwritev2");
    g_fn.pwritev64v2 = dlsym(RTLD_NEXT, "pwritev64v2");
    g_fn.lseek64 = dlsym(RTLD_NEXT, "lseek64");
    g_fn.fseeko64 = dlsym(RTLD_NEXT, "fseeko64");
    g_fn.ftello64 = dlsym(RTLD_NEXT, "ftello64");
    g_fn.statfs64 = dlsym(RTLD_NEXT, "statfs64");
    g_fn.fstatfs64 = dlsym(RTLD_NEXT, "fstatfs64");
    g_fn.gethostbyname_r = dlsym(RTLD_NEXT, "gethostbyname_r");
#ifdef __STATX__
    g_fn.statx = dlsym(RTLD_NEXT, "statx");
#endif // __STATX__
#endif // __LINUX__
    
    if ((g_netinfo = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    g_cfg.numNinfo = NET_ENTRIES;
    if (g_netinfo) memset(g_netinfo, 0, sizeof(struct net_info_t) * NET_ENTRIES);

    if ((g_fsinfo = (fs_info *)malloc(sizeof(struct fs_info_t) * FS_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    g_cfg.numFSInfo = FS_ENTRIES;
    if (g_fsinfo) memset(g_fsinfo, 0, sizeof(struct fs_info_t) * FS_ENTRIES);

    if (gethostname(g_cfg.hostname, sizeof(g_cfg.hostname)) != 0) {
        scopeLog("ERROR: Constructor:gethostname", -1, CFG_LOG_ERROR);
    }

    osGetProcname(g_cfg.procname, sizeof(g_cfg.procname));
    osInitTSC(&g_cfg);
    if (g_cfg.tsc_invariant == FALSE) {
        scopeLog("ERROR: TSC is not invariant", -1, CFG_LOG_ERROR);
    }
    

    {
        char* path = cfgPath(CFG_FILE_NAME);
        config_t* cfg = cfgRead(path);
        cfgProcessEnvironment(cfg);
        
        g_thread.interval = cfgOutPeriod(cfg);
        g_thread.startTime = time(NULL) + g_thread.interval;
        g_cfg.verbosity = cfgOutVerbosity(cfg);

        log_t* log = initLog(cfg);
        g_out = initOut(cfg, log);
        g_log = log; // Set after initOut to avoid infinite loop with socket
        if (path) free(path);
        cfgDestroy(&cfg);
    }

    scopeLog("Constructor", -1, CFG_LOG_INFO);
}

static void
doClose(int fd, const char *func)
{
    struct net_info_t *ninfo;
    struct fs_info_t *fsinfo;

    // report everything before the info is lost
    reportFD(getpid(), fd);

    if ((ninfo = getNetEntry(fd)) != NULL) {

        if (ninfo->listen == TRUE) {
            // Gauge tracking number of open ports
            atomicSub(&g_ctrs.openPorts, 1);
            doNetMetric(OPEN_PORTS, fd, EVENT_BASED, 0);
        }

        if (ninfo->accept == TRUE) {
            // Gauge tracking number of active TCP connections
            atomicSub(&g_ctrs.TCPConnections, 1);
            doNetMetric(TCP_CONNECTIONS, fd, EVENT_BASED, 0);

            if (ninfo->startTime != 0) {
                // Duration is in NS, the metric wants to be in MS
                ninfo->duration = getDuration(ninfo->startTime)  / 1000000;
                doNetMetric(CONNECTION_DURATION, fd, EVENT_BASED, 0);
            }
        }

        memset(ninfo, 0, sizeof(struct net_info_t));
        if (func) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: network", func);
            scopeLog(buf, fd, CFG_LOG_DEBUG);
        }
    }

    // Check both file desriptor tables
    if ((fsinfo = getFSEntry(fd)) != NULL) {
        if (func) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: file", func);
            scopeLog(buf, fd, CFG_LOG_TRACE);
        }
        
        doFSMetric(FS_CLOSE, fd, EVENT_BASED, func, 0, NULL);
        memset(fsinfo, 0, sizeof(struct fs_info_t));
    }
}

static void
doOpen(int fd, const char *path, enum fs_type_t type, const char *func)
{
    if (checkFSEntry(fd) == TRUE) {
        if (g_fsinfo[fd].fd == fd) {
            char buf[128];

            snprintf(buf, sizeof(buf), "%s:doOpen: duplicate(%d)", func, fd);
            scopeLog(buf, fd, CFG_LOG_DEBUG);
            DBG(NULL);
            doClose(fd, func);
        }
        
        if ((fd > g_cfg.numFSInfo) && (fd < MAX_FDS))  {
            int increase;
            fs_info *temp;

            if (fd < (MAX_FDS / 2)) {
                increase = MAX_FDS / 2;
            } else {
                increase = MAX_FDS;
            }

            // Need to realloc
            if ((temp = realloc(g_fsinfo, sizeof(struct fs_info_t) * increase)) == NULL) {
                scopeLog("ERROR: doOpen:realloc", fd, CFG_LOG_ERROR);
                DBG("re-alloc on FS table failed");
            } else {
                memset(&temp[g_cfg.numFSInfo], 0, sizeof(struct fs_info_t) * (increase - g_cfg.numFSInfo));
                g_fsinfo = temp;
                g_cfg.numFSInfo = increase;
            }
        }

        memset(&g_fsinfo[fd], 0, sizeof(struct fs_info_t));
        g_fsinfo[fd].fd = fd;
        g_fsinfo[fd].type = type;
        strncpy(g_fsinfo[fd].path, path, sizeof(g_fsinfo[fd].path));
        doFSMetric(FS_OPEN, fd, EVENT_BASED, func, 0, NULL);
        scopeLog(func, fd, CFG_LOG_TRACE);
    }
}

EXPORTON int
doDupFile(int newfd, int oldfd, const char *func)
{
    if ((newfd > g_cfg.numFSInfo) || (oldfd > g_cfg.numFSInfo)) {
        return -1;
    }

    doOpen(newfd, g_fsinfo[oldfd].path, g_fsinfo[oldfd].type, func);
    return 0;
}

EXPORTON int
doDupSock(int oldfd, int newfd)
{
    if ((newfd > g_cfg.numFSInfo) || (oldfd > g_cfg.numFSInfo)) {
        return -1;
    }

    bcopy(&g_netinfo[newfd], &g_netinfo[oldfd], sizeof(struct fs_info_t));
    g_netinfo[newfd].fd = newfd;
    g_netinfo[newfd].numTX = 0;
    g_netinfo[newfd].numRX = 0;
    g_netinfo[newfd].txBytes = 0;
    g_netinfo[newfd].rxBytes = 0;
    g_netinfo[newfd].startTime = 0;
    g_netinfo[newfd].duration = 0;

    return 0;
}

EXPORTON int
open(const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;
    
    WRAP_CHECK(open, -1);
    doThread(); // Will do nothing if a thread already exists
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.open(pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "open");
    }
    
    return fd;
}

EXPORTON int
openat(int dirfd, const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;
    
    WRAP_CHECK(openat, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.openat(dirfd, pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "openat");
    }
    
    return fd;
}

// Note: creat64 is defined to be obsolete
EXPORTON int
creat(const char *pathname, mode_t mode)
{
    int fd;
    
    WRAP_CHECK(creat, -1);
    doThread();
    fd = g_fn.creat(pathname, mode);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "creat");
    }
    
    return fd;
}

EXPORTON FILE *
fopen(const char *pathname, const char *mode)
{
    FILE *stream;
    
    WRAP_CHECK(fopen, NULL);
    doThread();
    stream = g_fn.fopen(pathname, mode);
    if (stream != NULL) {
        doOpen(fileno(stream), pathname, STREAM, "fopen");
    }
    
    return stream;
}

EXPORTON FILE *
freopen(const char *pathname, const char *mode, FILE *orig_stream)
{
    FILE *stream;
    
    WRAP_CHECK(freopen, NULL);
    doThread();
    stream = g_fn.freopen(pathname, mode, orig_stream);
    // freopen just changes the mode if pathname is null
    if ((stream != NULL) &&(pathname != NULL)) {
        doOpen(fileno(stream), pathname, STREAM, "freopen");
        doClose(fileno(orig_stream), "freopen");
    }
    
    return stream;
}

#ifdef __LINUX__
EXPORTON int
open64(const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;
    
    WRAP_CHECK(open64, -1);
    doThread(); // Will do nothing if a thread already exists
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.open64(pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "open64");
    }
    
    return fd;
}

EXPORTON int
openat64(int dirfd, const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;
    
    WRAP_CHECK(openat64, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.openat64(dirfd, pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "openat64");
    }
    
    return fd;
}

// Note: creat64 is defined to be obsolete
EXPORTON int
creat64(const char *pathname, mode_t mode)
{
    int fd;
    
    WRAP_CHECK(creat64, -1);
    doThread();
    fd = g_fn.creat64(pathname, mode);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "creat64");
    }
    
    return fd;
}

EXPORTON FILE *
fopen64(const char *pathname, const char *mode)
{
    FILE *stream;
    
    WRAP_CHECK(fopen64, NULL);
    doThread();
    stream = g_fn.fopen64(pathname, mode);
    if (stream != NULL) {
        doOpen(fileno(stream), pathname, STREAM, "fopen64");
    }
    
    return stream;
}

EXPORTON FILE *
freopen64(const char *pathname, const char *mode, FILE *orig_stream)
{
    FILE *stream;
    
    WRAP_CHECK(freopen64, NULL);
    doThread();
    stream = g_fn.freopen64(pathname, mode, orig_stream);
    // freopen just changes the mode if pathname is null
    if ((stream != NULL) && (pathname != NULL)) {
        doOpen(fileno(stream), pathname, STREAM, "freopen64");
        doClose(fileno(orig_stream), "freopen64");
    }
    
    return stream;
}

EXPORTON ssize_t
pread64(int fd, void *buf, size_t count, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pread64, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pread64(fd, buf, count, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pread64", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd) != NULL) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pread64", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "pread64", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
preadv(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};

    WRAP_CHECK(preadv, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.preadv(fd, iov, iovcnt, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("preadv", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd) != NULL) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
preadv2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(preadv2, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.preadv2(fd, iov, iovcnt, offset, flags);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("preadv2", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv2", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv2", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
preadv64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(preadv64v2, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.preadv64v2(fd, iov, iovcnt, offset, flags);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("preadv64v2", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv64v2", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv64v2", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
pwrite64(int fd, const void *buf, size_t nbyte, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pwrite64, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pwrite64(fd, buf, nbyte, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pwrite64", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwrite64", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwrite64", rc, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pwritev, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pwritev(fd, iov, iovcnt, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pwritev", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev", rc, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pwritev2, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pwritev2(fd, iov, iovcnt, offset, flags);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pwritev2", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev2", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev2", rc, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pwritev64v2, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pwritev64v2(fd, iov, iovcnt, offset, flags);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pwritev64v2", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev64v2", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev64v2", rc, NULL);
        }
    }
    return rc;
}

EXPORTON off_t
lseek64(int fd, off_t offset, int whence)
{
    off_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    
    WRAP_CHECK(lseek64, -1);
    doThread();
    rc = g_fn.lseek64(fd, offset, whence);
    
    if (rc != -1) {
        scopeLog("lseek64", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "lseek64", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
fseeko64(FILE *stream, off_t offset, int whence)
{
    off_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fseeko64, -1);
    doThread();
    rc = g_fn.fseeko64(stream, offset, whence);
    
    if (rc != -1) {
        scopeLog("fseeko64", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fseeko64", 0, NULL);
        }
    }
    return rc;
}

EXPORTON off_t
ftello64(FILE *stream)
{
    off_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    
    WRAP_CHECK(ftello64, -1);
    doThread();
    rc = g_fn.ftello64(stream);
    
    if (rc != -1) {
        scopeLog("ftello64", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "ftello64", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
statfs64(const char *path, struct statfs64 *buf)
{
    int rc;

    WRAP_CHECK(statfs64, -1);
    doThread();
    rc = g_fn.statfs64(path, buf);

    if (rc != -1) {
        scopeLog("statfs64", -1, CFG_LOG_DEBUG);
        doFSMetric(FS_STAT, -1, EVENT_BASED, "statfs64", 0, path);
    }
    return rc;
}

EXPORTON int
fstatfs64(int fd, struct statfs64 *buf)
{
    int rc;

    WRAP_CHECK(fstatfs64, -1);
    doThread();
    rc = g_fn.fstatfs64(fd, buf);

    if (rc != -1) {
        scopeLog("fstatfs64", fd, CFG_LOG_DEBUG);
        if (checkFSEntry(fd)) {
            doFSMetric(FS_STAT, fd, EVENT_BASED, "fstatfs64", 0, NULL);
        }
    }
    return rc;
}

#ifdef __STATX__
EXPORTON int
statx(int dirfd, const char *pathname, int flags,
      unsigned int mask, struct statx *statxbuf)
{
    int rc;

    WRAP_CHECK(statx, -1);
    doThread();
    rc = g_fn.statx(dirfd, pathname, flags, mask, statxbuf);

    if (rc != -1) {
        scopeLog("statx", -1, CFG_LOG_DEBUG);
        doFSMetric(FS_STAT, -1, EVENT_BASED, "statx", 0, pathname);
    }
    return rc;
}
#endif // __STATX__

EXPORTON int
stat(const char *pathname, struct stat *statbuf)
{
    int rc;

    WRAP_CHECK(stat, -1);
    doThread();
    rc = g_fn.stat(pathname, statbuf);

    if (rc != -1) {
        scopeLog("stat", -1, CFG_LOG_DEBUG);
        doFSMetric(FS_STAT, -1, EVENT_BASED, "stat", 0, pathname);
    }
    return rc;
}

EXPORTON int
fstat(int fd, struct stat *statbuf)
{
    int rc;

    WRAP_CHECK(fstat, -1);
    doThread();
    rc = g_fn.fstat(fd, statbuf);

    if (rc != -1) {
        scopeLog("fstat", fd, CFG_LOG_DEBUG);
        if (checkFSEntry(fd)) {
            doFSMetric(FS_STAT, fd, EVENT_BASED, "fstat", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
lstat(const char *pathname, struct stat *statbuf)
{
    int rc;

    WRAP_CHECK(lstat, -1);
    doThread();
    rc = g_fn.lstat(pathname, statbuf);

    if (rc != -1) {
        scopeLog("lstat", -1, CFG_LOG_DEBUG);
            doFSMetric(FS_STAT, -1, EVENT_BASED, "lstat", 0, pathname);
    }
    return rc;
}

EXPORTON int
statfs(const char *path, struct statfs *buf)
{
    int rc;

    WRAP_CHECK(statfs, -1);
    doThread();
    rc = g_fn.statfs(path, buf);

    if (rc != -1) {
        scopeLog("statfs", -1, CFG_LOG_DEBUG);
        doFSMetric(FS_STAT, -1, EVENT_BASED, "statfs", 0, path);
    }
    return rc;
}

EXPORTON int
fstatfs(int fd, struct statfs *buf)
{
    int rc;

    WRAP_CHECK(fstatfs, -1);
    doThread();
    rc = g_fn.fstatfs(fd, buf);

    if (rc != -1) {
        scopeLog("fstatfs", fd, CFG_LOG_DEBUG);
        if (checkFSEntry(fd)) {
            doFSMetric(FS_STAT, fd, EVENT_BASED, "fstatfs", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
statvfs(const char *path, struct statvfs *buf)
{
    int rc;

    WRAP_CHECK(statvfs, -1);
    doThread();
    rc = g_fn.statvfs(path, buf);

    if (rc != -1) {
        scopeLog("statvfs", -1, CFG_LOG_DEBUG);
        doFSMetric(FS_STAT, -1, EVENT_BASED, "statvfs", 0, path);
    }
    return rc;
}

EXPORTON int
fstatvfs(int fd, struct statvfs *buf)
{
    int rc;

    WRAP_CHECK(fstatvfs, -1);
    doThread();
    rc = g_fn.fstatvfs(fd, buf);

    if (rc != -1) {
        scopeLog("fstatvfs", fd, CFG_LOG_DEBUG);
        if (checkFSEntry(fd)) {
            doFSMetric(FS_STAT, fd, EVENT_BASED, "fstatvfs", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
gethostbyname_r(const char *name, struct hostent *ret, char *buf, size_t buflen,
                struct hostent **result, int *h_errnop)
{
    int rc;

    WRAP_CHECK(gethostbyname_r, -1);
    rc = g_fn.gethostbyname_r(name, ret, buf, buflen, result, h_errnop);
    if ((rc == 0) && (result != NULL)) {
        scopeLog("gethostbyname_r", -1, CFG_LOG_DEBUG);
        doDNSMetricName(name);
    }

    return rc;
}

#endif // __LINUX__

EXPORTON int
close(int fd)
{
    int rc;
    
    WRAP_CHECK(close, -1);
    doThread(); // Will do nothing if a thread already exists

    rc = g_fn.close(fd);
    if (rc != -1) {
        doClose(fd, "close");
    }
    
    return rc;
}

EXPORTON int
fclose(FILE *stream)
{
    int rc;

    WRAP_CHECK(fclose, EOF);
    doThread(); // Will do nothing if a thread already exists

    rc = g_fn.fclose(stream);
    if (rc != EOF) {
        doClose(fileno(stream), "fclose");
    }
    
    return rc;
}

EXPORTON int
fcloseall(void)
{
    int rc;

    WRAP_CHECK(close, EOF);
    doThread(); // Will do nothing if a thread already exists

    rc = g_fn.fcloseall();
    if (rc != EOF) {
        if (g_fsinfo) {
            int i;
            for (i = 0; i < g_cfg.numFSInfo; i++) {
                if ((g_fsinfo[i].fd != 0) &&
                    (g_fsinfo[i].type == STREAM)) {
                    doClose(i, "fcloseall");
                }
            }
        }
    }
    
    return rc;
}

#ifdef __MACOS__
EXPORTON int
close$NOCANCEL(int fd)
{
    int rc;

    WRAP_CHECK(close$NOCANCEL, -1);
    doThread();
    rc = g_fn.close$NOCANCEL(fd);
    if (rc != -1) {
        doClose(fd, "close$NOCANCEL");
    }
    
    return rc;
}


EXPORTON int
guarded_close_np(int fd, void *guard)
{
    int rc;

    WRAP_CHECK(guarded_close_np, -1);
    doThread();
    rc = g_fn.guarded_close_np(fd, guard);
    if (rc != -1) {
        doClose(fd, "guarded_close_np");
    }
    
    return rc;
}

EXPORTOFF int
close_nocancel(int fd)
{
    int rc;

    WRAP_CHECK(close_nocancel, -1);
    rc = g_fn.close_nocancel(fd);
    if (rc != -1) {
        doClose(fd, "close_nocancel");
    }
    
    return rc;
}

EXPORTON int
accept$NOCANCEL(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept$NOCANCEL, -1);
    doThread();
    sd = g_fn.accept$NOCANCEL(sockfd, addr, addrlen);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept$NOCANCEL");
    }

    return sd;
}

EXPORTON ssize_t
__sendto_nocancel(int sockfd, const void *buf, size_t len, int flags,
                  const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;

    WRAP_CHECK(__sendto_nocancel, -1);
    doThread();
    rc = g_fn.__sendto_nocancel(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog("__sendto_nocancel", sockfd, CFG_LOG_TRACE);

        doSetAddrs(sockfd);

        if ((getNetEntry(sockfd) != NULL) &&
            GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
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

    WRAP_CHECK(DNSServiceQueryRecord, -1);
    rc = g_fn.DNSServiceQueryRecord(sdRef, flags, interfaceIndex, fullname,
                                    rrtype, rrclass, callback, context);
    if (rc == 0) {
        scopeLog("DNSServiceQueryRecord", -1, CFG_LOG_DEBUG);
        doDNSMetricName(fullname);
    }

    return rc;
}

EXPORTOFF int
fstatat(int fd, const char *path, struct stat *buf, int flag)
{
    int rc;

    WRAP_CHECK(fstatat, -1);
    doThread();
    rc = g_fn.fstatat(fd, path, buf, flag);

    if (rc != -1) {
        scopeLog("fstatat", fd, CFG_LOG_DEBUG);
        if (checkFSEntry(fd)) {
            //doFSMetric(FS_XXX, fd, NULL, EVENT_BASED, "fstatat");
        }
    }
    return rc;
}

#endif // __MACOS__

EXPORTON off_t
lseek(int fd, off_t offset, int whence)
{
    off_t rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(lseek, -1);
    doThread();
    rc = g_fn.lseek(fd, offset, whence);
    
    if (rc != -1) {
        scopeLog("lseek", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "lseek", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
fseeko(FILE *stream, off_t offset, int whence)
{
    off_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    
    WRAP_CHECK(fseeko, -1);
    doThread();
    rc = g_fn.fseeko(stream, offset, whence);
    
    if (rc != -1) {
        scopeLog("fseeko", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fseeko", 0, NULL);
        }
    }
    return rc;
}

EXPORTON long
ftell(FILE *stream)
{
    long rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(ftell, -1);
    doThread();
    rc = g_fn.ftell(stream);
    
    if (rc != -1) {
        scopeLog("ftell", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "ftell", 0, NULL);
        }
    }
    return rc;
}

EXPORTON off_t
ftello(FILE *stream)
{
    off_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    
    WRAP_CHECK(ftello, -1);
    doThread();
    rc = g_fn.ftello(stream);
    
    if (rc != -1) {
        scopeLog("ftello", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "ftello", 0, NULL);
        }
    }
    return rc;
}

EXPORTON void
rewind(FILE *stream)
{
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK_VOID(rewind);
    doThread();
    g_fn.rewind(stream);
    
    scopeLog("rewind", fd, CFG_LOG_DEBUG);
    if (fs) {
        doFSMetric(FS_SEEK, fd, EVENT_BASED, "rewind", 0, NULL);
    }
    return;
}

EXPORTON int
fsetpos(FILE *stream, const fpos_t *pos)
{
    int rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    
    WRAP_CHECK(fsetpos, -1);
    doThread();
    rc = g_fn.fsetpos(stream, pos);

    if (rc == 0) {
        scopeLog("fsetpos", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fsetpos", 0, NULL);
        }
    }
    return rc;
}

EXPORTON int
fgetpos(FILE *stream,  fpos_t *pos)
{
    int rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fgetpos, -1);
    doThread();
    rc = g_fn.fgetpos(stream, pos);

    if (rc == 0) {
        scopeLog("fgetpos", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fgetpos", 0, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
write(int fd, const void *buf, size_t count)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(write, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.write(fd, buf, count);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("write", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "write", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "write", rc, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
pwrite(int fd, const void *buf, size_t nbyte, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pwrite, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pwrite(fd, buf, nbyte, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pwrite", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwrite", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwrite", rc, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
writev(int fd, const struct iovec *iov, int iovcnt)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(writev, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.writev(fd, iov, iovcnt);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("writev", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd) != NULL) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "writev", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "writev", rc, NULL);
        }
    }
    return rc;
}

EXPORTON size_t
fwrite(const void *restrict ptr, size_t size, size_t nitems, FILE *restrict stream)
{
    size_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(fwrite, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.fwrite(ptr, size, nitems, stream);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != 0) {
        scopeLog("fwrite", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "fwrite", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "fwrite", rc*size, NULL);
        }
    }
    return rc;
}

EXPORTON ssize_t
read(int fd, void *buf, size_t count)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(read, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.read(fd, buf, count);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("read", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "read", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "read", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
readv(int fd, const struct iovec *iov, int iovcnt)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(readv, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.readv(fd, iov, iovcnt);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("readv", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "readv", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "readv", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON ssize_t
pread(int fd, void *buf, size_t count, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(pread, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.pread(fd, buf, count, offset);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != -1) {
        scopeLog("pread", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pread", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "pread", rc, NULL);
        }
    }
    
    return rc;
}

EXPORTON size_t
fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);
    elapsed_t time = {0};
    
    WRAP_CHECK(fread, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }
    
    rc = g_fn.fread(ptr, size, nmemb, stream);
    
    if (fs) {
        time.duration = getDuration(time.initial);
    }
    
    if (rc != 0) {
        scopeLog("fread", fd, CFG_LOG_TRACE);
        if (getNetEntry(fd)) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (g_fsinfo && (fd <= g_cfg.numFSInfo) && (g_fsinfo[fd].fd == fd)) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "fread", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "fread", rc*size, NULL);
        }
    }
    
    return rc;
}

EXPORTON int
fcntl(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                    fArgs.arg[2], fArgs.arg[3]);
    if ((rc != -1) && getNetEntry(fd) && (cmd == F_DUPFD)) {
        // This is a network descriptor
        scopeLog("fcntl", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }
    
    return rc;
}

EXPORTON int
fcntl64(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl64, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl64(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                      fArgs.arg[2], fArgs.arg[3]);
    if ((rc != -1) && getNetEntry(fd) && (cmd == F_DUPFD)) {
        // This is a network descriptor
        scopeLog("fcntl", rc, CFG_LOG_DEBUG);
        doAddNewSock(rc);
    }
    
    return rc;
}

EXPORTON int
dup(int fd)
{
    int rc;

    WRAP_CHECK(dup, -1);
    doThread();
    rc = g_fn.dup(fd);
    if (rc != -1) {
        if (getNetEntry(fd)) {
            // This is a network descriptor
            scopeLog("dup", rc, CFG_LOG_DEBUG);
            doDupSock(fd, rc);
        } else if(getFSEntry(fd)) {
            doDupFile(fd, rc, "dup");
        }
    }

    return rc;
}

EXPORTON int
dup2(int oldfd, int newfd)
{
    int rc;

    WRAP_CHECK(dup2, -1);
    doThread();
    rc = g_fn.dup2(oldfd, newfd);
    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog("dup2", rc, CFG_LOG_DEBUG);
        if (getNetEntry(oldfd)) {
            if (getNetEntry(newfd)) {
                doClose(newfd, "dup2");
            }

            doDupSock(oldfd, newfd);
        } else if (getFSEntry(oldfd)) {
            if (getFSEntry(newfd)) {
                doClose(newfd, "dup2");
            }
            doDupFile(oldfd, newfd, "dup2");
        }
    }

    return rc;
}

EXPORTON int
dup3(int oldfd, int newfd, int flags)
{
    int rc;

    WRAP_CHECK(dup3, -1);
    doThread();
    rc = g_fn.dup3(oldfd, newfd, flags);
    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog("dup3", rc, CFG_LOG_DEBUG);
        if (getNetEntry(oldfd)) {
            if (getNetEntry(newfd)) {
                doClose(newfd, "dup3");
            }

            doDupSock(oldfd, newfd);
        } else if (getFSEntry(oldfd)) {
            if (getFSEntry(newfd)) {
                doClose(newfd, "dup3");
            }
            doDupFile(oldfd, newfd, "dup3");
        }
    }

    return rc;
}

EXPORTOFF void
vsyslog(int priority, const char *format, va_list ap)
{
    WRAP_CHECK_VOID(vsyslog);
    doThread();
    scopeLog("vsyslog", -1, CFG_LOG_DEBUG);
    g_fn.vsyslog(priority, format, ap);
    return;
}

EXPORTON pid_t
fork()
{
    pid_t rc;

    WRAP_CHECK(fork, -1);
    doThread();
    scopeLog("fork", -1, CFG_LOG_DEBUG);
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

    WRAP_CHECK(socket, -1);
    doThread();
    sd = g_fn.socket(socket_family, socket_type, protocol);
    if (sd != -1) {
        struct net_info_t *net;
        
        scopeLog("socket", sd, CFG_LOG_DEBUG);
        addSock(sd, socket_type);
        
        if (((net  = getNetEntry(sd)) != NULL) &&
            ((socket_family == AF_INET) ||
             (socket_family == AF_INET6)) &&            
            (socket_type == SOCK_DGRAM)) {
            // Tracking number of open ports
            atomicAdd(&g_ctrs.openPorts, 1);
            
            /*
             * State used in close()
             * We define that a UDP socket represents an open 
             * port when created and is open until the socket is closed
             *
             * a UDP socket is open we say the port is open
             * a UDP socket is closed we say the port is closed
             */
            net->listen = TRUE;
            doNetMetric(OPEN_PORTS, sd, EVENT_BASED, 0);
        }
    }

    return sd;
}

EXPORTON int
shutdown(int sockfd, int how)
{
    int rc;

    WRAP_CHECK(shutdown, -1);
    doThread();
    rc = g_fn.shutdown(sockfd, how);
    if (rc != -1) {
        doClose(sockfd, "shutdown");
    }
    
    return rc;
}

EXPORTON int
listen(int sockfd, int backlog)
{
    int rc;
    struct net_info_t *net = getNetEntry(sockfd);
    
    WRAP_CHECK(listen, -1);
    doThread();
    rc = g_fn.listen(sockfd, backlog);
    if (rc != -1) {
        scopeLog("listen", sockfd, CFG_LOG_DEBUG);
        
        // Tracking number of open ports
        atomicAdd(&g_ctrs.openPorts, 1);
        
        if (net) {
            net->listen = TRUE;
            net-> accept = TRUE;
            doNetMetric(OPEN_PORTS, sockfd, EVENT_BASED, 0);

            if (net->type == SOCK_STREAM) {
                atomicAdd(&g_ctrs.TCPConnections, 1);
                net->accept = TRUE;                            
                doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED, 0);
            }
        }
    }
    
    return rc;
}

EXPORTON int
accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept, -1);
    doThread();
    sd = g_fn.accept(sockfd, addr, addrlen);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept");
    }

    return sd;
}

EXPORTON int
accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags)
{
    int sd;

    WRAP_CHECK(accept4, -1);
    doThread();
    sd = g_fn.accept4(sockfd, addr, addrlen, flags);
    if ((sd != -1) && addr && addrlen) {
        doAccept(sd, addr, *addrlen, "accept4");
    }

    return sd;
}

EXPORTON int
bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;

    WRAP_CHECK(bind, -1);
    doThread();
    rc = g_fn.bind(sockfd, addr, addrlen);
    if (rc != -1) { 
        doSetConnection(sockfd, addr, addrlen, LOCAL);
        scopeLog("bind", sockfd, CFG_LOG_DEBUG);
    }
    
    return rc;

}

EXPORTON int
connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;
    struct net_info_t *net = getNetEntry(sockfd);
    
    WRAP_CHECK(connect, -1);
    doThread();
    rc = g_fn.connect(sockfd, addr, addrlen);
    if ((rc != -1) && net) {
        doSetConnection(sockfd, addr, addrlen, REMOTE);
        net->accept = TRUE;
        atomicAdd(&g_ctrs.activeConnections, 1);
        doNetMetric(ACTIVE_CONNECTIONS, sockfd, EVENT_BASED, 0);

        if (net->type == SOCK_STREAM) {
            atomicAdd(&g_ctrs.TCPConnections, 1);
            doNetMetric(TCP_CONNECTIONS, sockfd, EVENT_BASED, 0);
        }

        // Start the duration timer
        net->startTime = getTime();
        scopeLog("connect", sockfd, CFG_LOG_DEBUG);
    }
    
    return rc;
}

EXPORTON ssize_t
send(int sockfd, const void *buf, size_t len, int flags)
{
    ssize_t rc;
    struct net_info_t *net = getNetEntry(sockfd);
    
    WRAP_CHECK(send, -1);
    doThread();
    rc = g_fn.send(sockfd, buf, len, flags);
    if (rc != -1) {
        scopeLog("send", sockfd, CFG_LOG_TRACE);
        if (net &&
            GET_PORT(sockfd, net->remoteConn.ss_family, REMOTE) == DNS_PORT) {
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
    struct net_info_t *net = getNetEntry(sockfd);
    
    WRAP_CHECK(sendto, -1);
    doThread();
    rc = g_fn.sendto(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog("sendto", sockfd, CFG_LOG_TRACE);
        doSetConnection(sockfd, dest_addr, addrlen, REMOTE);

        if (net &&
            GET_PORT(sockfd, net->remoteConn.ss_family, REMOTE) == DNS_PORT) {
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
    struct net_info_t *net = getNetEntry(sockfd);
    
    WRAP_CHECK(sendmsg, -1);
    doThread();
    rc = g_fn.sendmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog("sendmsg", sockfd, CFG_LOG_TRACE);

        // For UDP connections the msg is a remote addr
        if (net && msg && (net->type != SOCK_STREAM)) {
            if (msg->msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msg->msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (net && GET_PORT(sockfd, net->remoteConn.ss_family, REMOTE) == DNS_PORT) {
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

    WRAP_CHECK(recv, -1);
    doThread();
    scopeLog("recv", sockfd, CFG_LOG_TRACE);
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

    WRAP_CHECK(recvfrom, -1);
    doThread();
    scopeLog("recvfrom", sockfd, CFG_LOG_TRACE);
    rc = g_fn.recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    if (rc != -1) {
        if (getNetEntry(sockfd)) {
            // We missed an accept...most likely
            // Or.. we are a child proc that inherited a socket
            int type;
            socklen_t len = sizeof(socklen_t);
                
            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type);
            } else {
                // Really can't add the socket at this point
                scopeLog("ERROR: recvfrom:getsockopt", sockfd, CFG_LOG_ERROR);
            }
        }

        if (src_addr && addrlen) {
            doSetConnection(sockfd, src_addr, *addrlen, REMOTE);
        }

        doNetMetric(NETRX, sockfd, EVENT_BASED, rc);
    }
    return rc;
}

EXPORTON ssize_t
recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    ssize_t rc;
    
    WRAP_CHECK(recvmsg, -1);
    doThread();
    rc = g_fn.recvmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog("recvmsg", sockfd, CFG_LOG_TRACE);

        // For UDP connections the msg is a remote addr
        if (msg && getNetEntry(sockfd)) {
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

EXPORTON struct hostent *
gethostbyname(const char *name)
{
    struct hostent *rc;

    WRAP_CHECK(gethostbyname, NULL);
    rc = g_fn.gethostbyname(name);
    if (rc != NULL) {
        scopeLog("gethostbyname", -1, CFG_LOG_DEBUG);
        doDNSMetricName(name);
    }

    return rc;
}

EXPORTON struct hostent *
gethostbyname2(const char *name, int af)
{
    struct hostent *rc;

    WRAP_CHECK(gethostbyname2, NULL);
    rc = g_fn.gethostbyname2(name, af);
    if (rc != NULL) {
        scopeLog("gethostbyname2", -1, CFG_LOG_DEBUG);
        doDNSMetricName(name);
    }

    return rc;
}

EXPORTON int
getaddrinfo(const char *node, const char *service,
            const struct addrinfo *hints,
            struct addrinfo **res)
{
    int rc;

    WRAP_CHECK(getaddrinfo, -1);
    rc = g_fn.getaddrinfo(node, service, hints, res);
    if (rc == 0) {
        scopeLog("getaddrinfo", -1, CFG_LOG_DEBUG);
        doDNSMetricName(node);
    }

    return rc;
}
