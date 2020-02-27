#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include "atomic.h"
#include "com.h"
#include "count.h"
#include "dbg.h"
#include "dns.h"
#include "format.h"
#include "os.h"
#include "plattime.h"

#define PROTOCOL_STR 16
#define SCOPE_UNIX 99
#define NET_ENTRIES 1024
#define FS_ENTRIES 1024

#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif

typedef struct metric_counters_t {
    uint64_t openPorts;
    uint64_t netConnectionsUdp;
    uint64_t netConnectionsTcp;
    uint64_t netConnectionsOther;
    uint64_t netrxBytes;
    uint64_t nettxBytes;
    uint64_t readBytes;
    uint64_t writeBytes;
    uint64_t numSeek;
    uint64_t numStat;
    uint64_t numOpen;
    uint64_t numClose;
    uint64_t numDNS;
    uint64_t fsDurationNum;
    uint64_t fsDurationTotal;
    uint64_t connDurationNum;
    uint64_t connDurationTotal;
    uint64_t dnsDurationNum;
    uint64_t dnsDurationTotal;
    uint64_t netConnectErrors;
    uint64_t netTxRxErrors;
    uint64_t netDNSErrors;
    uint64_t fsOpenCloseErrors;
    uint64_t fsRdWrErrors;
    uint64_t fsStatErrors;
} metric_counters;

typedef struct {
    struct {
        int open_close;
        int read_write;
        int stat;
        int seek;
        int error;
    } fs;
    struct {
        int open_close;
        int rx_tx;
        int dns;
        int error;
        int dnserror;
    } net;
} summary_t;

typedef struct net_info_t {
    int active;
    int type;
    bool urlRedirect;
    uint64_t numTX;
    uint64_t numRX;
    uint64_t txBytes;
    uint64_t rxBytes;
    bool dnsSend;
    uint64_t startTime;
    uint64_t numDuration;
    uint64_t totalDuration;
    uint64_t uid;
    uint64_t lnode;
    uint64_t rnode;
    char dnsName[MAX_HOSTNAME];
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
} net_info;

typedef struct fs_info_t {
    int active;
    fs_type_t type;
    uint64_t numOpen;
    uint64_t numClose;
    uint64_t numSeek;
    uint64_t numRead;
    uint64_t numWrite;
    uint64_t readBytes;
    uint64_t writeBytes;
    uint64_t numDuration;
    uint64_t totalDuration;
    uint64_t uid;
    char path[PATH_MAX];
} fs_info;

typedef struct {
    int numNinfo;
    int numFSInfo;
} counter_cfg_type;

static counter_cfg_type g_cfg;
static summary_t g_summary = {0};
static net_info *g_netinfo;
static fs_info *g_fsinfo;
static metric_counters g_ctrs = {0};
static int g_interval = DEFAULT_SUMMARY_PERIOD;


// interfaces
log_t* g_log = NULL;
out_t* g_out = NULL;
evt_t* g_evt = NULL;
ctl_t* g_ctl = NULL;
proc_id_t g_proc = {0};

// operational params (not config-based)
int g_urls = 0;
#define REDIRECTURL "fluentd"
#define OVERURL "<!DOCTYPE html>\r\n<html>\r\n<head>\r\n<meta http-equiv=\"refresh\" content=\"3; URL='http://cribl.io'\" />\r\n</head>\r\n<body>\r\n<h1>Welcome to Cribl!</h1>\r\n</body>\r\n</html>\r\n\r\n"
int g_blockconn = DEFAULT_PORTBLOCK;

#define DATA_FIELD(val)         STRFIELD("data",           (val),        1)
#define UNIT_FIELD(val)         STRFIELD("unit",           (val),        1)
#define CLASS_FIELD(val)        STRFIELD("class",          (val),        2)
#define PROTO_FIELD(val)        STRFIELD("proto",          (val),        2)
#define OP_FIELD(val)           STRFIELD("op",             (val),        3)
#define HOST_FIELD(val)         STRFIELD("host",           (val),        4)
#define PROC_FIELD(val)         STRFIELD("proc",           (val),        4)
#define DOMAIN_FIELD(val)       STRFIELD("domain",         (val),        5)
#define FILE_FIELD(val)         STRFIELD("file",           (val),        5)
#define LOCALIP_FIELD(val)      STRFIELD("localip",        (val),        6)
#define REMOTEIP_FIELD(val)     STRFIELD("remoteip",       (val),        6)
#define LOCALP_FIELD(val)       NUMFIELD("localp",         (val),        6)
#define LOCALN_FIELD(val)       NUMFIELD("localn",         (val),        6)
#define PORT_FIELD(val)         NUMFIELD("port",           (val),        6)
#define REMOTEP_FIELD(val)      NUMFIELD("remotep",        (val),        6)
#define REMOTEN_FIELD(val)      NUMFIELD("remoten",        (val),        6)
#define FD_FIELD(val)           NUMFIELD("fd",             (val),        7)
#define PID_FIELD(val)          NUMFIELD("pid",            (val),        7)
#define ARGS_FIELD(val)         STRFIELD("args",           (val),        7)
#define DURATION_FIELD(val)     NUMFIELD("duration",       (val),        8)
#define NUMOPS_FIELD(val)       NUMFIELD("numops",         (val),        8)

int
get_port(int fd, int type, control_type_t which) {
    in_port_t port;
    switch (type) {
    case AF_INET:
        if (which == LOCAL) {
            port = ((struct sockaddr_in *)&g_netinfo[fd].localConn)->sin_port;
        } else {
            port = ((struct sockaddr_in *)&g_netinfo[fd].remoteConn)->sin_port;
        }
        break;
    case AF_INET6:
        if (which == LOCAL) {
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].localConn)->sin6_port;
        } else {
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].remoteConn)->sin6_port;
        }
        break;
    default:
        port = (in_port_t)0;
        break;
    }
    return htons(port);
}

void
initCount()
{
    net_info *netinfoLocal;
    fs_info *fsinfoLocal;
    if ((netinfoLocal = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    g_cfg.numNinfo = NET_ENTRIES;
    if (netinfoLocal) memset(netinfoLocal, 0, sizeof(struct net_info_t) * NET_ENTRIES);

    // Per a Read Update & Change (RUC) model; now that the object is ready assign the global
    g_netinfo = netinfoLocal;

    if ((fsinfoLocal = (fs_info *)malloc(sizeof(struct fs_info_t) * FS_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    g_cfg.numFSInfo = FS_ENTRIES;
    if (fsinfoLocal) memset(fsinfoLocal, 0, sizeof(struct fs_info_t) * FS_ENTRIES);

    // Per RUC...
    g_fsinfo = fsinfoLocal;
}

void
resetCount()
{
    memset(&g_ctrs, 0, sizeof(struct metric_counters_t));
}

void
setCountInterval(int interval)
{
    g_interval = interval;
}

void
sendProcessStartMetric()
{
    char* urlEncodedCmd = fmtUrlEncode(g_proc.cmd);
    event_field_t fields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        HOST_FIELD(g_proc.hostname),
        ARGS_FIELD(urlEncodedCmd),
        UNIT_FIELD("process"),
        FIELDEND
    };
    event_t evt = INT_EVENT("proc.start", 1, DELTA, fields);
    outSendEvent(g_out, &evt);
    if (urlEncodedCmd) free(urlEncodedCmd);
}

void
scopeLog(const char* msg, int fd, cfg_log_level_t level)
{
    if (!g_log || !msg || !g_proc.procname[0]) return;

    char buf[strlen(msg) + 128];
    if (fd != -1) {
        snprintf(buf, sizeof(buf), "Scope: %s(pid:%d): fd:%d %s\n", g_proc.procname, g_proc.pid, fd, msg);
    } else {
        snprintf(buf, sizeof(buf), "Scope: %s(pid:%d): %s\n", g_proc.procname, g_proc.pid, msg);
    }
    logSend(g_log, buf, level);
}

static void
doMetric(evt_t* gev, const char *host, uint64_t uid, event_t *metric)
{
    // get a cJSON object for the given metric
    cJSON *json = msgEvtMetric(gev, metric, uid, &g_proc);

    // create cmd json and then output
    cmdPostEvtMsg(g_ctl, json);
}

static void
doEventLog(evt_t *gev, fs_info *fs, const void *buf, size_t len)
{
    // get a cJSON object for the given log msg
    cJSON *json = msgEvtLog(gev, fs->path, buf, len, fs->uid, &g_proc);

    // create cmd json and then output
    cmdPostEvtMsg(g_ctl, json);
}

static void
sendEvent(out_t* out, event_t* e)
{
    doMetric(g_evt, g_proc.hostname, getTime(), e);

    if (outSendEvent(out, e) == -1) {
        scopeLog("ERROR: doProcMetric:CPU:outSendEvent", -1, CFG_LOG_ERROR);
    }
}

// DEBUG
#if 0
static void
dumpAddrs(int sd, control_type_t endp)
{
    in_port_t port;
    char ip[INET6_ADDRSTRLEN];
    char buf[1024];

    inet_ntop(AF_INET,
              &((struct sockaddr_in *)&g_netinfo[sd].localConn)->sin_addr,
              ip, sizeof(ip));
    port = get_port(sd, g_netinfo[sd].localConn.ss_family, LOCAL);
    snprintf(buf, sizeof(buf), "%s:%d LOCAL: %s:%d", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);

    inet_ntop(AF_INET,
              &((struct sockaddr_in *)&g_netinfo[sd].remoteConn)->sin_addr,
              ip, sizeof(ip));
    port = get_port(sd, g_netinfo[sd].remoteConn.ss_family, REMOTE);
    snprintf(buf, sizeof(buf), "%s:%d REMOTE:%s:%d", __FUNCTION__, __LINE__, ip, port);
    scopeLog(buf, sd, CFG_LOG_DEBUG);

    if (get_port(sd, g_netinfo[sd].localConn.ss_family, REMOTE) == DNS_PORT) {
        scopeLog("DNS", sd, CFG_LOG_DEBUG);
    }
}
#endif

void
setVerbosity(unsigned verbosity)
{
    summary_t *summarize = &g_summary;

    summarize->fs.error =       (verbosity < 5);
    summarize->fs.open_close =  (verbosity < 6);
    summarize->fs.stat =        (verbosity < 7);
    summarize->fs.seek =        (verbosity < 8);
    summarize->fs.read_write =  (verbosity < 9);

    summarize->net.error =      (verbosity < 5);
    summarize->net.dnserror =   (verbosity < 5);
    summarize->net.dns =        (verbosity < 6);
    summarize->net.open_close = (verbosity < 7);
    summarize->net.rx_tx =      (verbosity < 9);
}

static bool
checkNetEntry(int fd)
{
    if (g_netinfo && (fd >= 0) && (fd < g_cfg.numNinfo)) {
        return TRUE;
    }

    return FALSE;
}

static bool
checkFSEntry(int fd)
{
    if (g_fsinfo && (fd >= 0) && (fd < g_cfg.numFSInfo)) {
        return TRUE;
    }

    return FALSE;
}

static net_info *
getNetEntry(int fd)
{
    if (g_netinfo && (fd >= 0) && (fd < g_cfg.numNinfo) &&
        g_netinfo[fd].active) {
        return &g_netinfo[fd];
    }
    return NULL;
}

static fs_info *
getFSEntry(int fd)
{
    if (g_fsinfo && (fd >= 0) && (fd < g_cfg.numFSInfo) &&
        g_fsinfo[fd].active) {
        return &g_fsinfo[fd];
    }

    const char* name;
    const char* description;
    if (g_fsinfo && ((fd >= 0 ) && (fd <= 2))) {
        switch(fd) {
            case 0:
                name = "stdin";
                description = "console input";
                break;
            case 1:
                name = "stdout";
                description = "console output";
                break;
            case 2:
                name = "stderr";
                description = "console output";
                break;
            default:
                DBG(NULL);
                return NULL;
        }

        doOpen(fd, name, FD, description);

        return &g_fsinfo[fd];
    }

    return NULL;
}

void
addSock(int fd, int type)
{
    if (checkNetEntry(fd) == TRUE) {
        if (g_netinfo[fd].active) {

            doClose(fd, "close: DuplicateSocket");

        }
/*
 * We need to do this realloc.
 * However, it needs to be done in such a way as to not
 * free the previous object that may be in use by a thread.
 * Possibly not use realloc. Leaving the code in place and this
 * comment as a reminder.
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
*/
        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].active = TRUE;
        g_netinfo[fd].type = type;
        g_netinfo[fd].uid = getTime();
#ifdef __LINUX__
        // Clear these bits so comparisons of type will work
        g_netinfo[fd].type &= ~SOCK_CLOEXEC;
        g_netinfo[fd].type &= ~SOCK_NONBLOCK;
#endif // __LINUX__
    }
}

int
doBlockConnection(int fd, const struct sockaddr *addr_arg)
{
    in_port_t port;

    if (g_blockconn == DEFAULT_PORTBLOCK) return 0;

    // We expect addr_arg to be supplied for connect() calls
    // and expect it to be NULL for accept() calls.  When it's
    // null, we will use addressing from the local side of the
    // accept fd.
    const struct sockaddr* addr;
    if (addr_arg) {
        addr = addr_arg;
    } else if (checkNetEntry(fd)) {
        addr = (struct sockaddr*)&g_netinfo[fd].localConn;
    } else {
        return 0;
    }

    if (addr->sa_family == AF_INET) {
        port = ((struct sockaddr_in *)addr)->sin_port;
    } else if (addr->sa_family == AF_INET6) {
        port = ((struct sockaddr_in6 *)addr)->sin6_port;
    } else {
        return 0;
    }

    if (g_blockconn == htons(port)) {
        scopeLog("doBlockConnection: blocked connection", fd, CFG_LOG_INFO);
        return 1;
    }

    return 0;
}

void
doErrorMetric(metric_t type, control_type_t source,
              const char *func, const char *name)
{
    if (!func || !name) return;

    const char err_name[] = "EFAULT";
    if (errno == EFAULT) {
        name = err_name;
    }

    switch (type) {
    case NET_ERR_CONN:
    case NET_ERR_RX_TX:
    {
        uint64_t* value = NULL;
        const char* class = "UNKNOWN";
        switch (type) {
            case NET_ERR_CONN:
                value = &g_ctrs.netConnectErrors;
                class = "connection";
                break;
            case NET_ERR_RX_TX:
                value = &g_ctrs.netTxRxErrors;
                class = "rx/tx";
                break;
            default:
                DBG(NULL);
                return;
        }

        if (source == EVENT_BASED) {
            atomicAddU64(value, 1);
        }

        // Don't report zeros.
        if (*value == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(func),
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t netErrMetric = INT_EVENT("net.error", *value, DELTA, fields);

        doMetric(g_evt, g_proc.hostname, getTime(), &netErrMetric);

        // Only report if enabled
        if ((g_summary.net.error) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &netErrMetric)) {
            scopeLog("ERROR: doErrorMetric:NET:outSendEvent", -1, CFG_LOG_ERROR);
        }

        atomicSwapU64(value, 0);
        break;
    }

    case FS_ERR_OPEN_CLOSE:
    case FS_ERR_READ_WRITE:
    case FS_ERR_STAT:
    case NET_ERR_DNS:
    {

        const char* metric = NULL;
        uint64_t* value = NULL;
        const char* class = "UNKNOWN";
        int* summarize = NULL;
        event_field_t file_field = FILE_FIELD(name);
        event_field_t domain_field = DOMAIN_FIELD(name);
        event_field_t* name_field;
        switch (type) {
            case FS_ERR_OPEN_CLOSE:
                metric = "fs.error";
                value = &g_ctrs.fsOpenCloseErrors;
                class = "open/close";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_READ_WRITE:
                metric = "fs.error";
                value = &g_ctrs.fsRdWrErrors;
                class = "read/write";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_STAT:
                metric = "fs.error";
                value = &g_ctrs.fsStatErrors;
                class = "stat";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case NET_ERR_DNS:
                metric = "net.error";
                value = &g_ctrs.netDNSErrors;
                class = "dns";
                summarize = &g_summary.net.dnserror;
                name_field = &domain_field;
                break;
            default:
                DBG(NULL);
                return;
        }

        if (source == EVENT_BASED) {
            atomicAddU64(value, 1);
        }

        // Don't report zeros.
        if (*value == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(func),
            *name_field,
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t fsErrMetric = INT_EVENT(metric, *value, DELTA, fields);

        doMetric(g_evt, g_proc.hostname, getTime(), &fsErrMetric);

        // Only report if enabled
        if (*summarize && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &fsErrMetric)) {
            scopeLog("ERROR: doErrorMetric:FS_ERR:outSendEvent", -1, CFG_LOG_ERROR);
        }

        atomicSwapU64(value, 0);
        break;
    }

    default:
        scopeLog("ERROR: doErrorMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doDNSMetricName(metric_t type, const char *domain, uint64_t duration)
{
    if (!domain) return;

    switch (type) {
    case DNS:
    {
        atomicAddU64(&g_ctrs.numDNS, 1);

        // Don't report zeros.
        if (g_ctrs.numDNS == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            DOMAIN_FIELD(domain),
            DURATION_FIELD(duration / 1000000), // convert ns to ms.
            UNIT_FIELD("request"),
            FIELDEND
        };

        event_t dnsMetric = INT_EVENT("net.dns", g_ctrs.numDNS, DELTA, fields);

        doMetric(g_evt, g_proc.hostname, getTime(), &dnsMetric);

        // Only report if enabled
        if (g_summary.net.dns) {
            return;
        }

        if (outSendEvent(g_out, &dnsMetric)) {
            scopeLog("ERROR: doDNSMetricName:DNS:outSendEvent", -1, CFG_LOG_ERROR);
        }
        atomicSwapU64(&g_ctrs.numDNS, 0);
        break;
    }

    case DNS_DURATION:
    {
        atomicAddU64(&g_ctrs.dnsDurationNum, 1);
        atomicAddU64(&g_ctrs.dnsDurationTotal, duration);

        uint64_t dur = 0ULL;
        int cachedDurationNum = g_ctrs.dnsDurationNum; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000000 converts ns to ms.
            dur = g_ctrs.dnsDurationTotal / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (dur == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            DOMAIN_FIELD(domain),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };

        event_t dnsDurMetric = INT_EVENT("net.dns.duration", dur, DELTA_MS, fields);

        doMetric(g_evt, g_proc.hostname, getTime(), &dnsDurMetric);

        // Only report if enabled
        if (g_summary.net.dns) {
            return;
        }

        if (outSendEvent(g_out, &dnsDurMetric)) {
            scopeLog("ERROR: doDNSMetricName:DNS_DURATION:outSendEvent", -1, CFG_LOG_ERROR);
        }
        atomicSwapU64(&g_ctrs.dnsDurationNum, 0);
        atomicSwapU64(&g_ctrs.dnsDurationTotal, 0);
        break;
    }

    default:
        scopeLog("ERROR: doDNSMetric:metric type", -1, CFG_LOG_ERROR);
    }

}

void
doProcMetric(metric_t type, long long measurement)
{
    switch (type) {
    case PROC_CPU:
    {
        {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                HOST_FIELD(g_proc.hostname),
                UNIT_FIELD("microsecond"),
                FIELDEND
            };
            event_t e = INT_EVENT("proc.cpu", measurement, DELTA, fields);
            sendEvent(g_out, &e);
        }

        // Avoid div by zero
        unsigned interval = g_interval;
        if (!interval) break;

        {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                HOST_FIELD(g_proc.hostname),
                UNIT_FIELD("percent"),
                FIELDEND
            };
            // convert measurement to double and scale to percent
            // convert interval from seconds to microseconds
            //
            // TBD: switch from using the configured to a measured interval
            double val = measurement * 100.0 / (interval*1000000.0);
            event_t e = FLT_EVENT("proc.cpu_perc", val, CURRENT, fields);
            sendEvent(g_out, &e);
        }
        break;
    }

    case PROC_MEM:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD("kibibyte"),
            FIELDEND
        };
        event_t e = INT_EVENT("proc.mem", measurement, DELTA, fields);
        sendEvent(g_out, &e);
        break;
    }

    case PROC_THREAD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD("thread"),
            FIELDEND
        };
        event_t e = INT_EVENT("proc.thread", measurement, CURRENT, fields);
        sendEvent(g_out, &e);
        break;
    }

    case PROC_FD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD("file"),
            FIELDEND
        };
        event_t e = INT_EVENT("proc.fd", measurement, CURRENT, fields);
        sendEvent(g_out, &e);
        break;
    }

    case PROC_CHILD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD("process"),
            FIELDEND
        };
        event_t e = INT_EVENT("proc.child", measurement, CURRENT, fields);
        sendEvent(g_out, &e);
        break;
    }

    default:
        scopeLog("ERROR: doProcMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doStatMetric(const char *op, const char *pathname)
{

    atomicAddU64(&g_ctrs.numStat, 1);

    event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(pathname),
            UNIT_FIELD("operation"),
            FIELDEND
    };

    event_t e = INT_EVENT("fs.op.stat", 1, DELTA, fields);
    doMetric(g_evt, g_proc.hostname, getTime(), &e);

    // Only report if enabled
    if (g_summary.fs.stat) {
        return;
    }

    if (outSendEvent(g_out, &e)) {
        scopeLog("doStatMetric", -1, CFG_LOG_ERROR);
    }

    //atomicSwapU64(&g_ctrs.numStat, 0);
}

void
doFSMetric(metric_t type, int fd, control_type_t source,
           const char *op, ssize_t size, const char *pathname)
{
    fs_info *fs;

    if ((fs = getFSEntry(fd)) == NULL) {
        return;
    }


    switch (type) {
    case FS_DURATION:
    {
        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            atomicAddU64(&g_fsinfo[fd].numDuration, 1);
            atomicAddU64(&g_fsinfo[fd].totalDuration, size);
            atomicAddU64(&g_ctrs.fsDurationNum, 1);
            atomicAddU64(&g_ctrs.fsDurationTotal, size);
        }

        uint64_t d = 0ULL;
        int cachedDurationNum = g_fsinfo[fd].numDuration; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000 converts ns to us.
            d = g_fsinfo[fd].totalDuration / ( 1000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (d == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("microsecond"),
            FIELDEND
        };
        event_t e = INT_EVENT("fs.duration", d, HISTOGRAM, fields);
        doMetric(g_evt, g_proc.hostname, g_fsinfo[fd].uid, &e);

        // Only report if enabled
        if ((g_summary.fs.read_write) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doFSMetric:FS_DURATION:outSendEvent", fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(&g_fsinfo[fd].numDuration, 0);
        atomicSwapU64(&g_fsinfo[fd].totalDuration, 0);
        atomicSwapU64(&g_ctrs.fsDurationNum, 0);
        atomicSwapU64(&g_ctrs.fsDurationTotal, 0);
        break;
    }

    case FS_READ:
    case FS_WRITE:
    {
        const char* metric = "UNKNOWN";
        uint64_t* numops = NULL;
        uint64_t* sizebytes = NULL;
        uint64_t* global_counter = NULL;
        const char* err_str = "UNKNOWN";
        switch (type) {
            case FS_READ:
                metric = "fs.read";
                numops = &g_fsinfo[fd].numRead;
                sizebytes = &g_fsinfo[fd].readBytes;
                global_counter = &g_ctrs.readBytes;
                err_str = "ERROR: doFSMetric:FS_READ:outSendEvent";
                break;
            case FS_WRITE:
                metric = "fs.write";
                numops = &g_fsinfo[fd].numWrite;
                sizebytes = &g_fsinfo[fd].writeBytes;
                global_counter = &g_ctrs.writeBytes;
                err_str = "ERROR: doFSMetric:FS_WRITE:outSendEvent";
                break;
            default:
                DBG(NULL);
                return;
        }

        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            atomicAddU64(numops, 1);
            atomicAddU64(sizebytes, size);
            atomicAddU64(global_counter, size); // not by fd
        }

        /*
         * Don't report zeros.
         * I think doing this here is right, even if we are
         * doing events, we don't want to report 0's.
         */
        if (*sizebytes == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            NUMOPS_FIELD(*numops),
            UNIT_FIELD("byte"),
            FIELDEND
        };

        event_t rwMetric = INT_EVENT(metric, *sizebytes, HISTOGRAM, fields);

        doMetric(g_evt, g_proc.hostname, g_fsinfo[fd].uid, &rwMetric);

        // Only report if enabled
        if ((g_summary.fs.read_write) && (source == EVENT_BASED)) {
            return;
        }


        if (outSendEvent(g_out, &rwMetric)) {
            scopeLog(err_str, fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(numops, 0);
        atomicSwapU64(sizebytes, 0);
        //atomicSwapU64(global_counter, 0);

        break;
    }

    case FS_OPEN:
    case FS_CLOSE:
    case FS_SEEK:
    {
        const char* metric = "UNKNOWN";
        uint64_t* numops = NULL;
        uint64_t* global_counter = NULL;
        int* summarize = NULL;
        const char* err_str = "UNKNOWN";
        switch (type) {
            case FS_OPEN:
                metric = "fs.op.open";
                numops = &g_fsinfo[fd].numOpen;
                global_counter = &g_ctrs.numOpen;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_OPEN:outSendEvent";
                break;
            case FS_CLOSE:
                metric = "fs.op.close";
                numops = &g_fsinfo[fd].numClose;
                global_counter = &g_ctrs.numClose;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_CLOSE:outSendEvent";
                break;
            case FS_SEEK:
                metric = "fs.op.seek";
                numops = &g_fsinfo[fd].numSeek;
                global_counter = &g_ctrs.numSeek;
                summarize = &g_summary.fs.seek;
                err_str = "ERROR: doFSMetric:FS_SEEK:outSendEvent";
                break;
            default:
                DBG(NULL);
                return;
        }

        // if called from an event, we update counters
        if (source == EVENT_BASED) {
            atomicAddU64(numops, 1);
            atomicAddU64(global_counter, 1);
        }

        // Don't report zeros.
        if (*numops == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t e = INT_EVENT(metric, *numops, DELTA, fields);
        doMetric(g_evt, g_proc.hostname, g_fsinfo[fd].uid, &e);

        // Only report if enabled
        if ((source == EVENT_BASED) && *summarize) {
            return;
        }

        if (outSendEvent(g_out, &e)) {
            scopeLog(err_str, fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(numops, 0);
        break;
    }

    default:
        DBG(NULL);
        scopeLog("ERROR: doFSMetric:metric type", fd, CFG_LOG_ERROR);
    }
}

void
doTotal(metric_t type)
{
    const char* metric = "UNKNOWN";
    uint64_t* value = NULL;
    const char* err_str = "UNKNOWN";
    const char* units = "byte";
    data_type_t aggregation_type = DELTA;
    switch (type) {
        case TOT_READ:
            metric = "fs.read";
            value = &g_ctrs.readBytes;
            err_str = "ERROR: doTotal:TOT_READ:outSendEvent";
            break;
        case TOT_WRITE:
            metric = "fs.write";
            value = &g_ctrs.writeBytes;
            err_str = "ERROR: doTotal:TOT_WRITE:outSendEvent";
            break;
        case TOT_RX:
            metric = "net.rx";
            value = &g_ctrs.netrxBytes;
            err_str = "ERROR: doTotal:TOT_RX:outSendEvent";
            break;
        case TOT_TX:
            metric = "net.tx";
            value = &g_ctrs.nettxBytes;
            err_str = "ERROR: doTotal:TOT_TX:outSendEvent";
            break;
        case TOT_SEEK:
            metric = "fs.seek";
            value = &g_ctrs.numSeek;
            err_str = "ERROR: doTotal:TOT_SEEK:outSendEvent";
            units = "operation";
            break;
        case TOT_STAT:
            metric = "fs.stat";
            value = &g_ctrs.numStat;
            err_str = "ERROR: doTotal:TOT_STAT:outSendEvent";
            units = "operation";
            break;
        case TOT_OPEN:
            metric = "fs.open";
            value = &g_ctrs.numOpen;
            err_str = "ERROR: doTotal:TOT_OPEN:outSendEvent";
            units = "operation";
            break;
        case TOT_CLOSE:
            metric = "fs.close";
            value = &g_ctrs.numClose;
            err_str = "ERROR: doTotal:TOT_CLOSE:outSendEvent";
            units = "operation";
            break;
        case TOT_DNS:
            metric = "net.dns";
            value = &g_ctrs.numDNS;
            err_str = "ERROR: doTotal:TOT_DNS:outSendEvent";
            units = "operation";
            break;
        case TOT_PORTS:
            metric = "net.port";
            value = &g_ctrs.openPorts;
            err_str = "ERROR: doTotal:TOT_PORTS:outSendEvent";
            units = "instance";
            aggregation_type = CURRENT;
            break;
        case TOT_TCP_CONN:
            metric = "net.tcp";
            value = &g_ctrs.netConnectionsTcp;
            err_str = "ERROR: doTotal:TOT_TCP_CONN:outSendEvent";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_UDP_CONN:
            metric = "net.udp";
            value = &g_ctrs.netConnectionsUdp;
            err_str = "ERROR: doTotal:TOT_UDP_CONN:outSendEvent";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_OTHER_CONN:
            metric = "net.other";
            value = &g_ctrs.netConnectionsOther;
            err_str = "ERROR: doTotal:TOT_OTHER_CONN:outSendEvent";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        default:
            DBG(NULL);
            return;
    }

    // Don't report zeros.
    if (*value == 0) return;

    event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD(units),
            CLASS_FIELD("summary"),
            FIELDEND
    };
    event_t e = INT_EVENT(metric, *value, aggregation_type, fields);
    if (outSendEvent(g_out, &e)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report (if it's not a gauge)
    if (aggregation_type != CURRENT) atomicSwapU64(value, 0);
}

void
doTotalDuration(metric_t type)
{
    const char* metric = "UNKNOWN";
    uint64_t* value = NULL;
    uint64_t* num = NULL;
    data_type_t aggregation_type = DELTA_MS;
    const char* units = "UNKNOWN";
    uint64_t factor = 1ULL;
    const char* err_str = "UNKNOWN";
    switch (type) {
        case TOT_FS_DURATION:
            metric = "fs.duration";
            value = &g_ctrs.fsDurationTotal;
            num = &g_ctrs.fsDurationNum;
            aggregation_type = HISTOGRAM;
            units = "microsecond";
            factor = 1000;
            err_str = "ERROR: doTotalDuration:TOT_FS_DURATION:outSendEvent";
            break;
        case TOT_NET_DURATION:
            metric = "net.conn_duration";
            value = &g_ctrs.connDurationTotal;
            num = &g_ctrs.connDurationNum;
            aggregation_type = DELTA_MS;
            units = "millisecond";
            factor = 1000000;
            err_str = "ERROR: doTotalDuration:TOT_NET_DURATION:outSendEvent";
            break;
        case TOT_DNS_DURATION:
            metric = "net.dns.duration";
            value = &g_ctrs.dnsDurationTotal;
            num = &g_ctrs.dnsDurationNum;
            aggregation_type = DELTA_MS;
            units = "millisecond";
            factor = 1000000;
            err_str = "ERROR: doTotalDuration:TOT_DNS_DURATION:outSendEvent";
            break;
        default:
            DBG(NULL);
            return;
    }

    uint64_t d = 0ULL;
    int cachedDurationNum = *num; // avoid div by zero
    if (cachedDurationNum >= 1) {
        // factor is there to scale from ns to the appropriate units
        d = *value / ( factor * cachedDurationNum);
    }

    // Don't report zeros.
    if (d == 0) return;

    event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD(units),
            CLASS_FIELD("summary"),
            FIELDEND
    };
    event_t e = INT_EVENT(metric, d, aggregation_type, fields);
    if (outSendEvent(g_out, &e)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report
    atomicSwapU64(value, 0);
    atomicSwapU64(num, 0);
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
doUnixEndpoint(int sd, net_info *net)
{
    ino_t rnode;
    struct stat sbuf;

    if (!net) return;

    if ((fstat(sd, &sbuf) == -1) ||
        ((sbuf.st_mode & S_IFMT) != S_IFSOCK)) {
        net->lnode = 0;
        net->rnode = 0;
        return;
    }

    if ((rnode = osUnixSockPeer(sbuf.st_ino)) != -1) {
        net->lnode = sbuf.st_ino;
        net->rnode = rnode;
    } else {
        net->lnode = 0;
        net->rnode = 0;
    }
    return;
}

void
doNetMetric(metric_t type, int fd, control_type_t source, ssize_t size)
{
    char proto[PROTOCOL_STR];
    in_port_t localPort, remotePort;

    if (getNetEntry(fd) == NULL) {
        return;
    }

    getProtocol(g_netinfo[fd].type, proto, sizeof(proto));
    localPort = get_port(fd, g_netinfo[fd].localConn.ss_family, LOCAL);
    remotePort = get_port(fd, g_netinfo[fd].remoteConn.ss_family, REMOTE);

    switch (type) {
    case OPEN_PORTS:
    case NET_CONNECTIONS:
    {
        const char* metric = "UNKNOWN";
        uint64_t* value = NULL;
        const char* units = "UNKNOWN";
        const char* err_str = "UNKNOWN";
        switch (type) {
        case OPEN_PORTS:
            metric = "net.port";
            value = &g_ctrs.openPorts;
            units = "instance";
            err_str = "ERROR: doNetMetric:OPEN_PORTS:outSendEvent";
            break;
        case NET_CONNECTIONS:
            if (g_netinfo[fd].type == SOCK_STREAM) {
                metric = "net.tcp";
                value = &g_ctrs.netConnectionsTcp;
            } else if (g_netinfo[fd].type == SOCK_DGRAM) {
                metric = "net.udp";
                value = &g_ctrs.netConnectionsUdp;
            } else {
                metric = "net.other";
                value = &g_ctrs.netConnectionsOther;
            }
            units = "connection";
            err_str = "ERROR: doNetMetric:NET_CONNECTIONS:outSendEvent";
            break;
        default:
            DBG(NULL);
            return;
        }

        // if called from an event, we update counters
        if ((source == EVENT_BASED) && size) {
            if (size < 0) {
               atomicSubU64(value, labs(size));
            } else {
               atomicAddU64(value, size);
            }
            if (!g_netinfo[fd].startTime)   g_netinfo[fd].startTime = getTime();
        }

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_proc.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            UNIT_FIELD(units),
            FIELDEND
        };
        event_t e = INT_EVENT(metric, *value, CURRENT, fields);
        doMetric(g_evt, g_proc.hostname, g_netinfo[fd].uid, &e);

        // Only report if enabled
        if ((g_summary.net.open_close) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &e)) {
            scopeLog(err_str, fd, CFG_LOG_ERROR);
        }

        // Don't reset the info if we tried to report.  It's a gauge.
        // atomicSwapU64(value, 0);

        break;
    }

    case CONNECTION_DURATION:
    {

        uint64_t new_duration = 0ULL;
        if (g_netinfo[fd].startTime != 0ULL) {
            new_duration = getDuration(g_netinfo[fd].startTime);
            g_netinfo[fd].startTime = 0ULL;
        }

        // if called from an event, we update counters
        if ((source == EVENT_BASED) && new_duration) {
            atomicAddU64(&g_netinfo[fd].numDuration, 1);
            atomicAddU64(&g_netinfo[fd].totalDuration, new_duration);
            atomicAddU64(&g_ctrs.connDurationNum, 1);
            atomicAddU64(&g_ctrs.connDurationTotal, new_duration);
        }

        uint64_t d = 0ULL;
        int cachedDurationNum = g_netinfo[fd].numDuration; // avoid div by zero
        if (cachedDurationNum >= 1 ) {
            // factor of 1000000 converts ns to ms.
            d = g_netinfo[fd].totalDuration / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (d == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_proc.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };
        event_t e = INT_EVENT("net.conn_duration", d, DELTA_MS, fields);
        doMetric(g_evt, g_proc.hostname, g_netinfo[fd].uid, &e);

        // Only report if enabled
        if ((g_summary.net.open_close) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &e)) {
            scopeLog("ERROR: doNetMetric:CONNECTION_DURATION:outSendEvent", fd, CFG_LOG_ERROR);
        }

        atomicSwapU64(&g_netinfo[fd].numDuration, 0);
        atomicSwapU64(&g_netinfo[fd].totalDuration, 0);
        atomicSwapU64(&g_ctrs.connDurationNum, 0);
        atomicSwapU64(&g_ctrs.connDurationTotal, 0);

        break;
    }

    case NETRX:
    {
        event_t rxMetric;
        event_field_t rxFields[20];
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        if (source == EVENT_BASED) {
            atomicAddU64(&g_netinfo[fd].numRX, 1);
            atomicAddU64(&g_netinfo[fd].rxBytes, size);
            atomicAddU64(&g_ctrs.netrxBytes, size);
        }

        // Don't report zeros.
        if (g_netinfo[fd].rxBytes == 0ULL) return;

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        // Do we need to define domain=LOCAL or NETLINK?
        if ((g_netinfo[fd].type == SCOPE_UNIX) ||
            (g_netinfo[fd].localConn.ss_family == AF_LOCAL) ||
            (g_netinfo[fd].localConn.ss_family == AF_NETLINK)) {
            doUnixEndpoint(fd, &g_netinfo[fd]);
            localPort = g_netinfo[fd].lnode;
            remotePort = g_netinfo[fd].rnode;

            if (g_netinfo[fd].localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("UNIX"),
                PROTO_FIELD(proto),
                LOCALN_FIELD(localPort),
                REMOTEN_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(g_netinfo[fd].numRX),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&rxFields, &fields, sizeof(fields));
            event_t rxUnixMetric = INT_EVENT("net.rx", g_netinfo[fd].rxBytes, DELTA, rxFields);
            memmove(&rxMetric, &rxUnixMetric, sizeof(event_t));
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
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("AF_INET"),
                PROTO_FIELD(proto),
                LOCALIP_FIELD(lip),
                LOCALP_FIELD(localPort),
                REMOTEIP_FIELD(rip),
                REMOTEP_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(g_netinfo[fd].numRX),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&rxFields, &fields, sizeof(fields));
            event_t rxNetMetric = INT_EVENT("net.rx", g_netinfo[fd].rxBytes, DELTA, rxFields);
            memmove(&rxMetric, &rxNetMetric, sizeof(event_t));
        }

        doMetric(g_evt, g_proc.hostname, g_netinfo[fd].uid, &rxMetric);

        if ((g_summary.net.rx_tx) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &rxMetric)) {
            scopeLog("ERROR: doNetMetric:NETRX:outSendEvent", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(&g_netinfo[fd].numRX, 0);
        atomicSwapU64(&g_netinfo[fd].rxBytes, 0);
        //atomicSwapU64(&g_ctrs.netrx, 0);

        break;
    }

    case NETTX:
    {
        event_t txMetric;
        event_field_t txFields[20];
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        if (source == EVENT_BASED) {
            atomicAddU64(&g_netinfo[fd].numTX, 1);
            atomicAddU64(&g_netinfo[fd].txBytes, size);
            atomicAddU64(&g_ctrs.nettxBytes, size);
        }

        // Don't report zeros.
        if (g_netinfo[fd].txBytes == 0ULL) return;

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if ((g_netinfo[fd].type == SCOPE_UNIX) ||
            (g_netinfo[fd].localConn.ss_family == AF_LOCAL) ||
            (g_netinfo[fd].localConn.ss_family == AF_NETLINK)) {
            doUnixEndpoint(fd, &g_netinfo[fd]);
            localPort = g_netinfo[fd].lnode;
            remotePort = g_netinfo[fd].rnode;

            if (g_netinfo[fd].localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("UNIX"),
                PROTO_FIELD(proto),
                LOCALN_FIELD(localPort),
                REMOTEN_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(g_netinfo[fd].numRX),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&txFields, &fields, sizeof(fields));
            event_t txUnixMetric = INT_EVENT("net.tx", g_netinfo[fd].txBytes, DELTA, txFields);
            memmove(&txMetric, &txUnixMetric, sizeof(event_t));
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

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("AF_INET"),
                PROTO_FIELD(proto),
                LOCALIP_FIELD(lip),
                LOCALP_FIELD(localPort),
                REMOTEIP_FIELD(rip),
                REMOTEP_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(g_netinfo[fd].numRX),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&txFields, &fields, sizeof(fields));
            event_t txNetMetric = INT_EVENT("net.tx", g_netinfo[fd].txBytes, DELTA, txFields);
            memmove(&txMetric, &txNetMetric, sizeof(event_t));
        }

        doMetric(g_evt, g_proc.hostname, g_netinfo[fd].uid, &txMetric);

        if ((g_summary.net.rx_tx) && (source == EVENT_BASED)) {
            return;
        }

        if (outSendEvent(g_out, &txMetric)) {
            scopeLog("ERROR: doNetMetric:NETTX:outSendEvent", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(&g_netinfo[fd].numTX, 0);
        atomicSwapU64(&g_netinfo[fd].txBytes, 0);
        //atomicSwapU64(&g_ctrs.nettx, 0);

        break;
    }

    case DNS:
    {
        if (g_netinfo[fd].dnsSend == FALSE) {
            break;
        }

        // For next time
        g_netinfo[fd].dnsSend = FALSE;

        // TBD - this is only called by doSend.  Consider calling this directly
        // from there?
        doDNSMetricName(DNS, g_netinfo[fd].dnsName, 0);

        break;
    }

    default:
        scopeLog("ERROR: doNetMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doSetConnection(int sd, const struct sockaddr *addr, socklen_t len, control_type_t endp)
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

int
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
int
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
                scopeLog("ERROR: doAddNewSock:getsockopt", sockfd, CFG_LOG_ERROR);
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

static int
isLegalLabelChar(char x)
{
    // Technically, RFC 1035 has more constraints than this... positionally.
    // Must start with a letter, end with a letter or digit, and can contain
    // a hyphen.  This is the check we can make if we don't care about position.

    // I'm not using isalnum because I don't want locale to affect it.
    if (x >= 'a' && x <= 'z') return 1;
    if (x >= 'A' && x <= 'Z') return 1;
    if (x >= '0' && x <= '9') return 1;
    if (x == '-') return 1;
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
 * See RFC 1035 for details.
 */

int
getDNSName(int sd, void *pkt, int pktlen)
{
    dns_query *query;
    struct dns_header *header;
    char *dname;
    char dnsName[MAX_HOSTNAME+1];
    int dnsNameBytesUsed = 0;

    if (getNetEntry(sd) == NULL) {
        return -1;
    }

    query = (struct dns_query_t *)pkt;
    header = &query->qhead;
    if ((dname = (char *)&query->name) == NULL) {
        return -1;
    }

    /*
      An opcode is represented in a DNS header and is defined foe every query packet
      We look for the opcode in the header to be a query.
      Based on the table below we want to only handle a type of 0.
      OpCode    Name    Reference
      0 Query   [RFC1035]
      1 IQuery (Inverse Query, OBSOLETE)        [RFC3425]
      2 Status  [RFC1035]
      3 Unassigned
      4 Notify  [RFC1996]
      5 Update  [RFC2136]
      6 DNS Stateful Operations (DSO)   [RFC8490]
      7-15      Unassigned

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

    if ((header->opcode != OPCODE_QUERY) && (header->qr != 1)) {
        return 0;
    }

    // We think we have a direct DNS request
    char* pkt_end = (char*)pkt + pktlen;

    while ((*dname != '\0') && (dname < pkt_end)) {
        // handle one label

        int label_len = (int)*dname++;
        if (label_len > 63) return -1; // labels must be 63 chars or less
        if (&dname[label_len] >= pkt_end) return -1; // honor packet end
        // Ensure we don't overrun the size of dnsName
        if ((dnsNameBytesUsed + label_len) >= sizeof(dnsName)) return -1;

        for ( ; (label_len > 0); label_len--) {
            if (!isLegalLabelChar(*dname)) return -1;
            dnsName[dnsNameBytesUsed++] = *dname++;
        }
        dnsName[dnsNameBytesUsed++] = '.';
    }

    dnsName[dnsNameBytesUsed-1] = '\0'; // overwrite the last period

    if (strncmp(dnsName, g_netinfo[sd].dnsName, dnsNameBytesUsed) == 0) {
        // Already sent this from an interposed function
        g_netinfo[sd].dnsSend = FALSE;
    } else {
        strncpy(g_netinfo[sd].dnsName, dnsName, dnsNameBytesUsed);
        g_netinfo[sd].dnsSend = TRUE;
    }

    return 0;
}

int
doURL(int sockfd, const void *buf, size_t len, metric_t src)
{
    if (g_urls == 0) return 0;

    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
    }

    if ((src == NETTX) && (strstr(buf, REDIRECTURL) != NULL)) {
        g_netinfo[sockfd].urlRedirect = TRUE;
        return 0;
    }

    if ((src == NETRX) && (g_netinfo[sockfd].urlRedirect == TRUE) &&
        (len >= strlen(OVERURL))) {
        g_netinfo[sockfd].urlRedirect = FALSE;
        // explicit vars as it's nice to have in the debugger
        //char *sbuf = (char *)buf;
        char *url = OVERURL;
        int urllen = strlen(url);
        strncpy((char *)buf, url, urllen);
        return urllen;
    }
    return 0;
}

int
doAccessRights(struct msghdr *msg)
{
    int *recvfd;
    struct cmsghdr *cmptr;
    struct stat sbuf;

    if (!msg) return -1;

    if (((cmptr = CMSG_FIRSTHDR(msg)) != NULL) &&
        (cmptr->cmsg_len >= CMSG_LEN(sizeof(int))) &&
        (cmptr->cmsg_level == SOL_SOCKET) &&
        (cmptr->cmsg_type == SCM_RIGHTS)) {
        // voila; we have a new fd
        int i, numfds;

        numfds = (cmptr->cmsg_len - CMSG_ALIGN(sizeof(struct cmsghdr))) / sizeof(int);
        if (numfds <= 0) return -1;
        recvfd = ((int *) CMSG_DATA(cmptr));

        for (i = 0; i < numfds; i++) {
            // file or socket?
            if (fstat(recvfd[i], &sbuf) != -1) {
                if ((sbuf.st_mode & S_IFMT) == S_IFSOCK) {
                    doAddNewSock(recvfd[i]);
                } else {
                    doOpen(recvfd[i], "Received_File_Descriptor", FD, "recvmsg");
                }
            } else {
                DBG("errno: %d", errno);
                return -1;
            }
        }
    }

    return 0;
}

int
doRecv(int sockfd, ssize_t rc)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doNetMetric(NETRX, sockfd, EVENT_BASED, rc);
    }
    return 0;
}

int
doSend(int sockfd, ssize_t rc)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doNetMetric(NETTX, sockfd, EVENT_BASED, rc);

        if (get_port(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            // tbd - consider calling doDNSMetricName instead...
            doNetMetric(DNS, sockfd, EVENT_BASED, 0);
        }
    }
    return 0;
}

void
doAccept(int sd, struct sockaddr *addr, socklen_t *addrlen, char *func)
{

    scopeLog(func, sd, CFG_LOG_DEBUG);
    addSock(sd, SOCK_STREAM);

    if (getNetEntry(sd) != NULL) {
        if (addr && addrlen) doSetConnection(sd, addr, *addrlen, REMOTE);
        doNetMetric(OPEN_PORTS, sd, EVENT_BASED, 1);
        doNetMetric(NET_CONNECTIONS, sd, EVENT_BASED, 1);
    }
}


//
// reportFD is called in two cases:
//   1) when a socket or file is being closed
//   2) during periodic reporting
void
reportFD(int fd, control_type_t source)
{
    struct net_info_t *ninfo = getNetEntry(fd);
    if (ninfo) {
        if (!g_summary.net.rx_tx) {
            doNetMetric(NETTX, fd, source, 0);
            doNetMetric(NETRX, fd, source, 0);
        }
        if (!g_summary.net.open_close) {
            doNetMetric(OPEN_PORTS, fd, source, 0);
            doNetMetric(NET_CONNECTIONS, fd, source, 0);
            doNetMetric(CONNECTION_DURATION, fd, source, 0);
        }
    }

    struct fs_info_t *finfo = getFSEntry(fd);
    if (finfo) {
        if (!g_summary.fs.read_write) {
            doFSMetric(FS_DURATION, fd, source, "read/write", 0, NULL);
            doFSMetric(FS_READ, fd, source, "read", 0, NULL);
            doFSMetric(FS_WRITE, fd, source, "write", 0, NULL);
        }
        if (!g_summary.fs.seek) {
            doFSMetric(FS_SEEK, fd, source, "seek", 0, NULL);
        }
    }
}

void
reportAllFds(control_type_t source)
{
    int i;
    for (i = 0; i < MAX(g_cfg.numNinfo, g_cfg.numFSInfo); i++) {
        reportFD(i, source);
    }
}

void
doRead(int fd, uint64_t initialTime, int success, ssize_t bytes, const char* func)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    if (success) {
        scopeLog(func, fd, CFG_LOG_TRACE);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, bytes);
        } else if (fs) {
            // Don't count data from stdin
            if ((fd > 2) || strncmp(fs->path, "std", 3)) {
                uint64_t duration = getDuration(initialTime);
                doFSMetric(FS_DURATION, fd, EVENT_BASED, func, duration, NULL);
                doFSMetric(FS_READ, fd, EVENT_BASED, func, bytes, NULL);
            }
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, func, fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, func, "nopath");
        }
    }
}

void
doWrite(int fd, uint64_t initialTime, int success, const void* buf, ssize_t bytes, const char* func)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    if (success) {
        scopeLog(func, fd, CFG_LOG_TRACE);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, bytes);
        } else if (fs) {
            // Don't count data from stdout, stderr
            if ((fd > 2) || strncmp(fs->path, "std", 3)) {
                uint64_t duration = getDuration(initialTime);
                doFSMetric(FS_DURATION, fd, EVENT_BASED, func, duration, NULL);
                doFSMetric(FS_WRITE, fd, EVENT_BASED, func, bytes, NULL);
            }
            doEventLog(g_evt, fs, buf, bytes);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, func, fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, func, "nopath");
        }
    }
}

void
doSeek(int fd, int success, const char* func)
{
    struct fs_info_t *fs = getFSEntry(fd);
    if (success) {
        scopeLog(func, fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, func, 0, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, func, fs->path);
        }
    }
}

#ifdef __LINUX__
void
doStatPath(const char* path, int rc, const char* func)
{
    if (rc != -1) {
        scopeLog(func, -1, CFG_LOG_DEBUG);
        doStatMetric(func, path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, func, path);
    }
}

void
doStatFd(int fd, int rc, const char* func)
{
    struct fs_info_t *fs = getFSEntry(fd);

    if (rc != -1) {
        scopeLog(func, fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric(func, fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, func, fs->path);
        }
    }
}
#endif // __LINUX__

int
doDupFile(int newfd, int oldfd, const char *func)
{
    if ((newfd >= g_cfg.numFSInfo) || (oldfd >= g_cfg.numFSInfo)) {
        return -1;
    }

    doOpen(newfd, g_fsinfo[oldfd].path, g_fsinfo[oldfd].type, func);
    return 0;
}

int
doDupSock(int oldfd, int newfd)
{
    if ((newfd >= g_cfg.numFSInfo) || (oldfd >= g_cfg.numFSInfo)) {
        return -1;
    }

    bcopy(&g_netinfo[newfd], &g_netinfo[oldfd], sizeof(struct fs_info_t));
    g_netinfo[newfd].active = TRUE;
    g_netinfo[newfd].numTX = 0;
    g_netinfo[newfd].numRX = 0;
    g_netinfo[newfd].txBytes = 0;
    g_netinfo[newfd].rxBytes = 0;
    g_netinfo[newfd].startTime = 0;
    g_netinfo[newfd].totalDuration = 0;
    g_netinfo[newfd].numDuration = 0;

    return 0;
}

void
doDup(int fd, int rc, const char* func, int copyNet)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
    if (rc != -1) {
        if (net) {
            // This is a network descriptor
            scopeLog(func, rc, CFG_LOG_DEBUG);
            if (copyNet) {
                doDupSock(fd, rc);
            } else {
                doAddNewSock(rc);
            }
        } else if (fs) {
            doDupFile(fd, rc, func);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, func, fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, func, "nopath");
        }
    }
}

void
doDup2(int oldfd, int newfd, int rc, const char* func)
{
    struct fs_info_t *fs = getFSEntry(oldfd);
    struct net_info_t *net = getNetEntry(oldfd);

    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog(func, rc, CFG_LOG_DEBUG);
        if (net) {
            if (getNetEntry(newfd)) {
                doClose(newfd, func);
            }
            doDupSock(oldfd, newfd);
        } else if (fs) {
            if (getFSEntry(newfd)) {
                doClose(newfd, func);
            }
            doDupFile(oldfd, newfd, func);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, func, fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, func, "nopath");
        }
    }
}

void
doClose(int fd, const char *func)
{
    struct net_info_t *ninfo;
    struct fs_info_t *fsinfo;

    if ((ninfo = getNetEntry(fd)) != NULL) {

        doNetMetric(OPEN_PORTS, fd, EVENT_BASED, -1);
        doNetMetric(NET_CONNECTIONS, fd, EVENT_BASED, -1);
        doNetMetric(CONNECTION_DURATION, fd, EVENT_BASED, -1);

        if (func) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: network", func);
            scopeLog(buf, fd, CFG_LOG_DEBUG);
        }
    }

    // Check both file desriptor tables
    if ((fsinfo = getFSEntry(fd)) != NULL) {

        doFSMetric(FS_CLOSE, fd, EVENT_BASED, func, 0, NULL);

        if (func) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%s: file", func);
            scopeLog(buf, fd, CFG_LOG_TRACE);
        }
    }

    // report everything before the info is lost
    reportFD(fd, EVENT_BASED);

    if (ninfo) memset(ninfo, 0, sizeof(struct net_info_t));
    if (fsinfo) memset(fsinfo, 0, sizeof(struct fs_info_t));

}

void
doOpen(int fd, const char *path, fs_type_t type, const char *func)
{
    if (checkFSEntry(fd) == TRUE) {
        if (g_fsinfo[fd].active) {
            char buf[128];

            snprintf(buf, sizeof(buf), "%s:doOpen: duplicate(%d)", func, fd);
            scopeLog(buf, fd, CFG_LOG_DEBUG);
            DBG(NULL);
            doClose(fd, func);
        }
/*
 * We need to do this realloc.
 * However, it needs to be done in such a way as to not
 * free the previous object that may be in use by a thread.
 * Possibly not use realloc. Leaving the code in place and this
 * comment as a reminder.

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
*/
        memset(&g_fsinfo[fd], 0, sizeof(struct fs_info_t));
        g_fsinfo[fd].active = TRUE;
        g_fsinfo[fd].type = type;
        g_fsinfo[fd].uid = getTime();
        strncpy(g_fsinfo[fd].path, path, sizeof(g_fsinfo[fd].path));

        doFSMetric(FS_OPEN, fd, EVENT_BASED, func, 0, NULL);
        scopeLog(func, fd, CFG_LOG_TRACE);
    }
}

void
doSendFile(int out_fd, int in_fd, uint64_t initialTime, int rc, const char* func)
{
    struct fs_info_t *fsrd = getFSEntry(in_fd);
    struct net_info_t *nettx = getNetEntry(out_fd);

    if (rc != -1) {
        scopeLog(func, in_fd, CFG_LOG_TRACE);
        if (nettx) {
            doSetAddrs(out_fd);
            doSend(out_fd, rc);
        }

        if (fsrd) {
            uint64_t duration = getDuration(initialTime);
            doFSMetric(FS_DURATION, in_fd, EVENT_BASED, func, duration, NULL);
            doFSMetric(FS_WRITE, in_fd, EVENT_BASED, func, rc, NULL);
        }
    } else {
        /*
         * We don't want to increment an error twice
         * We don't know which fd the error is associated with
         * We emit one metric with the input pathname
         */
        if (fsrd) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, func, fsrd->path);
        }

        if (nettx) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, func, "nopath");
        }
    }
}

void
doCloseAndReportFailures(int fd, int success, const char* func)
{
    struct fs_info_t *fs;
    if (success) {
        doClose(fd, func);
    } else {
        if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, func, fs->path);
        }
    }
}

void
doCloseAllStreams()
{
    if (!g_fsinfo) return;
    int i;
    for (i = 0; i < g_cfg.numFSInfo; i++) {
        if ((g_fsinfo[i].active) &&
            (g_fsinfo[i].type == STREAM)) {
            doClose(i, "fcloseall");
        }
    }
}

int
remotePortIsDNS(int sockfd)
{
    struct net_info_t *net = getNetEntry(sockfd);
    if (!net) return FALSE;

    return (get_port(sockfd, net->remoteConn.ss_family, REMOTE) == DNS_PORT);
}

int
sockIsTCP(int sockfd)
{
    struct net_info_t *net = getNetEntry(sockfd);
    if (!net) return FALSE;
    return (net->type == SOCK_STREAM);
}

