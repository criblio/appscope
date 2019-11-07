#define _GNU_SOURCE
#include <dlfcn.h>
#include "atomic.h"
#include "cfg.h"
#include "cfgutils.h"
#include "dbg.h"
#include "log.h"
#include "out.h"
#include "scopetypes.h"
#include "wrap.h"

interposed_funcs g_fn;
rtconfig g_cfg = {0};
static net_info *g_netinfo;
static fs_info *g_fsinfo;
static metric_counters g_ctrs = {0};
static thread_timing g_thread = {0};
static log_t* g_log = NULL;
static out_t* g_out = NULL;
static evt_t* g_evt = NULL;
static config_t *g_staticfg = NULL;
static log_t *g_prevlog = NULL;
static out_t *g_prevout = NULL;
static evt_t *g_prevevt = NULL;
__thread int g_getdelim = 0;

// Forward declaration
static void *periodic(void *);
static void doClose(int, const char *);
static void doConfig(config_t *);
static void doOpen(int, const char *, enum fs_type_t, const char *);

static void
scopeLog(const char* msg, int fd, cfg_log_level_t level)
{
    if (!g_log || !msg || !g_cfg.procname[0]) return;

    char buf[strlen(msg) + 128];
    if (fd != -1) {
        snprintf(buf, sizeof(buf), "Scope: %s(pid:%d): fd:%d %s\n", g_cfg.procname, g_cfg.pid, fd, msg);
    } else {
        snprintf(buf, sizeof(buf), "Scope: %s(pid:%d): %s\n", g_cfg.procname, g_cfg.pid, msg);
    }
    if (logSend(g_log, buf, level) == DEFAULT_BADFD) {
        // We lost our fd, re-open
        // should just do initLog, not everything
        doConfig(g_staticfg);
    }
}

static void
sendEvent(out_t* out, event_t* e)
{
    int rc;

    rc = outSendEvent(out, e);
    if (rc == DEFAULT_BADFD) {
        // We lost our fd, re-open
        // should just do initOut, not everything
        doConfig(g_staticfg);
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
#define PORT_FIELD(val)         NUMFIELD("port",           (val),        6)
#define REMOTEP_FIELD(val)      NUMFIELD("remotep",        (val),        6)
#define FD_FIELD(val)           NUMFIELD("fd",             (val),        7)
#define PID_FIELD(val)          NUMFIELD("pid",            (val),        7)
#define DURATION_FIELD(val)     NUMFIELD("duration",       (val),        8)
#define NUMOPS_FIELD(val)       NUMFIELD("numops",         (val),        8)

static void
setVerbosity(rtconfig* c, unsigned verbosity)
{
    if (!c) return;

    summary_t* summarize = &c->summarize;

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


static void
doConfig(config_t *cfg)
{
    // Save the current objects to get cleaned up on the periodic thread
    g_prevout = g_out;
    g_prevlog = g_log;
    g_prevevt = g_evt;

    g_thread.interval = cfgOutPeriod(cfg);
    if (!g_thread.startTime) {
        g_thread.startTime = time(NULL) + g_thread.interval;
    }
    setVerbosity(&g_cfg, cfgOutVerbosity(cfg));
    g_cfg.cmddir = cfgCmdDir(cfg);

    log_t* log = initLog(cfg);
    g_out = initOut(cfg);
    g_log = log; // Set after initOut to avoid infinite loop with socket
    g_evt = initEvt(cfg);
}

// Process dynamic config change if they are available
static int
dynConfig(void)
{
    FILE *fs;
    char path[PATH_MAX];

    snprintf(path, sizeof(path), "%s/%s.%d", g_cfg.cmddir, DYN_CONFIG_PREFIX, g_cfg.pid);

    // Is there a command file for this pid
    if (osIsFilePresent(g_cfg.pid, path) == -1) return 0;

    // Open the command file
    if ((fs = g_fn.fopen(path, "r")) == NULL) return -1;

    // Modify the static config from the command file
    cfgProcessCommands(g_staticfg, fs);

    // Apply the config
    doConfig(g_staticfg);

    g_fn.fclose(fs);
    unlink(path);
    return 0;
}

// Return the time delta from start to now in nanoseconds
static uint64_t
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

static int
doEventLog(evt_t *gev, fs_info *fs, const void *buf, size_t len)
{
    if (fs->event == TRUE) {
        return evtLog(gev, g_cfg.hostname, fs->path, buf, len, fs->uid);
    }
    return -1;
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

    if (((fd == 1) || (fd == 2)) && g_fsinfo &&
        (evtSource(g_evt, CFG_SRC_CONSOLE) != DEFAULT_SRC_CONSOLE)) {
        if (fd == 1) doOpen(fd, "stdout", FD, "console output");
        if (fd == 2) doOpen(fd, "stderr", FD, "console output");
        if (g_fsinfo[fd].fd == fd) {
            g_fsinfo[fd].event = TRUE;
            return &g_fsinfo[fd];
        }
    }

    return NULL;    
}

static void
addSock(int fd, int type)
{
    if (checkNetEntry(fd) == TRUE) {
        if (g_netinfo[fd].fd == fd) {

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
        g_netinfo[fd].fd = fd;
        g_netinfo[fd].type = type;
        g_netinfo[fd].uid = getTime();
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
doErrorMetric(enum metric_t type, enum control_type_t source,
              const char *func, const char *name)
{
    if (!func || !name) return;

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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(func),
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t netErrMetric = {"net.error", *value, DELTA, fields};

        evtMetric(g_evt, g_cfg.hostname, getTime(), &netErrMetric);

        // Only report if enabled
        if ((g_cfg.summarize.net.error) && (source == EVENT_BASED)) {
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
                summarize = &g_cfg.summarize.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_READ_WRITE:
                metric = "fs.error";
                value = &g_ctrs.fsRdWrErrors;
                class = "read/write";
                summarize = &g_cfg.summarize.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_STAT:
                metric = "fs.error";
                value = &g_ctrs.fsStatErrors;
                class = "stat";
                summarize = &g_cfg.summarize.fs.error;
                name_field = &file_field;
                break;
            case NET_ERR_DNS:
                metric = "net.error";
                value = &g_ctrs.netDNSErrors;
                class = "dns";
                summarize = &g_cfg.summarize.net.dnserror;
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(func),
            *name_field,
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t fsErrMetric = {metric, *value, DELTA, fields};

        evtMetric(g_evt, g_cfg.hostname, getTime(), &fsErrMetric);

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

static void
doDNSMetricName(enum metric_t type, const char *domain, uint64_t duration)
{
    if (!domain) return;

    switch (type) {
    case DNS:
    {
        atomicAddU64(&g_ctrs.numDNS, 1);

        // Don't report zeros.
        if (g_ctrs.numDNS == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            DOMAIN_FIELD(domain),
            DURATION_FIELD(duration / 1000000), // convert ns to ms.
            UNIT_FIELD("request"),
            FIELDEND
        };

        event_t dnsMetric = {"net.dns", g_ctrs.numDNS, DELTA, fields};

        evtMetric(g_evt, g_cfg.hostname, getTime(), &dnsMetric);

        // Only report if enabled
        if (g_cfg.summarize.net.dns) {
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            DOMAIN_FIELD(domain),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };

        event_t dnsDurMetric = {"net.dns.duration", dur, DELTA_MS, fields};

        evtMetric(g_evt, g_cfg.hostname, getTime(), &dnsDurMetric);

        // Only report if enabled
        if (g_cfg.summarize.net.dns) {
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

static void
doProcMetric(enum metric_t type, long long measurement)
{
    switch (type) {
    case PROC_CPU:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD("microsecond"),
            FIELDEND
        };
        event_t e = {"proc.cpu", measurement, DELTA, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_MEM:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD("kibibyte"),
            FIELDEND
        };
        event_t e = {"proc.mem", measurement, DELTA, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_THREAD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD("thread"),
            FIELDEND
        };
        event_t e = {"proc.thread", measurement, CURRENT, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_FD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD("file"),
            FIELDEND
        };
        event_t e = {"proc.fd", measurement, CURRENT, fields};
        sendEvent(g_out, &e);
        break;
    }

    case PROC_CHILD:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD("process"),
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
doStatMetric(const char *op, const char *pathname)
{

    atomicAddU64(&g_ctrs.numStat, 1);

    event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(op),
            FILE_FIELD(pathname),
            UNIT_FIELD("operation"),
            FIELDEND
    };


    // Only report if enabled
    if (g_cfg.summarize.fs.stat) {
        return;
    }

    event_t e = {"fs.op.stat", 1, DELTA, fields};
    if (outSendEvent(g_out, &e)) {
        scopeLog("doStatMetric", -1, CFG_LOG_ERROR);
    }

    //atomicSwapU64(&g_ctrs.numStat, 0);
}

static void
doFSMetric(enum metric_t type, int fd, enum control_type_t source,
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

        // Only report if enabled
        if ((g_cfg.summarize.fs.read_write) && (source == EVENT_BASED)) {
            return;
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("microsecond"),
            FIELDEND
        };
        event_t e = {"fs.duration", d, HISTOGRAM, fields};
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            NUMOPS_FIELD(*numops),
            UNIT_FIELD("byte"),
            FIELDEND
        };

        event_t rwMetric = {metric, *sizebytes, HISTOGRAM, fields};

        evtMetric(g_evt, g_cfg.hostname, g_fsinfo[fd].uid, &rwMetric);

        // Only report if enabled
        if ((g_cfg.summarize.fs.read_write) && (source == EVENT_BASED)) {
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
                summarize = &g_cfg.summarize.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_OPEN:outSendEvent";
                break;
            case FS_CLOSE:
                metric = "fs.op.close";
                numops = &g_fsinfo[fd].numClose;
                global_counter = &g_ctrs.numClose;
                summarize = &g_cfg.summarize.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_CLOSE:outSendEvent";
                break;
            case FS_SEEK:
                metric = "fs.op.seek";
                numops = &g_fsinfo[fd].numSeek;
                global_counter = &g_ctrs.numSeek;
                summarize = &g_cfg.summarize.fs.seek;
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

        // Only report if enabled
        if ((source == EVENT_BASED) && *summarize) {
            return;
        }

        // Don't report zeros.
        if (*numops == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            OP_FIELD(op),
            FILE_FIELD(g_fsinfo[fd].path),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t e = {metric, *numops, DELTA, fields};
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


static void
doTotal(enum metric_t type)
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD(units),
            CLASS_FIELD("summary"),
            FIELDEND
    };
    event_t e = {metric, *value, aggregation_type, fields};
    if (outSendEvent(g_out, &e)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report (if it's not a gauge)
    if (aggregation_type != CURRENT) atomicSwapU64(value, 0);
}


static void
doTotalDuration(enum metric_t type)
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            HOST_FIELD(g_cfg.hostname),
            UNIT_FIELD(units),
            CLASS_FIELD("summary"),
            FIELDEND
    };
    event_t e = {metric, d, aggregation_type, fields};
    if (outSendEvent(g_out, &e)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report
    atomicSwapU64(value, 0);
    atomicSwapU64(num, 0);
}


static void
doNetMetric(enum metric_t type, int fd, enum control_type_t source, ssize_t size)
{
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

        // Only report if enabled
        if ((g_cfg.summarize.net.open_close) && (source == EVENT_BASED)) {
            return;
        }

        event_field_t fields[] = {
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            UNIT_FIELD(units),
            FIELDEND
        };
        event_t e = {metric, *value, CURRENT, fields};
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

        // Only report if enabled
        if ((g_cfg.summarize.net.open_close) && (source == EVENT_BASED)) {
            return;
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };
        event_t e = {"net.conn_duration", d, DELTA_MS, fields};
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
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

        event_t rxMetric = {"net.rx", g_netinfo[fd].rxBytes, DELTA, fields};

        evtMetric(g_evt, g_cfg.hostname, g_netinfo[fd].uid, &rxMetric);

        if ((g_cfg.summarize.net.rx_tx) && (source == EVENT_BASED)) {
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
            PROC_FIELD(g_cfg.procname),
            PID_FIELD(g_cfg.pid),
            FD_FIELD(fd),
            HOST_FIELD(g_cfg.hostname),
            PROTO_FIELD(proto),
            LOCALIP_FIELD(lip),
            LOCALP_FIELD(localPort),
            REMOTEIP_FIELD(rip),
            REMOTEP_FIELD(remotePort),
            DATA_FIELD(data),
            NUMOPS_FIELD(g_netinfo[fd].numTX),
            UNIT_FIELD("byte"),
            FIELDEND
        };

        event_t txMetric = {"net.tx", g_netinfo[fd].txBytes, DELTA, fields};

        evtMetric(g_evt, g_cfg.hostname, g_netinfo[fd].uid, &txMetric);

        if ((g_cfg.summarize.net.rx_tx) && (source == EVENT_BASED)) {
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
            // tbd - consider calling doDNSMetricName instead...
            doNetMetric(DNS, sockfd, EVENT_BASED, 0);
        }
    }
    return 0;
}

static void
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

static void
doReset()
{
    g_cfg.pid = getpid();
    g_thread.once = 0;
    g_thread.startTime = time(NULL) + g_thread.interval;
    memset(&g_ctrs, 0, sizeof(struct metric_counters_t));
}

//
// reportFD is called in two cases:
//   1) when a socket or file is being closed
//   2) during periodic reporting
static void
reportFD(int fd, enum control_type_t source)
{
    struct net_info_t *ninfo = getNetEntry(fd);
    if (ninfo) {
        if (!g_cfg.summarize.net.rx_tx) {
            doNetMetric(NETTX, fd, source, 0);
            doNetMetric(NETRX, fd, source, 0);
        }
        if (!g_cfg.summarize.net.open_close) {
            doNetMetric(OPEN_PORTS, fd, source, 0);
            doNetMetric(NET_CONNECTIONS, fd, source, 0);
            doNetMetric(CONNECTION_DURATION, fd, source, 0);
        }
    }

    struct fs_info_t *finfo = getFSEntry(fd);
    if (finfo) {
        if (!g_cfg.summarize.fs.read_write) {
            doFSMetric(FS_DURATION, fd, source, "read/write", 0, NULL);
            doFSMetric(FS_READ, fd, source, "read", 0, NULL);
            doFSMetric(FS_WRITE, fd, source, "write", 0, NULL);
        }
        if (!g_cfg.summarize.fs.seek) {
            doFSMetric(FS_SEEK, fd, source, "seek", 0, NULL);
        }
    }
}

static void
reportPeriodicStuff(void)
{
    long mem;
    int i, nthread, nfds, children;
    long long cpu, cpuState = 0;

    // This is called by periodic(), and due to atexit().
    // If it's actively running for one reason, then skip the second.
    static uint64_t reentrancy_guard = 0ULL;
    if (!atomicCasU64(&reentrancy_guard, 0ULL, 1ULL)) return;


    // We report CPU time for this period.
    cpu = doGetProcCPU();
    doProcMetric(PROC_CPU, cpu - cpuState);
    cpuState = cpu;

    mem = osGetProcMemory(g_cfg.pid);
    doProcMetric(PROC_MEM, mem);

    nthread = osGetNumThreads(g_cfg.pid);
    doProcMetric(PROC_THREAD, nthread);

    nfds = osGetNumFds(g_cfg.pid);
    doProcMetric(PROC_FD, nfds);

    children = osGetNumChildProcs(g_cfg.pid);
    doProcMetric(PROC_CHILD, children);

    // report totals (not by file descriptor/socket descriptor)
    doTotal(TOT_READ);
    doTotal(TOT_WRITE);
    doTotal(TOT_RX);
    doTotal(TOT_TX);
    doTotal(TOT_SEEK);
    doTotal(TOT_STAT);
    doTotal(TOT_OPEN);
    doTotal(TOT_CLOSE);
    doTotal(TOT_DNS);

    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_UDP_CONN);
    doTotal(TOT_OTHER_CONN);

    doTotalDuration(TOT_FS_DURATION);
    doTotalDuration(TOT_NET_DURATION);
    doTotalDuration(TOT_DNS_DURATION);

    // Report errors
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary");
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary");
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary");
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary");
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary");
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary");

    // report net and file by descriptor
    for (i = 0; i < MAX(g_cfg.numNinfo, g_cfg.numFSInfo); i++) {
        reportFD(i, PERIODIC);
    }

    // Process any events that have been posted
    evtEvents(g_evt);

    if (!atomicCasU64(&reentrancy_guard, 1ULL, 0ULL)) {
         DBG(NULL);
    }
}

static void
handleExit(void)
{
    reportPeriodicStuff();
    outFlush(g_out);
    logFlush(g_log);
    evtFlush(g_evt);
}

static void *
periodic(void *arg)
{
    while (1) {
        reportPeriodicStuff();

        // Process dynamic config changes, if any
        dynConfig();

        // TODO: need to ensure that the previous object is no longer in use
        // Clean up previous objects if they exist.
        //if (g_prevout) outDestroy(&g_prevout);
        //if (g_prevlog) logDestroy(&g_prevlog);
        //if (g_prevevt) evtDestroy(&g_prevevt);

        if (evtNeedsConnection(g_evt)) {
            evtConnect(g_evt, g_staticfg);
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
    g_fn.__fread_chk = dlsym(RTLD_NEXT, "__fread_chk");
    g_fn.fread_unlocked = dlsym(RTLD_NEXT, "fread_unlocked");
    g_fn.fgets = dlsym(RTLD_NEXT, "fgets");
    g_fn.__fgets_chk = dlsym(RTLD_NEXT, "__fgets_chk");
    g_fn.fgets_unlocked = dlsym(RTLD_NEXT, "fgets_unlocked");
    g_fn.fgetws = dlsym(RTLD_NEXT, "fgetws");
    g_fn.__fgetws_chk = dlsym(RTLD_NEXT, "__fgetws_chk");
    g_fn.fgetwc = dlsym(RTLD_NEXT, "fgetwc");
    g_fn.fgetc = dlsym(RTLD_NEXT, "fgetc");
    g_fn.fscanf = dlsym(RTLD_NEXT, "fscanf");
    g_fn.fputc = dlsym(RTLD_NEXT, "fputc");
    g_fn.fputc_unlocked = dlsym(RTLD_NEXT, "fputc_unlocked");
    g_fn.fputwc = dlsym(RTLD_NEXT, "fputwc");
    g_fn.putwc = dlsym(RTLD_NEXT, "putwc");
    g_fn.getline = dlsym(RTLD_NEXT, "getline");
    g_fn.getdelim = dlsym(RTLD_NEXT, "getdelim");
    g_fn.__getdelim = dlsym(RTLD_NEXT, "__getdelim");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.pwrite = dlsym(RTLD_NEXT, "pwrite");
    g_fn.writev = dlsym(RTLD_NEXT, "writev");
    g_fn.fwrite = dlsym(RTLD_NEXT, "fwrite");
    g_fn.sendfile = dlsym(RTLD_NEXT, "sendfile");
    g_fn.fputs = dlsym(RTLD_NEXT, "fputs");
    g_fn.fputs_unlocked = dlsym(RTLD_NEXT, "fputs_unlocked");
    g_fn.fputws = dlsym(RTLD_NEXT, "fputws");
    g_fn.lseek = dlsym(RTLD_NEXT, "lseek");
    g_fn.fseek = dlsym(RTLD_NEXT, "fseek");
    g_fn.fseeko = dlsym(RTLD_NEXT, "fseeko");
    g_fn.ftell = dlsym(RTLD_NEXT, "ftell");
    g_fn.ftello = dlsym(RTLD_NEXT, "ftello");
    g_fn.fgetpos = dlsym(RTLD_NEXT, "fgetpos");
    g_fn.fsetpos = dlsym(RTLD_NEXT, "fsetpos");
    g_fn.fsetpos64 = dlsym(RTLD_NEXT, "fsetpos64");
    g_fn.stat = dlsym(RTLD_NEXT, "stat");
    g_fn.lstat = dlsym(RTLD_NEXT, "lstat");
    g_fn.fstat = dlsym(RTLD_NEXT, "fstat");
    g_fn.fstatat = dlsym(RTLD_NEXT, "fstatat");
    g_fn.statfs = dlsym(RTLD_NEXT, "statfs");
    g_fn.fstatfs = dlsym(RTLD_NEXT, "fstatfs");
    g_fn.statvfs = dlsym(RTLD_NEXT, "statvfs");
    g_fn.fstatvfs = dlsym(RTLD_NEXT, "fstatvfs");
    g_fn.access = dlsym(RTLD_NEXT, "access");
    g_fn.faccessat = dlsym(RTLD_NEXT, "faccessat");
    g_fn.rewind = dlsym(RTLD_NEXT, "rewind");
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
#endif // __MACOS__

#ifdef __LINUX__
    g_fn.open64 = dlsym(RTLD_NEXT, "open64");
    g_fn.openat64 = dlsym(RTLD_NEXT, "openat64");
    g_fn.__open_2 = dlsym(RTLD_NEXT, "__open_2");
    g_fn.__open64_2 = dlsym(RTLD_NEXT, "__open64_2");
    g_fn.__openat_2 = dlsym(RTLD_NEXT, "__openat_2");
    g_fn.fopen64 = dlsym(RTLD_NEXT, "fopen64");
    g_fn.freopen64 = dlsym(RTLD_NEXT, "freopen64");
    g_fn.creat64 = dlsym(RTLD_NEXT, "creat64");
    g_fn.pread64 = dlsym(RTLD_NEXT, "pread64");
    g_fn.preadv = dlsym(RTLD_NEXT, "preadv");
    g_fn.preadv2 = dlsym(RTLD_NEXT, "preadv2");
    g_fn.preadv64v2 = dlsym(RTLD_NEXT, "preadv64v2");
    g_fn.__pread_chk = dlsym(RTLD_NEXT, "__pread_chk");
    g_fn.__read_chk = dlsym(RTLD_NEXT, "__read_chk");
    g_fn.__fread_unlocked_chk = dlsym(RTLD_NEXT, "__fread_unlocked_chk");
    g_fn.pwrite64 = dlsym(RTLD_NEXT, "pwrite64");
    g_fn.pwritev = dlsym(RTLD_NEXT, "pwritev");
    g_fn.pwritev64 = dlsym(RTLD_NEXT, "pwritev64");
    g_fn.pwritev2 = dlsym(RTLD_NEXT, "pwritev2");
    g_fn.pwritev64v2 = dlsym(RTLD_NEXT, "pwritev64v2");
    g_fn.fwrite_unlocked = dlsym(RTLD_NEXT, "fwrite_unlocked");
    g_fn.sendfile64 = dlsym(RTLD_NEXT, "sendfile64");
    g_fn.lseek64 = dlsym(RTLD_NEXT, "lseek64");
    g_fn.fseeko64 = dlsym(RTLD_NEXT, "fseeko64");
    g_fn.ftello64 = dlsym(RTLD_NEXT, "ftello64");
    g_fn.statfs64 = dlsym(RTLD_NEXT, "statfs64");
    g_fn.fstatfs64 = dlsym(RTLD_NEXT, "fstatfs64");
    g_fn.fstatvfs64 = dlsym(RTLD_NEXT, "fstatvfs64");
    g_fn.fgetpos64 = dlsym(RTLD_NEXT, "fgetpos64");
    g_fn.statvfs64 = dlsym(RTLD_NEXT, "statvfs64");
    g_fn.__lxstat = dlsym(RTLD_NEXT, "__lxstat");
    g_fn.__lxstat64 = dlsym(RTLD_NEXT, "__lxstat64");
    g_fn.__xstat = dlsym(RTLD_NEXT, "__xstat");
    g_fn.__xstat64 = dlsym(RTLD_NEXT, "__xstat64");
    g_fn.__fxstat = dlsym(RTLD_NEXT, "__fxstat");
    g_fn.__fxstat64 = dlsym(RTLD_NEXT, "__fxstat64");
    g_fn.__fxstatat = dlsym(RTLD_NEXT, "__fxstatat");
    g_fn.__fxstatat64 = dlsym(RTLD_NEXT, "__fxstatat64");
    g_fn.gethostbyname_r = dlsym(RTLD_NEXT, "gethostbyname_r");
    g_fn.syscall = dlsym(RTLD_NEXT, "syscall");
    g_fn.prctl = dlsym(RTLD_NEXT, "prctl");
#ifdef __STATX__
    g_fn.statx = dlsym(RTLD_NEXT, "statx");
#endif // __STATX__
#endif // __LINUX__
    
    net_info *netinfoLocal;
    fs_info *fsinfoLocal;
    if ((netinfoLocal = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    g_cfg.pid = getpid();

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

    if (gethostname(g_cfg.hostname, sizeof(g_cfg.hostname)) != 0) {
        scopeLog("ERROR: Constructor:gethostname", -1, CFG_LOG_ERROR);
    }

    osGetProcname(g_cfg.procname, sizeof(g_cfg.procname));
    osInitTSC(&g_cfg);
    if (g_cfg.tsc_invariant == FALSE) {
        scopeLog("ERROR: TSC is not invariant", -1, CFG_LOG_ERROR);
    }

    char* path = cfgPath();
    config_t* cfg = cfgRead(path);
    cfgProcessEnvironment(cfg);
    doConfig(cfg);
    g_staticfg = cfg;
    if (path) free(path);
    if (!g_dbg) dbgInit();
    g_getdelim = 0;
    scopeLog("Constructor (Scope Version: " SCOPE_VER ")", -1, CFG_LOG_INFO);
    if (atexit(handleExit)) {
        DBG(NULL);
    }
}

static void
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
        g_fsinfo[fd].fd = fd;
        g_fsinfo[fd].type = type;
        g_fsinfo[fd].uid = getTime();
        strncpy(g_fsinfo[fd].path, path, sizeof(g_fsinfo[fd].path));

        if (evtSource(g_evt, CFG_SRC_LOGFILE) != DEFAULT_SRC_LOGFILE) {
            regmatch_t match = {0};
            if (regexec(evtLogFileFilter(g_evt), path, 1, &match, 0) == 0) {
                g_fsinfo[fd].event = TRUE;
            }
        }

        doFSMetric(FS_OPEN, fd, EVENT_BASED, func, 0, NULL);
        scopeLog(func, fd, CFG_LOG_TRACE);
    }
}

static int
doDupFile(int newfd, int oldfd, const char *func)
{
    if ((newfd > g_cfg.numFSInfo) || (oldfd > g_cfg.numFSInfo)) {
        return -1;
    }

    doOpen(newfd, g_fsinfo[oldfd].path, g_fsinfo[oldfd].type, func);
    return 0;
}

static int
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
    g_netinfo[newfd].totalDuration = 0;
    g_netinfo[newfd].numDuration = 0;

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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "open", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "openat", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "creat", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fopen", pathname);
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
    if (stream != NULL) {
        if (pathname != NULL) {
            doOpen(fileno(stream), pathname, STREAM, "freopen");
            doClose(fileno(orig_stream), "freopen");
        }
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "freopen", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "open64", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "openat64", pathname);
    }

    return fd;
}

EXPORTON int
__open_2(const char *file, int oflag)
{
    int fd;

    WRAP_CHECK(__open_2, -1);
    doThread();
    fd = g_fn.__open_2(file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__open_2");
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "__open_2", file);
    }

    return fd;
}

EXPORTON int
__open64_2(const char *file, int oflag)
{
    int fd;

    WRAP_CHECK(__open64_2, -1);
    doThread();
    fd = g_fn.__open64_2(file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__open_2");
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "__open64_2", file);
    }

    return fd;
}

EXPORTON int
__openat_2(int fd, const char *file, int oflag)
{
    WRAP_CHECK(__openat_2, -1);
    doThread();
    fd = g_fn.__openat_2(fd, file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__openat_2");
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "__openat_2", file);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "creat64", pathname);
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fopen64", pathname);
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
    if (stream != NULL) {
        if (pathname != NULL) {
            doOpen(fileno(stream), pathname, STREAM, "freopen64");
            doClose(fileno(orig_stream), "freopen64");
        }
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "freopen64", pathname);
    }

    return stream;
}

EXPORTON ssize_t
pread64(int fd, void *buf, size_t count, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pread64", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "pread64", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pread64", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pread64", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
preadv(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "preadv", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "preadv", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
preadv2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv2", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv2", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "preadv2", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "preadv2", "nopath");
        }
    }
    
    return rc;
}

EXPORTON ssize_t
preadv64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "preadv64v2", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "preadv64v2", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "preadv64v2", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "preadv64v2", "nopath");
        }
    }
    
    return rc;
}

EXPORTON ssize_t
__pread_chk(int fd, void * buf, size_t nbytes, off_t offset, size_t buflen)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
    elapsed_t time = {0};

    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__pread_chk, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }

    rc = g_fn.__pread_chk(fd, buf, nbytes, offset, buflen);

    if (fs) {
        time.duration = getDuration(time.initial);
    }

    if (rc != -1) {
        scopeLog("__pread_chk", fd, CFG_LOG_TRACE);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "__pread_chk", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "__pread_chk", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "__pread_chk", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "__pread_chk", "nopath");
        }
    }

    return rc;
}

EXPORTOFF ssize_t
__read_chk(int fd, void *buf, size_t nbytes, size_t buflen)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
    elapsed_t time = {0};

    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__read_chk, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }

    rc = g_fn.__read_chk(fd, buf, nbytes, buflen);

    if (fs) {
        time.duration = getDuration(time.initial);
    }

    if (rc != -1) {
        scopeLog("__read_chk", fd, CFG_LOG_TRACE);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "__read_chk", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "__read_chk", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "__read_chk", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "__read_chk", "nopath");
        }
    }

    return rc;
}

EXPORTOFF ssize_t
__fread_unlocked_chk(void *ptr, size_t ptrlen, size_t size, size_t nmemb, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fread_unlocked_chk, -1);
    IOSTREAMPRE(__fread_unlocked_chk, size_t);
    rc = g_fn.__fread_unlocked_chk(ptr, ptrlen, size, nmemb, stream);
    IOSTREAMPOST(__fread_unlocked_chk, ptr, rc * size, 0, (enum event_type_t)EVENT_RX);
}

EXPORTON ssize_t
pwrite64(int fd, const void *buf, size_t nbyte, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwrite64", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwrite64", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwrite64", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwrite64", "nopath");
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwritev", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwritev", "nopath");
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev64(int fd, const struct iovec *iov, int iovcnt, off64_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
    elapsed_t time = {0};

    WRAP_CHECK(pwritev64, -1);
    doThread();
    if (fs) {
        time.initial = getTime();
    }

    rc = g_fn.pwritev64(fd, iov, iovcnt, offset);

    if (fs) {
        time.duration = getDuration(time.initial);
    }

    if (rc != -1) {
        scopeLog("pwritev64", fd, CFG_LOG_TRACE);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev64", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev64", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwritev64", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwritev64", "nopath");
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev2", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev2", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwritev2", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwritev2", "nopath");
        }
    }
    return rc;
}

EXPORTON ssize_t
pwritev64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwritev64v2", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwritev64v2", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwritev64v2", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwritev64v2", "nopath");
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
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "lseek64", fs->path);
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
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fseek64", fs->path);
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
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "ftello64", fs->path);
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
        doStatMetric("statfs64", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "statfs64", path);
    }
    return rc;
}

EXPORTON int
fstatfs64(int fd, struct statfs64 *buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstatfs64, -1);
    doThread();
    rc = g_fn.fstatfs64(fd, buf);

    if (rc != -1) {
        scopeLog("fstatfs64", fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric("fstatfs64", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatfs64", fs->path);
        }
    }
    return rc;
}

EXPORTON int
fsetpos64(FILE *stream, const fpos64_t *pos)
{
    int rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fsetpos64, -1);
    doThread();
    rc = g_fn.fsetpos64(stream, pos);

    if (rc == 0) {
        scopeLog("fsetpos64", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fsetpos64", 0, NULL);
        }
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fsetpos64", fs->path);
    }

    return rc;
}

EXPORTON int
__xstat(int ver, const char *path, struct stat *stat_buf)
{
    int rc;

    WRAP_CHECK(__xstat, -1);
    doThread();
    rc = g_fn.__xstat(ver, path, stat_buf);

    if (rc != -1) {
        scopeLog("__xstat", -1, CFG_LOG_DEBUG);
        doStatMetric("__xstat", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__xstat", path);
    }
    return rc;    
}

EXPORTON int
__xstat64(int ver, const char *path, struct stat64 *stat_buf)
{
    int rc;

    WRAP_CHECK(__xstat64, -1);
    doThread();
    rc = g_fn.__xstat64(ver, path, stat_buf);

    if (rc != -1) {
        scopeLog("__xstat64", -1, CFG_LOG_DEBUG);
        doStatMetric("__xstat64", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__xstat64", path);
    }
    return rc;    
}

EXPORTON int
__lxstat(int ver, const char *path, struct stat *stat_buf)
{
    int rc;

    WRAP_CHECK(__lxstat, -1);
    doThread();
    rc = g_fn.__lxstat(ver, path, stat_buf);

    if (rc != -1) {
        scopeLog("__lxstat", -1, CFG_LOG_DEBUG);
        doStatMetric("__lxstat", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__lxstat", path);
    }
    return rc;
}

EXPORTON int
__lxstat64(int ver, const char *path, struct stat64 *stat_buf)
{
    int rc;

    WRAP_CHECK(__lxstat64, -1);
    doThread();
    rc = g_fn.__lxstat64(ver, path, stat_buf);

    if (rc != -1) {
        scopeLog("__lxstat64", -1, CFG_LOG_DEBUG);
        doStatMetric("__lxstat64", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__lxstat64", path);
    }
    return rc;
}

EXPORTON int
__fxstat(int ver, int fd, struct stat *stat_buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(__fxstat, -1);
    doThread();
    rc = g_fn.__fxstat(ver, fd, stat_buf);

    if (rc != -1) {
        scopeLog("__fxstat", -1, CFG_LOG_DEBUG);
        if (fs) doStatMetric("__fxstat", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__fxstat", fs->path);
        }
    }
    return rc;
}

EXPORTON int
__fxstat64(int ver, int fd, struct stat64 * stat_buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(__fxstat64, -1);
    doThread();
    rc = g_fn.__fxstat64(ver, fd, stat_buf);

    if (rc != -1) {
        scopeLog("__fxstat64", -1, CFG_LOG_DEBUG);
        if (fs) doStatMetric("__fxstat64", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__xstat64", fs->path);
        }
    }
    return rc;
}

EXPORTON int
__fxstatat(int ver, int dirfd, const char *path, struct stat *stat_buf, int flags)
{
    int rc;

    WRAP_CHECK(__fxstatat, -1);
    doThread();
    rc = g_fn.__fxstatat(ver, dirfd, path, stat_buf, flags);

    if (rc != -1) {
        scopeLog("__fxstatat", -1, CFG_LOG_DEBUG);
        doStatMetric("__fxstatat", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__fxstatat", path);
    }
    return rc;
}

EXPORTON int
__fxstatat64(int ver, int dirfd, const char * path, struct stat64 * stat_buf, int flags)
{
    int rc;

    WRAP_CHECK(__fxstatat64, -1);
    doThread();
    rc = g_fn.__fxstatat64(ver, dirfd, path, stat_buf, flags);

    if (rc != -1) {
        scopeLog("__fxstatat64", -1, CFG_LOG_DEBUG);
        doStatMetric("__fxstatat64", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "__fxstatat64", path);
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
        doStatMetric("statx", pathname);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "xstatx", pathname);
    }
    return rc;
}
#endif // __STATX__

EXPORTON int
statfs(const char *path, struct statfs *buf)
{
    int rc;

    WRAP_CHECK(statfs, -1);
    doThread();
    rc = g_fn.statfs(path, buf);

    if (rc != -1) {
        scopeLog("statfs", -1, CFG_LOG_DEBUG);
        doStatMetric("statfs", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatfs", path);
    }
    return rc;
}

EXPORTON int
fstatfs(int fd, struct statfs *buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstatfs, -1);
    doThread();
    rc = g_fn.fstatfs(fd, buf);

    if (rc != -1) {
        scopeLog("fstatfs", fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric("fstatfs", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatfs", fs->path);
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
        doStatMetric("statvfs", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "statvfs", path);
    }
    return rc;
}

EXPORTON int
statvfs64(const char *path, struct statvfs64 *buf)
{
    int rc;

    WRAP_CHECK(statvfs64, -1);
    doThread();
    rc = g_fn.statvfs64(path, buf);

    if (rc != -1) {
        scopeLog("statvfs64", -1, CFG_LOG_DEBUG);
        doStatMetric("statvfs64", path);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "statvfs64", path);
    }
    return rc;
}

EXPORTON int
fstatvfs(int fd, struct statvfs *buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstatvfs, -1);
    doThread();
    rc = g_fn.fstatvfs(fd, buf);

    if (rc != -1) {
        scopeLog("fstatvfs", fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric("fstatvfs", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatvfs", fs->path);
        }
    }
    return rc;
}

EXPORTON int
fstatvfs64(int fd, struct statvfs64 *buf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstatvfs64, -1);
    doThread();
    rc = g_fn.fstatvfs64(fd, buf);

    if (rc != -1) {
        scopeLog("fstatvfs64", fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric("fstatvfs64", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatvfs64", fs->path);
        }
    }
    return rc;
}

EXPORTON int
access(const char *pathname, int mode)
{
    int rc;

    WRAP_CHECK(access, -1);
    doThread();
    rc = g_fn.access(pathname, mode);

    if (rc != -1) {
        scopeLog("access", -1, CFG_LOG_DEBUG);
        doStatMetric("access", pathname);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "access", pathname);
    }
    return rc;
}

EXPORTON int
faccessat(int dirfd, const char *pathname, int mode, int flags)
{
    int rc;

    WRAP_CHECK(faccessat, -1);
    doThread();
    rc = g_fn.faccessat(dirfd, pathname, mode, flags);

    if (rc != -1) {
        scopeLog("faccessat", -1, CFG_LOG_DEBUG);
        doStatMetric("faccessat", pathname);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "faccessat", pathname);
    }
    return rc;
}

EXPORTON int
gethostbyname_r(const char *name, struct hostent *ret, char *buf, size_t buflen,
                struct hostent **result, int *h_errnop)
{
    int rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname_r, -1);
    time.initial = getTime();
    rc = g_fn.gethostbyname_r(name, ret, buf, buflen, result, h_errnop);
    time.duration = getDuration(time.initial);

    if ((rc == 0) && (result != NULL)) {
        scopeLog("gethostbyname_r", -1, CFG_LOG_DEBUG);
        doDNSMetricName(DNS, name, time.duration);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    }  else {
        doErrorMetric(NET_ERR_DNS, EVENT_BASED, "gethostbyname_r", name);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    }

    return rc;
}

/*
 * We explicitly don't interpose these stat functions on macOS
 * These are not exported symbols in Linux. Therefore, we
 * have them turned off for now.
 * stat, fstat, lstat.
 */
EXPORTOFF int
stat(const char *pathname, struct stat *statbuf)
{
    int rc;

    WRAP_CHECK(stat, -1);
    doThread();
    rc = g_fn.stat(pathname, statbuf);

    if (rc != -1) {
        scopeLog("stat", -1, CFG_LOG_DEBUG);
        doStatMetric("stat", pathname);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "stat", pathname);
    }
    return rc;
}

EXPORTOFF int
fstat(int fd, struct stat *statbuf)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstat, -1);
    doThread();
    rc = g_fn.fstat(fd, statbuf);

    if (rc != -1) {
        scopeLog("fstat", fd, CFG_LOG_DEBUG);
        if (fs) doStatMetric("fstat", fs->path);
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstat", fs->path);
        }
    }
    return rc;
}

EXPORTOFF int
lstat(const char *pathname, struct stat *statbuf)
{
    int rc;

    WRAP_CHECK(lstat, -1);
    doThread();
    rc = g_fn.lstat(pathname, statbuf);

    if (rc != -1) {
        scopeLog("lstat", -1, CFG_LOG_DEBUG);
        doStatMetric("lstat", pathname);
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "lstat", pathname);
    }
    return rc;
}

EXPORTON int
fstatat(int fd, const char *path, struct stat *buf, int flag)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fstatat, -1);
    doThread();
    rc = g_fn.fstatat(fd, path, buf, flag);

    if (rc != -1) {
        scopeLog("fstatat", fd, CFG_LOG_DEBUG);
        if (fs) {
            doStatMetric("fstatat", path);
        }
    } else {
        doErrorMetric(FS_ERR_STAT, EVENT_BASED, "fstatat", path);
    }

    return rc;
}

EXPORTON int
prctl(int option, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(prctl, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, option);

    if (option == PR_SET_SECCOMP) {
        return 0;
    }

    return g_fn.prctl(option, fArgs.arg[0], fArgs.arg[1], fArgs.arg[2], fArgs.arg[3]);
}

/*
 * Note:
 * The syscall function in libc is called from the loader for
 * at least mmap, possibly more. The result is that we can not
 * do any dynamic memory allocation while this executes. Be careful.
 * The DBG() output is ignored until after the constructor runs.
 */
EXPORTON long
syscall(long number, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(syscall, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, number);

    switch (number) {
    case SYS_accept4:
    {
        int rc;
        rc = g_fn.syscall(number, fArgs.arg[0], fArgs.arg[1],
                          fArgs.arg[2], fArgs.arg[3]);
        if (rc != -1) {
            doAccept(rc, (struct sockaddr *)fArgs.arg[1],
                     (socklen_t *)fArgs.arg[2], "accept4");
        } else {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, "accept4", "nopath");
        }
        return rc;
    }

    /*
     * These messages are in place as they represent
     * functions that use syscall() in libuv, used with node.js.
     * These are functions defined in libuv/src/unix/linux-syscalls.c
     * that we are otherwise interposing. The DBG call allows us to
     * check to see how many of these are called and therefore
     * what we are missing. So far, we only see accept4 used.
     */
    case SYS_sendmmsg:
        //DBG("syscall-sendmsg");
        break;

    case SYS_recvmmsg:
        //DBG("syscall-recvmsg");
        break;

    case SYS_preadv:
        //DBG("syscall-preadv");
        break;

    case SYS_pwritev:
        //DBG("syscall-pwritev");
        break;

    case SYS_dup3:
        //DBG("syscall-dup3");
        break;
#ifdef __STATX__
    case SYS_statx:
        //DBG("syscall-statx");
        break;
#endif // __STATX__
    default:
        // Supplying args is fine, but is a touch more work.
        // On splunk, in a container on my laptop, I saw this statement being
        // hit every 10-15 microseconds over a 15 minute duration.  Wow.
        // DBG("syscall-number: %d", number);
        //DBG(NULL);
        break;
    }

    return g_fn.syscall(number, fArgs.arg[0], fArgs.arg[1], fArgs.arg[2],
                        fArgs.arg[3], fArgs.arg[4], fArgs.arg[5]);
}

EXPORTON size_t
fwrite_unlocked(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
    WRAP_CHECK(fwrite_unlocked, -1);
    IOSTREAMPRE(fwrite_unlocked, size_t);
    rc = g_fn.fwrite_unlocked(ptr, size, nitems, stream);
    IOSTREAMPOST(fwrite_unlocked, ptr, rc * size, 0, (enum event_type_t)EVENT_TX);
}

/*
 * Note: in_fd must be a file
 * out_fd can be a file or a socket
 *
 * Not sure is this is the way we want to do this, but:
 * We emit metrics for the input file that is being sent
 * We optionally emit metrics if the destination uses a socket
 * We do not emit a separate metric if the destination is a file
 */
EXPORTON ssize_t
sendfile(int out_fd, int in_fd, off_t *offset, size_t count)
{
    doSendfile(sendfile);
}

EXPORTON ssize_t
sendfile64(int out_fd, int in_fd, off64_t *offset, size_t count)
{
    doSendfile(sendfile64);
}

#endif // __LINUX__

EXPORTON int
close(int fd)
{
    int rc;
    struct fs_info_t *fs;

    WRAP_CHECK(close, -1);
    doThread(); // Will do nothing if a thread already exists

    rc = g_fn.close(fd);
    if (rc != -1) {
        doClose(fd, "close");
    } else {
        if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "close", fs->path);
        }
    }

    return rc;
}

EXPORTON int
fclose(FILE *stream)
{
    int rc, fd;
    struct fs_info_t *fs;

    WRAP_CHECK(fclose, EOF);
    doThread(); // Will do nothing if a thread already exists
    fd = fileno(stream);

    rc = g_fn.fclose(stream);
    if (rc != EOF) {
        doClose(fd, "fclose");
    } else {
        if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fclose", fs->path);
        }
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
    } else {
        doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fcloseall", "nopath");
    }

    return rc;
}

#ifdef __MACOS__
EXPORTON int
close$NOCANCEL(int fd)
{
    int rc;
    struct fs_info_t *fs;

    WRAP_CHECK(close$NOCANCEL, -1);
    doThread();
    rc = g_fn.close$NOCANCEL(fd);
    if (rc != -1) {
        doClose(fd, "close$NOCANCEL");
    } else {
        if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "close$NOCANCEL", fs->path);
        }
    }

    return rc;
}


EXPORTON int
guarded_close_np(int fd, void *guard)
{
    int rc;
    struct fs_info_t *fs;

    WRAP_CHECK(guarded_close_np, -1);
    doThread();
    rc = g_fn.guarded_close_np(fd, guard);
    if (rc != -1) {
        doClose(fd, "guarded_close_np");
    } else {
         if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "guarded_close_np", fs->path);
        }
    }

    return rc;
}

EXPORTOFF int
close_nocancel(int fd)
{
    int rc;
    struct fs_info_t *fs;

    WRAP_CHECK(close_nocancel, -1);
    rc = g_fn.close_nocancel(fd);
    if (rc != -1) {
        doClose(fd, "close_nocancel");
    } else {
        if ((fs = getFSEntry(fd))) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "close_nocancel", fs->path);
        }
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
    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept$NOCANCEL");
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "accept$NOCANCEL", "nopath");
    }

    return sd;
}

EXPORTON ssize_t
__sendto_nocancel(int sockfd, const void *buf, size_t len, int flags,
                  const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;
    struct net_info_t *net = getNetEntry(sockfd);

    WRAP_CHECK(__sendto_nocancel, -1);
    doThread();
    rc = g_fn.__sendto_nocancel(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog("__sendto_nocancel", sockfd, CFG_LOG_TRACE);
        doSetAddrs(sockfd);

        if (net &&
            GET_PORT(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc);
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "__sendto_nocancel", "nopath");
    }

    return rc;
}

EXPORTON uint32_t
DNSServiceQueryRecord(void *sdRef, uint32_t flags, uint32_t interfaceIndex,
                      const char *fullname, uint16_t rrtype, uint16_t rrclass,
                      void *callback, void *context)
{
    uint32_t rc;
    elapsed_t time = {0};

    WRAP_CHECK(DNSServiceQueryRecord, -1);
    time.initial = getTime();
    rc = g_fn.DNSServiceQueryRecord(sdRef, flags, interfaceIndex, fullname,
                                    rrtype, rrclass, callback, context);
    time.duration = getDuration(time.initial);

    if (rc == 0) {
        scopeLog("DNSServiceQueryRecord", -1, CFG_LOG_DEBUG);
        doDNSMetricName(DNS, fullname, time.duration);
        doDNSMetricName(DNS_DURATION, fullname, time.duration);
    } else {
        doErrorMetric(NET_ERR_DNS, EVENT_BASED, "DNSServiceQueryRecord", fullname);
        doDNSMetricName(DNS_DURATION, fullname, time.duration);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "lseek", fs->path);
    }

    return rc;
}

EXPORTON int
fseek(FILE *stream, long offset, int whence)
{
    off_t rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fseek, -1);
    doThread();
    rc = g_fn.fseek(stream, offset, whence);

    if (rc != -1) {
        scopeLog("fseek", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fseek", 0, NULL);
        }
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fseek", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fseeko", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "ftell", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "ftello", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "rewind", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fsetpos", fs->path);
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
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fgetpos", fs->path);
    }

    return rc;
}

EXPORTON int
fgetpos64(FILE *stream,  fpos64_t *pos)
{
    int rc;
    int fd = fileno(stream);
    struct fs_info_t *fs = getFSEntry(fd);

    WRAP_CHECK(fgetpos64, -1);
    doThread();
    rc = g_fn.fgetpos64(stream, pos);

    if (rc == 0) {
        scopeLog("fgetpos64", fd, CFG_LOG_DEBUG);
        if (fs) {
            doFSMetric(FS_SEEK, fd, EVENT_BASED, "fgetpos64", 0, NULL);
        }
    } else if (fs) {
        doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "fgetpos64", fs->path);
    }

    return rc;
}

EXPORTON ssize_t
write(int fd, const void *buf, size_t count)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "write", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "write", rc, NULL);
            doEventLog(g_evt, fs, buf, count);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "write", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "write", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
pwrite(int fd, const void *buf, size_t nbyte, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pwrite", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "pwrite", rc, NULL);
        }
    } else {
         if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pwrite", fs->path);
         } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pwrite", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
writev(int fd, const struct iovec *iov, int iovcnt)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net != NULL) {
            // This is a network descriptor
            doSetAddrs(fd);
            doSend(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "writev", time.duration, NULL);
            doFSMetric(FS_WRITE, fd, EVENT_BASED, "writev", rc, NULL);
        }
    } else {
         if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "writev", fs->path);
         } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "writev", "nopath");
        }
    }

    return rc;
}

EXPORTON size_t
fwrite(const void *restrict ptr, size_t size, size_t nitems, FILE *restrict stream)
{
    WRAP_CHECK(fwrite, -1);
    IOSTREAMPRE(fwrite, size_t);
    rc = g_fn.fwrite(ptr, size, nitems, stream);
    IOSTREAMPOST(fwrite, ptr, rc * size, 0, (enum event_type_t)EVENT_TX);
}

EXPORTON int
fputs(const char *s, FILE *stream)
{
    WRAP_CHECK(fputs, EOF);
    IOSTREAMPRE(fputs, int);
    rc = g_fn.fputs(s, stream);
    IOSTREAMPOST(fputs, s, strlen(s), EOF, (enum event_type_t)EVENT_TX);
}

EXPORTON int
fputs_unlocked(const char *s, FILE *stream)
{
    WRAP_CHECK(fputs_unlocked, EOF);
    IOSTREAMPRE(fputs_unlocked, int);
    rc = g_fn.fputs_unlocked(s, stream);
    IOSTREAMPOST(fputs_unlocked, s, strlen(s), EOF, (enum event_type_t)EVENT_TX);
}

EXPORTON int
fputws(const wchar_t *ws, FILE *stream)
{
    WRAP_CHECK(fputws, EOF);
    IOSTREAMPRE(fputws, int);
    rc = g_fn.fputws(ws, stream);
    IOSTREAMPOST(fputws, ws, wcslen(ws), EOF, (enum event_type_t)EVENT_TX);
}

EXPORTON ssize_t
read(int fd, void *buf, size_t count)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "read", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "read", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "read", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "read", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
readv(int fd, const struct iovec *iov, int iovcnt)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "readv", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "readv", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "readv", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "readv", "nopath");
        }
    }

    return rc;
}

EXPORTON ssize_t
pread(int fd, void *buf, size_t count, off_t offset)
{
    ssize_t rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
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
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            doRecv(fd, rc);
        } else if (fs) {
            doFSMetric(FS_DURATION, fd, EVENT_BASED, "pread", time.duration, NULL);
            doFSMetric(FS_READ, fd, EVENT_BASED, "pread", rc, NULL);
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_READ_WRITE, EVENT_BASED, "pread", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "pread", "nopath");
        }
    }

    return rc;
}

EXPORTON size_t
fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    WRAP_CHECK(fread, -1);
    IOSTREAMPRE(fread, size_t);
    rc = g_fn.fread(ptr, size, nmemb, stream);
    IOSTREAMPOST(fread, ptr, rc * size, 0, (enum event_type_t)EVENT_RX);
}

EXPORTON size_t
__fread_chk(void *ptr, size_t ptrlen, size_t size, size_t nmemb, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fread_chk, -1);
    IOSTREAMPRE(__fread_chk, size_t);
    rc = g_fn.__fread_chk(ptr, ptrlen, size, nmemb, stream);
    IOSTREAMPOST(__fread_chk, ptr, rc * size, 0, (enum event_type_t)EVENT_RX);
}

EXPORTON size_t
fread_unlocked(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    WRAP_CHECK(fread_unlocked, 0);
    IOSTREAMPRE(fread_unlocked, size_t);
    rc = g_fn.fread_unlocked(ptr, size, nmemb, stream);
    IOSTREAMPOST(fread_unlocked, ptr, rc * size, 0, (enum event_type_t)EVENT_RX);
}

EXPORTON char *
fgets(char *s, int n, FILE *stream)
{
    WRAP_CHECK(fgets, NULL);
    IOSTREAMPRE(fgets, char *);
    rc = g_fn.fgets(s, n, stream);
    IOSTREAMPOST(fgets, s, n, NULL, (enum event_type_t)EVENT_RX);
}

EXPORTON char *
__fgets_chk(char *s, size_t size, int strsize, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fgets_chk, NULL);
    IOSTREAMPRE(__fgets_chk, char *);
    rc = g_fn.__fgets_chk(s, size, strsize, stream);
    IOSTREAMPOST(__fgets_chk, s, size, NULL, (enum event_type_t)EVENT_RX);
}

EXPORTON char *
fgets_unlocked(char *s, int n, FILE *stream)
{
    WRAP_CHECK(fgets_unlocked, NULL);
    IOSTREAMPRE(fgets_unlocked, char *);
    rc = g_fn.fgets_unlocked(s, n, stream);
    IOSTREAMPOST(fgets_unlocked, s, n, NULL, (enum event_type_t)EVENT_RX);
}

EXPORTON wchar_t *
__fgetws_chk(wchar_t *ws, size_t size, int strsize, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fgetws_chk, NULL);
    IOSTREAMPRE(__fgetws_chk, wchar_t *);
    rc = g_fn.__fgetws_chk(ws, size, strsize, stream);
    IOSTREAMPOST(__fgetws_chk, ws, size, NULL, (enum event_type_t)EVENT_RX);
}

EXPORTON wchar_t *
fgetws(wchar_t *ws, int n, FILE *stream)
{
    WRAP_CHECK(fgetws, NULL);
    IOSTREAMPRE(fgetws, wchar_t *);
    rc = g_fn.fgetws(ws, n, stream);
    IOSTREAMPOST(fgetws, ws, n, NULL, (enum event_type_t)EVENT_RX);
}

EXPORTON wint_t
fgetwc(FILE *stream)
{
    WRAP_CHECK(fgetwc, WEOF);
    IOSTREAMPRE(fgetwc, wint_t);
    rc = g_fn.fgetwc(stream);
    IOSTREAMPOST(fgetwc, NULL, 1, WEOF, (enum event_type_t)EVENT_RX);
}

EXPORTON int
fgetc(FILE *stream)
{
    WRAP_CHECK(fgetc, EOF);
    IOSTREAMPRE(fgetc, int);
    rc = g_fn.fgetc(stream);
    IOSTREAMPOST(fgetc, NULL, 1, EOF, (enum event_type_t)EVENT_FS);
}

EXPORTON int
fputc(int c, FILE *stream)
{
    WRAP_CHECK(fputc, EOF);
    IOSTREAMPRE(fputc, int);
    rc = g_fn.fputc(c, stream);
    IOSTREAMPOST(fputc, NULL, 1, EOF, (enum event_type_t)EVENT_FS);
}

EXPORTON int
fputc_unlocked(int c, FILE *stream)
{
    WRAP_CHECK(fputc_unlocked, EOF);
    IOSTREAMPRE(fputc_unlocked, int);
    rc = g_fn.fputc_unlocked(c, stream);
    IOSTREAMPOST(fputc_unlocked, NULL, 1, EOF, (enum event_type_t)EVENT_FS);
}

EXPORTON wint_t
putwc(wchar_t wc, FILE *stream)
{
    WRAP_CHECK(putwc, WEOF);
    IOSTREAMPRE(putwc, int);
    rc = g_fn.putwc(wc, stream);
    IOSTREAMPOST(putwc, NULL, 1, WEOF, (enum event_type_t)EVENT_FS);
}

EXPORTON wint_t
fputwc(wchar_t wc, FILE *stream)
{
    WRAP_CHECK(fputwc, WEOF);
    IOSTREAMPRE(fputwc, int);
    rc = g_fn.fputwc(wc, stream);
    IOSTREAMPOST(fputwc, NULL, 1, WEOF, (enum event_type_t)EVENT_FS);
}

EXPORTOFF int
fscanf(FILE *stream, const char *format, ...)
{
    struct FuncArgs fArgs;
    LOAD_FUNC_ARGS_VALIST(fArgs, format);
    WRAP_CHECK(fscanf, EOF);
    IOSTREAMPRE(fscanf, int);
    rc = g_fn.fscanf(stream, format,
                     fArgs.arg[0], fArgs.arg[1],
                     fArgs.arg[2], fArgs.arg[3],
                     fArgs.arg[4], fArgs.arg[5]);
    IOSTREAMPOST(fscanf, NULL, rc, EOF, (enum event_type_t)EVENT_RX);
}

EXPORTON ssize_t
getline (char **lineptr, size_t *n, FILE *stream)
{
    WRAP_CHECK(getline, -1);
    IOSTREAMPRE(getline, ssize_t);
    rc = g_fn.getline(lineptr, n, stream);
    if (n) {
        IOSTREAMPOST(getline, NULL, *n, -1, (enum event_type_t)EVENT_RX);
    } else {
        IOSTREAMPOST(getline, NULL, 0, -1, (enum event_type_t)EVENT_RX);
    }
}

EXPORTON ssize_t
getdelim (char **lineptr, size_t *n, int delimiter, FILE *stream)
{
    WRAP_CHECK(getdelim, -1);
    IOSTREAMPRE(getdelim, ssize_t);
    g_getdelim = 1;
    rc = g_fn.getdelim(lineptr, n, delimiter, stream);
    if (n) {
        IOSTREAMPOST(getdelim, NULL, *n, -1, (enum event_type_t)EVENT_RX);
    } else {
        IOSTREAMPOST(getdelim, NULL, 0, -1, (enum event_type_t)EVENT_RX);
    }
}

EXPORTON ssize_t
__getdelim (char **lineptr, size_t *n, int delimiter, FILE *stream)
{
    WRAP_CHECK(__getdelim, -1);
    IOSTREAMPRE(__getdelim, ssize_t);
    rc = g_fn.__getdelim(lineptr, n, delimiter, stream);
    if (g_getdelim == 1) {
        g_getdelim = 0;
        return rc;
    }

    if (n) {
        IOSTREAMPOST(__getdelim, NULL, *n, -1, (enum event_type_t)EVENT_RX);
    } else {
        IOSTREAMPOST(__getdelim, NULL, 0, -1, (enum event_type_t)EVENT_RX);
    }
}

EXPORTON int
fcntl(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    WRAP_CHECK(fcntl, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                    fArgs.arg[2], fArgs.arg[3]);
    if (cmd == F_DUPFD) {
        if (rc != -1) {
            if (net) {
                // This is a network descriptor
                scopeLog("fcntl", rc, CFG_LOG_DEBUG);
                doAddNewSock(rc);
            } else if (fs) {
                doDupFile(fd, rc, "fcntl");
            }
        } else {
            if (fs) {
                doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fcntl", fs->path);
            } else if (net) {
                doErrorMetric(NET_ERR_CONN, EVENT_BASED, "fcntl", "nopath");
            }
        }
    }
    
    return rc;
}

EXPORTON int
fcntl64(int fd, int cmd, ...)
{
    int rc;
    struct FuncArgs fArgs;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    WRAP_CHECK(fcntl64, -1);
    doThread();
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    rc = g_fn.fcntl64(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                      fArgs.arg[2], fArgs.arg[3]);
    if (cmd == F_DUPFD) {
        if (rc != -1) {
            if (net) {
                // This is a network descriptor
                scopeLog("fcntl", rc, CFG_LOG_DEBUG);
                doAddNewSock(rc);
            } else if (fs) {
                doDupFile(fd, rc, "fcntl64");
            }
        } else {
            if (fs) {
                doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "fcntl", fs->path);
            } else if (net) {
                doErrorMetric(NET_ERR_CONN, EVENT_BASED, "fcntl", "nopath");
            }
        }
    }

    return rc;
}

EXPORTON int
dup(int fd)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    WRAP_CHECK(dup, -1);
    doThread();
    rc = g_fn.dup(fd);
    if (rc != -1) {
        if (net) {
            // This is a network descriptor
            scopeLog("dup", rc, CFG_LOG_DEBUG);
            doDupSock(fd, rc);
        } else if (fs) {
            doDupFile(fd, rc, "dup");
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "dup", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, "dup", "nopath");
        }
    }

    return rc;
}

EXPORTON int
dup2(int oldfd, int newfd)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(oldfd);
    struct net_info_t *net = getNetEntry(oldfd);

    WRAP_CHECK(dup2, -1);
    doThread();
    rc = g_fn.dup2(oldfd, newfd);
    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog("dup2", rc, CFG_LOG_DEBUG);
        if (net) {
            if (getNetEntry(newfd)) {
                doClose(newfd, "dup2");
            }
            doDupSock(oldfd, newfd);
        } else if (fs) {
            if (getFSEntry(newfd)) {
                doClose(newfd, "dup2");
            }
            doDupFile(oldfd, newfd, "dup2");
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "dup2", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, "dup2", "nopath");
        }
    }

    return rc;
}

EXPORTON int
dup3(int oldfd, int newfd, int flags)
{
    int rc;
    struct fs_info_t *fs = getFSEntry(oldfd);
    struct net_info_t *net = getNetEntry(oldfd);

    WRAP_CHECK(dup3, -1);
    doThread();
    rc = g_fn.dup3(oldfd, newfd, flags);
    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog("dup3", rc, CFG_LOG_DEBUG);
        if (net) {
            if (getNetEntry(newfd)) {
                doClose(newfd, "dup3");
            }
            doDupSock(oldfd, newfd);
        } else if (fs) {
            if (getFSEntry(newfd)) {
                doClose(newfd, "dup3");
            }
            doDupFile(oldfd, newfd, "dup3");
        }
    } else {
        if (fs) {
            doErrorMetric(FS_ERR_OPEN_CLOSE, EVENT_BASED, "dup3", fs->path);
        } else if (net) {
            doErrorMetric(NET_ERR_CONN, EVENT_BASED, "dup3", "nopath");
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
             (socket_family == AF_INET6))) {

            /*
             * State used in close()
             * We define that a UDP socket represents an open 
             * port when created and is open until the socket is closed
             *
             * a UDP socket is open we say the port is open
             * a UDP socket is closed we say the port is closed
             */
            doNetMetric(OPEN_PORTS, sd, EVENT_BASED, 1);
        }
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "socket", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "shutdown", "nopath");
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

        if (net) {
            doNetMetric(OPEN_PORTS, sockfd, EVENT_BASED, 1);
            doNetMetric(NET_CONNECTIONS, sockfd, EVENT_BASED, 1);
        }
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "listen", "nopath");
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
    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept");
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "accept", "nopath");
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
    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept4");
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "accept4", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "bind", "nopath");
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
        doNetMetric(NET_CONNECTIONS, sockfd, EVENT_BASED, 1);

        scopeLog("connect", sockfd, CFG_LOG_DEBUG);
    } else {
        doErrorMetric(NET_ERR_CONN, EVENT_BASED, "connect", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "send", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "sendto", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "sendmsg", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "recv", "nopath");
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
    rc = g_fn.recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    if (rc != -1) {
        scopeLog("recvfrom", sockfd, CFG_LOG_TRACE);
        doRecv(sockfd, rc);
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "recvfrom", "nopath");
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
    } else {
        doErrorMetric(NET_ERR_RX_TX, EVENT_BASED, "recvmsg", "nopath");
    }
    
    return rc;
}

EXPORTON struct hostent *
gethostbyname(const char *name)
{
    struct hostent *rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname, NULL);
    time.initial = getTime();
    rc = g_fn.gethostbyname(name);
    time.duration = getDuration(time.initial);

    if (rc != NULL) {
        scopeLog("gethostbyname", -1, CFG_LOG_DEBUG);
        doDNSMetricName(DNS, name, time.duration);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    } else {
        doErrorMetric(NET_ERR_DNS, EVENT_BASED, "gethostbyname", name);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    }

    return rc;
}

EXPORTON struct hostent *
gethostbyname2(const char *name, int af)
{
    struct hostent *rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname2, NULL);
    time.initial = getTime();
    rc = g_fn.gethostbyname2(name, af);
    time.duration = getDuration(time.initial);

    if (rc != NULL) {
        scopeLog("gethostbyname2", -1, CFG_LOG_DEBUG);
        doDNSMetricName(DNS, name, time.duration);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    } else {
        doErrorMetric(NET_ERR_DNS, EVENT_BASED, "gethostbyname2", name);
        doDNSMetricName(DNS_DURATION, name, time.duration);
    }

    return rc;
}

EXPORTON int
getaddrinfo(const char *node, const char *service,
            const struct addrinfo *hints,
            struct addrinfo **res)
{
    int rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(getaddrinfo, -1);
    time.initial = getTime();
    rc = g_fn.getaddrinfo(node, service, hints, res);
    time.duration = getDuration(time.initial);

    if (rc == 0) {
        scopeLog("getaddrinfo", -1, CFG_LOG_DEBUG);
        doDNSMetricName(DNS, node, time.duration);
        doDNSMetricName(DNS_DURATION, node, time.duration);
    } else {
        doErrorMetric(NET_ERR_DNS, EVENT_BASED, "getaddrinfo", node);
        doDNSMetricName(DNS_DURATION, node, time.duration);
    }


    return rc;
}

#ifdef __LINUX__

static const char scope_help[] =
"ENV VARIABLES:\n"
"\n"
"    SCOPE_CONF_PATH\n"
"        Directly specify location and name of config file.\n"
"        Used only at start-time.\n"
"    SCOPE_HOME\n"
"        Specify a directory from which conf/scope.yml or ./scope.yml can\n"
"        be found.  Used only at start-time only if SCOPE_CONF_PATH does\n"
"        not exist.  For more info see Config File Resolution below.\n"
"    SCOPE_OUT_VERBOSITY\n"
"        0-9 are valid values.  Default is 4.\n"
"        For more info see Verbosity below.\n"
"    SCOPE_OUT_SUM_PERIOD\n"
"        Number of seconds between output summarizations.  Default is 10\n"
"    SCOPE_OUT_DEST\n"
"        Default is udp://localhost:8125\n"
"        Format is one of:\n"
"            file:///tmp/output.log\n"
"            udp://server:123         (server is servername or address;\n"
"                                      123 is port number or service name)\n"
"    SCOPE_OUT_FORMAT\n"
"        metricstatsd, metricjson\n"
"        Default is metricstatsd\n"
"    SCOPE_STATSD_PREFIX\n"
"        Specify a string to be prepended to every scope metric.\n"
"    SCOPE_STATSD_MAXLEN\n"
"        Default is 512\n"
"    SCOPE_EVENT_DEST\n"
"        same format as SCOPE_OUT_DEST above.\n"
"        Default is tcp://localhost:9109\n"
"    SCOPE_EVENT_FORMAT\n"
"        eventjsonrawjson, eventjsonrawstatsd\n"
"        Default is eventjsonrawjson\n"
"    SCOPE_EVENT_LOGFILE\n"
"        Create events from logs that match SCOPE_EVENT_LOG_FILTER.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_CONSOLE\n"
"        Create events from stdout, stderr.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_SYSLOG\n"
"        Create events from syslog, vsyslog functions.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_METRICS\n"
"        Create events from metrics.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_LOG_FILTER\n"
"        An extended regular expression that describes log file names.\n"
"        Only used if SCOPE_EVENT_LOGFILE is true.  Default is .*log.*\n"
"    SCOPE_LOG_LEVEL\n"
"        debug, info, warning, error, none.  Default is error.\n"
"    SCOPE_LOG_DEST\n"
"        same format as SCOPE_OUT_DEST above.\n"
"        Default is file:///tmp/scope.log\n"
"    SCOPE_TAG_\n"
"        Specify a tag to be applied to every metric.  Environment variable\n"
"        expansion is available e.g. SCOPE_TAG_user=$USER\n"
"    SCOPE_CMD_DIR\n"
"        Specifies a directory to look for dynamic configuration files.\n"
"        See Dynamic Configuration below.\n"
"        Default is /tmp\n"
"\n"
"FEATURES:\n"
"\n"
"    Verbosity\n"
"        Controls two different aspects of output - \n"
"        Tag Cardinality and Summarization.\n"
"\n"
"        Tag Cardinality\n"
"            0   No expanded statsd tags\n"
"            1   adds 'data', 'unit'\n"
"            2   adds 'class', 'proto'\n"
"            3   adds 'op'\n"
"            4   adds 'host', 'proc'\n"
"            5   adds 'domain', 'file'\n"
"            6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'\n"
"            7   adds 'fd', 'pid'\n"
"            8   adds 'duration'\n"
"            9   adds 'numops'\n"
"\n"
"        Summarization\n"
"            0-4 has full event summarization\n"
"            5   turns off 'error'\n"
"            6   turns off 'filesystem open/close' and 'dns'\n"
"            7   turns off 'filesystem stat' and 'network connect'\n"
"            8   turns off 'filesystem seek'\n"
"            9   turns off 'filesystem read/write' and 'network send/receive'\n"
"\n"
"    Config File Resolution\n"
"        If the SCOPE_CONF_PATH env variable is defined and points to a\n"
"        file that can be opened, it will use this as the config file.  If\n"
"        not, it searches for the config file in this priority order using the\n"
"        first it finds.  If this fails too, it will look for scope.yml in the\n"
"        same directory as LD_PRELOAD.\n"
"\n"
"            $SCOPE_HOME/conf/scope.yml\n"
"            $SCOPE_HOME/scope.yml\n"
"            /etc/scope/scope.yml\n"
"            ~/conf/scope.yml\n"
"            ~/scope.yml\n"
"            ./conf/scope.yml\n"
"            ./scope.yml\n"
"\n"
"    Dynamic Configuration\n"
"        Dynamic Configuration allows configuration settings to be\n"
"        changed on the fly after process start-time.  At every\n"
"        SCOPE_OUT_SUM_PERIOD the library looks in SCOPE_CMD_DIR to\n"
"        see if a file scope.<pid> exists.  If it exists, it processes\n"
"        every line, looking for environment variable-style commands.\n"
"        (e.g. SCOPE_CMD_DBG_PATH=/tmp/outfile.txt)  It changes the\n"
"        configuration to match the new settings, and deletes the\n"
"        scope.<pid> file when it's complete.\n";

// assumes that we're only building for 64 bit...
char const __invoke_dynamic_linker__[] __attribute__ ((section (".interp"))) = "/lib64/ld-linux-x86-64.so.2";

void
__scope_main(void)
{
    printf("Scope Version: " SCOPE_VER "\n");

    char buf[64];
    if (snprintf(buf, sizeof(buf), "/proc/%d/exe", getpid()) == -1) exit(0);
    char path[1024] = {0};
    if (readlink(buf, path, sizeof(path)) == -1) exit(0);
    printf("\n");
    printf("   Usage: LD_PRELOAD=%s <command name>\n ", path);
    printf("\n");
    printf("\n");
    printf("%s", scope_help);
    printf("\n");
    exit(0);
}

#endif // __LINUX__
