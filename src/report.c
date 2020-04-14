#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "atomic.h"
#include "com.h"
#include "dbg.h"
#include "mtcformat.h"
#include "os.h"
#include "plattime.h"
#include "report.h"
#include "state_private.h"
#include "linklist.h"

#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif


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
#define RATE_FIELD(val)         NUMFIELD("req_per_sec",    (val),        8)
#define HREQ_FIELD(val)         STRFIELD("req",            (val),        8)
#define HRES_FIELD(val)         STRFIELD("resp",           (val),        8)


// TBD - Ideally, we'd remove this dependency on the configured interval
// and replace it with a measured value.  It'd be one less dependency
// and could be more accurate.
int g_interval = DEFAULT_SUMMARY_PERIOD;
static list_t *g_maplist;

static void
destroyHttpMap(void *data)
{
    if (!data) return;
    http_map *map = (http_map *)data;

    if (map->req) free(map->req);
    if (map->resp) free(map->resp);
    if (map) free(map);
}

void
initReporting()
{
    g_maplist = lstCreate(destroyHttpMap);
}

void
setReportingInterval(int seconds)
{
    g_interval = seconds;
}

static void
sendEvent(mtc_t *mtc, event_t *event)
{
    cmdSendEvent(g_ctl, event, getTime(), &g_proc);

    if (cmdSendMetric(mtc, event) == -1) {
        scopeLog("ERROR: doProcMetric:CPU:cmdSendMetric", -1, CFG_LOG_ERROR);
    }
}

void
sendProcessStartMetric()
{
    char *urlEncodedCmd = fmtUrlEncode(g_proc.cmd);
    event_field_t fields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        HOST_FIELD(g_proc.hostname),
        ARGS_FIELD(urlEncodedCmd),
        UNIT_FIELD("process"),
        FIELDEND
    };
    event_t evt = INT_EVENT("proc.start", 1, DELTA, fields);
    cmdSendMetric(g_mtc, &evt);
    if (urlEncodedCmd) free(urlEncodedCmd);
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

void
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

static void
destroyProto(protocol_info *proto)
{
    if (!proto) return;

    if ((proto->ptype == EVT_HREQ) || (proto->ptype == EVT_HRES)) {
        http_post *post = (http_post *)proto->data;
        if (post) free(post);
    }
}

void
doProtocolMetric(protocol_info *proto)
{
    if (!proto) return;

    if ((proto->ptype == EVT_HREQ) || (proto->ptype == EVT_HRES)) {
        http_post *post = (http_post *)proto->data;
        http_map *map;

        if ((map = lstFind(g_maplist, post->id)) == NULL) {
            // lazy open
            if ((map = calloc(1, sizeof(http_map))) == NULL) {
                destroyProto(proto);
                return;
            }

            if (lstInsert(g_maplist, post->id, map) == FALSE) {
                destroyHttpMap(map);
                destroyProto(proto);
                return;
            }

            map->id = post->id;
            map->first_time = time(NULL);
        }

        map->frequency++;

        if (proto->ptype == EVT_HREQ) {
            map->start_time = post->start_duration;
            map->req = (char *)post->hdr;

            event_field_t fields[] = {
                HREQ_FIELD(map->req),
                UNIT_FIELD("byte"),
                FIELDEND
            };

            event_t sendEvent = INT_EVENT("http-req", proto->len, SET, fields);
            cmdSendHttp(g_ctl, &sendEvent, map->id, &g_proc);
        } else if (proto->ptype == EVT_HRES) {
            int rps = map->frequency;
            int sec = (map->first_time > 0) ? (int)time(NULL) - map->first_time : 1;
            if (sec > 0) {
                rps = map->frequency / sec;
            }

            map->duration = getDuration(post->start_duration);
            map->duration = map->duration / 1000;
            map->resp = (char *)post->hdr;

            if (!map->req) {
                map->req = strdup("None");
            }

            event_field_t hfields[] = {
                HREQ_FIELD(map->req),
                HRES_FIELD(map->resp),
                UNIT_FIELD("byte"),
                FIELDEND
            };

            event_t hevent = INT_EVENT("http-resp", proto->len, SET, hfields);
            cmdSendHttp(g_ctl, &hevent, map->id, &g_proc);

            event_field_t mfields[] = {
                DURATION_FIELD(map->duration),
                RATE_FIELD(rps),
                PROC_FIELD(g_proc.procname),
                FD_FIELD(proto->fd),
                PID_FIELD(g_proc.pid),
                UNIT_FIELD("byte"),
                FIELDEND
            };

            event_t mevent = INT_EVENT("http-metrics", proto->len, SET, mfields);
            cmdSendHttp(g_ctl, &mevent, map->id, &g_proc);

            // Done; we remove the list entry; complete when reported
            if (lstDelete(g_maplist, post->id) == FALSE) DBG(NULL);
        }
    }

    destroyProto(proto);
}

void
resetInterfaceCounts(counters_element_t* value)
{
    if (!value) return;
    atomicSwapU64(&value->mtc, 0);
    atomicSwapU64(&value->evt, 0);
}

void
addToInterfaceCounts(counters_element_t* value, uint64_t x)
{
    if (!value) return;
    atomicAddU64(&value->mtc, x);
    atomicAddU64(&value->evt, x);
}

void
subFromInterfaceCounts(counters_element_t* value, uint64_t x)
{
     if (!value) return;
     atomicSubU64(&value->mtc, x);
     atomicSubU64(&value->evt, x);
}

void
doErrorMetric(metric_t type, control_type_t source,
              const char *func, const char *name, void* ctr)
{
    if (!func || !name) return;

    metric_counters* ctrs = (ctr) ? (metric_counters*) ctr : &g_ctrs;

    const char err_name[] = "EFAULT";
    if (errno == EFAULT) {
        name = err_name;
    }

    switch (type) {
    case NET_ERR_CONN:
    case NET_ERR_RX_TX:
    {
        counters_element_t* value = NULL;
        const char* class = "UNKNOWN";
        switch (type) {
            case NET_ERR_CONN:
                value = &ctrs->netConnectErrors;
                class = "connection";
                break;
            case NET_ERR_RX_TX:
                value = &ctrs->netTxRxErrors;
                class = "rx_tx";
                break;
            default:
                DBG(NULL);
                return;
        }

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(func),
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        // Don't report zeros.
        if (value->evt != 0ULL) {
             event_t netErrMetric = INT_EVENT("net.error", value->evt, DELTA, fields);
             cmdSendEvent(g_ctl, &netErrMetric, getTime(), &g_proc);
             atomicSwapU64(&value->evt, 0);
        }

        // Only report if enabled
        if ((g_summary.net.error) && (source == EVENT_BASED)) {
            return;
        }
        // Don't report zeros.
        if (value->mtc == 0) return;

        event_t netErrMetric = INT_EVENT("net.error", value->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &netErrMetric)) {
            scopeLog("ERROR: doErrorMetric:NET:cmdSendMetric", -1, CFG_LOG_ERROR);
        }
        atomicSwapU64(&value->mtc, 0);
        break;
    }

    case FS_ERR_OPEN_CLOSE:
    case FS_ERR_READ_WRITE:
    case FS_ERR_STAT:
    case NET_ERR_DNS:
    {

        const char* metric = NULL;
        counters_element_t* value = NULL;
        const char* class = "UNKNOWN";
        int* summarize = NULL;
        event_field_t file_field = FILE_FIELD(name);
        event_field_t domain_field = DOMAIN_FIELD(name);
        event_field_t* name_field;
        switch (type) {
            case FS_ERR_OPEN_CLOSE:
                metric = "fs.error";
                value = &ctrs->fsOpenCloseErrors;
                class = "open_close";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_READ_WRITE:
                metric = "fs.error";
                value = &ctrs->fsRdWrErrors;
                class = "read_write";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case FS_ERR_STAT:
                metric = "fs.error";
                value = &ctrs->fsStatErrors;
                class = "stat";
                summarize = &g_summary.fs.error;
                name_field = &file_field;
                break;
            case NET_ERR_DNS:
                metric = "net.error";
                value = &ctrs->netDNSErrors;
                class = "dns";
                summarize = &g_summary.net.dnserror;
                name_field = &domain_field;
                break;
            default:
                DBG(NULL);
                return;
        }

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

        // Don't report zeros.
        if (value->evt != 0ULL) {
            event_t fsErrMetric = INT_EVENT(metric, value->evt, DELTA, fields);
            cmdSendEvent(g_ctl, &fsErrMetric, getTime(), &g_proc);
            atomicSwapU64(&value->evt, 0);
        }

        // Only report if enabled
        if ((source == EVENT_BASED) && *summarize) {
            return;
        }
        // Don't report zeros.
        if (value->mtc == 0) return;

        event_t fsErrMetric = INT_EVENT(metric, value->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &fsErrMetric)) {
            scopeLog("ERROR: doErrorMetric:FS_ERR:cmdSendMetric", -1, CFG_LOG_ERROR);
        }
        atomicSwapU64(&value->mtc, 0);
        break;
    }

    default:
        scopeLog("ERROR: doErrorMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doDNSMetricName(metric_t type, const char *domain, counters_element_t* duration, void* ctr)
{
    if (!domain || !domain[0]) return;

    metric_counters* ctrs = (ctr) ? (metric_counters*) ctr : &g_ctrs;

    switch (type) {
    case DNS:
    {
        // Don't report zeros.
        if (ctrs->numDNS.evt != 0) {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD(domain),
                DURATION_FIELD(duration->evt / 1000000), // convert ns to ms.
                UNIT_FIELD("request"),
                FIELDEND
            };
            event_t dnsMetric = INT_EVENT("net.dns", ctrs->numDNS.evt, DELTA, fields);
            cmdSendEvent(g_ctl, &dnsMetric, getTime(), &g_proc);
        }

        // Only report if enabled
        if (g_summary.net.dns) {
            return;
        }

        // Don't report zeros.
        if (ctrs->numDNS.mtc == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            DOMAIN_FIELD(domain),
            DURATION_FIELD(duration->mtc / 1000000), // convert ns to ms.
            UNIT_FIELD("request"),
            FIELDEND
        };
        event_t dnsMetric = INT_EVENT("net.dns", ctrs->numDNS.mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &dnsMetric)) {
            scopeLog("ERROR: doDNSMetricName:DNS:cmdSendMetric", -1, CFG_LOG_ERROR);
        }
        break;
    }

    case DNS_DURATION:
    {
        addToInterfaceCounts(&ctrs->dnsDurationNum, 1);
        atomicAddU64(&ctrs->dnsDurationTotal.mtc, duration->mtc);
        atomicAddU64(&ctrs->dnsDurationTotal.evt, duration->evt);

        uint64_t dur = 0ULL;
        int cachedDurationNum = ctrs->dnsDurationNum.evt; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000000 converts ns to ms.
            dur = ctrs->dnsDurationTotal.evt / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (dur != 0ULL) {
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
            cmdSendEvent(g_ctl, &dnsDurMetric, getTime(), &g_proc);
            atomicSwapU64(&ctrs->dnsDurationNum.evt, 0);
            atomicSwapU64(&ctrs->dnsDurationTotal.evt, 0);
        }

        // Only report if enabled
        if (g_summary.net.dns) {
            return;
        }

        dur = 0ULL;
        cachedDurationNum = ctrs->dnsDurationNum.mtc; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000000 converts ns to ms.
            dur = ctrs->dnsDurationTotal.mtc / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros
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
        if (cmdSendMetric(g_mtc, &dnsDurMetric)) {
            scopeLog("ERROR: doDNSMetricName:DNS_DURATION:cmdSendMetric", -1, CFG_LOG_ERROR);
        }
        atomicSwapU64(&ctrs->dnsDurationNum.mtc, 0);
        atomicSwapU64(&ctrs->dnsDurationTotal.mtc, 0);
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
            event_t event = INT_EVENT("proc.cpu", measurement, DELTA, fields);
            sendEvent(g_mtc, &event);
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
            event_t event = FLT_EVENT("proc.cpu_perc", val, CURRENT, fields);
            sendEvent(g_mtc, &event);
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
        event_t event = INT_EVENT("proc.mem", measurement, DELTA, fields);
        sendEvent(g_mtc, &event);
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
        event_t event = INT_EVENT("proc.thread", measurement, CURRENT, fields);
        sendEvent(g_mtc, &event);
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
        event_t event = INT_EVENT("proc.fd", measurement, CURRENT, fields);
        sendEvent(g_mtc, &event);
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
        event_t event = INT_EVENT("proc.child", measurement, CURRENT, fields);
        sendEvent(g_mtc, &event);
        break;
    }

    default:
        scopeLog("ERROR: doProcMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doStatMetric(const char *op, const char *pathname, void* ctr)
{

    metric_counters* ctrs = (ctr) ? (metric_counters*) ctr : &g_ctrs;

    event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(pathname),
            UNIT_FIELD("operation"),
            FIELDEND
    };

    if (ctrs->numStat.evt != 0) {
        event_t evt = INT_EVENT("fs.op.stat", ctrs->numStat.evt, DELTA, fields);
        cmdSendEvent(g_ctl, &evt, getTime(), &g_proc);
    }

    // Only report if enabled
    if (g_summary.fs.stat) {
        return;
    }

    // Do not report zeros
    if (ctrs->numStat.mtc == 0) return;

    event_t evt = INT_EVENT("fs.op.stat", ctrs->numStat.mtc, DELTA, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLog("doStatMetric", -1, CFG_LOG_ERROR);
    }
}

void
doFSMetric(metric_t type, fs_info *fs, control_type_t source,
           const char *op, ssize_t size, const char *pathname)
{
    if (!fs) return;

    switch (type) {
    case FS_DURATION:
    {

        uint64_t dur = 0ULL;
        int cachedDurationNum = fs->numDuration.evt; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000 converts ns to us.
            dur = fs->totalDuration.evt / ( 1000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (dur != 0ULL) {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fs->fd),
                HOST_FIELD(g_proc.hostname),
                OP_FIELD(op),
                FILE_FIELD(fs->path),
                NUMOPS_FIELD(cachedDurationNum),
                UNIT_FIELD("microsecond"),
                FIELDEND
            };

            event_t evt = INT_EVENT("fs.duration", dur, HISTOGRAM, fields);
            cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
            atomicSwapU64(&fs->numDuration.evt, 0);
            atomicSwapU64(&fs->totalDuration.evt, 0);
            //atomicSwapU64(&g_ctrs.fsDurationNum.evt, 0);
            //atomicSwapU64(&g_ctrs.fsDurationTotal.evt, 0);
        }

        dur = 0ULL;
        cachedDurationNum = fs->numDuration.mtc; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000 converts ns to us.
            dur = fs->totalDuration.mtc / ( 1000 * cachedDurationNum);
        }

        // Don't report zeros
        if (dur == 0ULL) return;

        // Only report if enabled
        if ((g_summary.fs.read_write) && (source == EVENT_BASED)) {
            return;
        }

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fs->fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(fs->path),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("microsecond"),
            FIELDEND
        };
        event_t evt = INT_EVENT("fs.duration", dur, HISTOGRAM, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLog("ERROR: doFSMetric:FS_DURATION:cmdSendMetric", fs->fd, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        atomicSwapU64(&fs->numDuration.mtc, 0);
        atomicSwapU64(&fs->totalDuration.mtc, 0);
        //atomicSwapU64(&g_ctrs.fsDurationNum.mtc, 0);
        //atomicSwapU64(&g_ctrs.fsDurationTotal.mtc, 0);
        break;
    }

    case FS_READ:
    case FS_WRITE:
    {
        const char* metric = "UNKNOWN";
        counters_element_t* numops = NULL;
        counters_element_t* sizebytes = NULL;
        counters_element_t* global_counter = NULL;
        const char* err_str = "UNKNOWN";
        switch (type) {
            case FS_READ:
                metric = "fs.read";
                numops = &fs->numRead;
                sizebytes = &fs->readBytes;
                global_counter = &g_ctrs.readBytes;
                err_str = "ERROR: doFSMetric:FS_READ:cmdSendMetric";
                break;
            case FS_WRITE:
                metric = "fs.write";
                numops = &fs->numWrite;
                sizebytes = &fs->writeBytes;
                global_counter = &g_ctrs.writeBytes;
                err_str = "ERROR: doFSMetric:FS_WRITE:cmdSendMetric";
                break;
            default:
                DBG(NULL);
                return;
        }

        // Don't report zeros
        if (sizebytes->evt != 0ULL) {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(fs->fd),
                HOST_FIELD(g_proc.hostname),
                OP_FIELD(op),
                FILE_FIELD(fs->path),
                NUMOPS_FIELD(numops->evt),
                UNIT_FIELD("byte"),
                FIELDEND
            };

            event_t rwMetric = INT_EVENT(metric, sizebytes->evt, HISTOGRAM, fields);
            cmdSendEvent(g_ctl, &rwMetric, fs->uid, &g_proc);
            atomicSwapU64(&numops->evt, 0);
            atomicSwapU64(&sizebytes->evt, 0);
            //atomicSwapU64(global_counter->evt, 0);
        }

        // Only report if enabled
        if ((g_summary.fs.read_write) && (source == EVENT_BASED)) {
            return;
        }

        // Don't report zeros
        if (sizebytes->mtc == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fs->fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(fs->path),
            NUMOPS_FIELD(numops->mtc),
            UNIT_FIELD("byte"),
            FIELDEND
        };

        event_t rwMetric = INT_EVENT(metric, sizebytes->mtc, HISTOGRAM, fields);

        if (cmdSendMetric(g_mtc, &rwMetric)) {
            scopeLog(err_str, fs->fd, CFG_LOG_ERROR);
        }
        subFromInterfaceCounts(global_counter, sizebytes->mtc);
        atomicSwapU64(&numops->mtc, 0);
        atomicSwapU64(&sizebytes->mtc, 0);

        break;
    }
    case FS_OPEN:
    case FS_CLOSE:
    case FS_SEEK:
    {
        const char* metric = "UNKNOWN";
        counters_element_t* numops = NULL;
        counters_element_t* global_counter = NULL;
        int* summarize = NULL;
        const char* err_str = "UNKNOWN";
        switch (type) {
            case FS_OPEN:
                metric = "fs.op.open";
                numops = &fs->numOpen;
                global_counter = &g_ctrs.numOpen;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_OPEN:cmdSendMetric";
                break;
            case FS_CLOSE:
                metric = "fs.op.close";
                numops = &fs->numClose;
                global_counter = &g_ctrs.numClose;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_CLOSE:cmdSendMetric";
                break;
            case FS_SEEK:
                metric = "fs.op.seek";
                numops = &fs->numSeek;
                global_counter = &g_ctrs.numSeek;
                summarize = &g_summary.fs.seek;
                err_str = "ERROR: doFSMetric:FS_SEEK:cmdSendMetric";
                break;
            default:
                DBG(NULL);
                return;
        }

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(fs->fd),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(fs->path),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        // Don't report zeros.
        if (numops->evt != 0ULL) {
            event_t evt = INT_EVENT(metric, numops->evt, DELTA, fields);
            cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
            atomicSwapU64(&numops->evt, 0);
        }

        // Only report if enabled
        if (*summarize && (source == EVENT_BASED)) {
            return;
        }

        // Don't report zeros.
        if (numops->mtc == 0ULL) return;

        event_t evt = INT_EVENT(metric, numops->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLog(err_str, fs->fd, CFG_LOG_ERROR);
        }
        subFromInterfaceCounts(global_counter, numops->mtc);
        atomicSwapU64(&numops->mtc, 0);
        break;
    }

    default:
        DBG(NULL);
    }
}

const char *bucketName[SOCK_NUM_BUCKETS] = {
    "inet_tcp", //    INET_TCP,
    "inet_udp", //    INET_UDP,
    "unix_tcp", //    UNIX_TCP,
    "unix_udp", //    UNIX_UDP,
    "other"     //    SOCK_OTHER,
};

static void
doTotalNetRxTx(metric_t type)
{
    // This just exists because TOT_RX and TOT_TX have sub-buckets...
    const char* metric = "UNKNOWN";
    counters_element_t (*value)[SOCK_NUM_BUCKETS] = NULL;
    const char* err_str = "UNKNOWN";
    const char* units = "byte";
    switch (type) {
        case TOT_RX:
            metric = "net.rx";
            value = &g_ctrs.netrxBytes;
            err_str = "ERROR: doTotal:TOT_RX:cmdSendMetric";
            break;
        case TOT_TX:
            metric = "net.tx";
            value = &g_ctrs.nettxBytes;
            err_str = "ERROR: doTotal:TOT_TX:cmdSendMetric";
            break;
        default:
            DBG(NULL);
            return;
    }

    sock_summary_bucket_t bucket;
    for (bucket = INET_TCP; bucket < SOCK_NUM_BUCKETS; bucket++) {

        // Don't report zeros.
        if ((*value)[bucket].mtc == 0) continue;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            UNIT_FIELD(units),
            CLASS_FIELD(bucketName[bucket]),
            FIELDEND
        };
        event_t evt = INT_EVENT(metric, (*value)[bucket].mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLog(err_str, -1, CFG_LOG_ERROR);
        }

        // Reset the info we tried to report (if it's not a gauge)
        atomicSwapU64(&(*value)[bucket].mtc, 0);
    }
}

void
doTotal(metric_t type)
{
    const char* metric = "UNKNOWN";
    counters_element_t* value = NULL;
    const char* err_str = "UNKNOWN";
    const char* units = "byte";
    data_type_t aggregation_type = DELTA;
    switch (type) {
        case TOT_READ:
            metric = "fs.read";
            value = &g_ctrs.readBytes;
            err_str = "ERROR: doTotal:TOT_READ:cmdSendMetric";
            break;
        case TOT_WRITE:
            metric = "fs.write";
            value = &g_ctrs.writeBytes;
            err_str = "ERROR: doTotal:TOT_WRITE:cmdSendMetric";
            break;
        case TOT_RX:
        case TOT_TX:
            doTotalNetRxTx(type);
            return;   // <--  We're doing the work above; nothing to see here.
        case TOT_SEEK:
            metric = "fs.seek";
            value = &g_ctrs.numSeek;
            err_str = "ERROR: doTotal:TOT_SEEK:cmdSendMetric";
            units = "operation";
            break;
        case TOT_STAT:
            metric = "fs.stat";
            value = &g_ctrs.numStat;
            err_str = "ERROR: doTotal:TOT_STAT:cmdSendMetric";
            units = "operation";
            break;
        case TOT_OPEN:
            metric = "fs.open";
            value = &g_ctrs.numOpen;
            err_str = "ERROR: doTotal:TOT_OPEN:cmdSendMetric";
            units = "operation";
            break;
        case TOT_CLOSE:
            metric = "fs.close";
            value = &g_ctrs.numClose;
            err_str = "ERROR: doTotal:TOT_CLOSE:cmdSendMetric";
            units = "operation";
            break;
        case TOT_DNS:
            metric = "net.dns";
            value = &g_ctrs.numDNS;
            err_str = "ERROR: doTotal:TOT_DNS:cmdSendMetric";
            units = "operation";
            break;
        case TOT_PORTS:
            metric = "net.port";
            value = &g_ctrs.openPorts;
            err_str = "ERROR: doTotal:TOT_PORTS:cmdSendMetric";
            units = "instance";
            aggregation_type = CURRENT;
            break;
        case TOT_TCP_CONN:
            metric = "net.tcp";
            value = &g_ctrs.netConnectionsTcp;
            err_str = "ERROR: doTotal:TOT_TCP_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_UDP_CONN:
            metric = "net.udp";
            value = &g_ctrs.netConnectionsUdp;
            err_str = "ERROR: doTotal:TOT_UDP_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_OTHER_CONN:
            metric = "net.other";
            value = &g_ctrs.netConnectionsOther;
            err_str = "ERROR: doTotal:TOT_OTHER_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        default:
            DBG(NULL);
            return;
    }

    // Don't report zeros.
    if (value->mtc == 0) return;

    event_field_t fields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        HOST_FIELD(g_proc.hostname),
        UNIT_FIELD(units),
        CLASS_FIELD("summary"),
        FIELDEND
    };
    event_t evt = INT_EVENT(metric, value->mtc, aggregation_type, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report (if it's not a gauge)
    if (aggregation_type != CURRENT) atomicSwapU64(&value->mtc, 0);
}

void
doTotalDuration(metric_t type)
{
    const char* metric = "UNKNOWN";
    counters_element_t* value = NULL;
    counters_element_t* num = NULL;
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
            err_str = "ERROR: doTotalDuration:TOT_FS_DURATION:cmdSendMetric";
            break;
        case TOT_NET_DURATION:
            metric = "net.conn_duration";
            value = &g_ctrs.connDurationTotal;
            num = &g_ctrs.connDurationNum;
            aggregation_type = DELTA_MS;
            units = "millisecond";
            factor = 1000000;
            err_str = "ERROR: doTotalDuration:TOT_NET_DURATION:cmdSendMetric";
            break;
        case TOT_DNS_DURATION:
            metric = "net.dns.duration";
            value = &g_ctrs.dnsDurationTotal;
            num = &g_ctrs.dnsDurationNum;
            aggregation_type = DELTA_MS;
            units = "millisecond";
            factor = 1000000;
            err_str = "ERROR: doTotalDuration:TOT_DNS_DURATION:cmdSendMetric";
            break;
        default:
            DBG(NULL);
            return;
    }

    uint64_t dur = 0ULL;
    int cachedDurationNum = num->mtc; // avoid div by zero
    if (cachedDurationNum >= 1) {
        // factor is there to scale from ns to the appropriate units
        dur = value->mtc / ( factor * cachedDurationNum);
    }

    // Don't report zeros.
    if (dur == 0) return;

    event_field_t fields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        HOST_FIELD(g_proc.hostname),
        UNIT_FIELD(units),
        CLASS_FIELD("summary"),
        FIELDEND
    };
    event_t evt = INT_EVENT(metric, dur, aggregation_type, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLog(err_str, -1, CFG_LOG_ERROR);
    }

    // Reset the info we tried to report
    atomicSwapU64(&value->mtc, 0);
    atomicSwapU64(&num->mtc, 0);
}

void
doNetMetric(metric_t type, net_info *net, control_type_t source, ssize_t size)
{
    char proto[PROTOCOL_STR];
    in_port_t localPort, remotePort;

    if (!net) return;

    getProtocol(net->type, proto, sizeof(proto));
    localPort = get_port_net(net, net->localConn.ss_family, LOCAL);
    remotePort = get_port_net(net, net->remoteConn.ss_family, REMOTE);

    switch (type) {
    case OPEN_PORTS:
    case NET_CONNECTIONS:
    {
        const char* metric = "UNKNOWN";
        counters_element_t* value = NULL;
        const char* units = "UNKNOWN";
        const char* err_str = "UNKNOWN";

        switch (type) {
        case OPEN_PORTS:
            metric = "net.port";
            value = &g_ctrs.openPorts;
            units = "instance";
            err_str = "ERROR: doNetMetric:OPEN_PORTS:cmdSendMetric";
            break;
        case NET_CONNECTIONS:
            if (net->type == SOCK_STREAM) {
                metric = "net.tcp";
                value = &g_ctrs.netConnectionsTcp;
            } else if (net->type == SOCK_DGRAM) {
                metric = "net.udp";
                value = &g_ctrs.netConnectionsUdp;
            } else {
                metric = "net.other";
                value = &g_ctrs.netConnectionsOther;
            }
            units = "connection";
            err_str = "ERROR: doNetMetric:NET_CONNECTIONS:cmdSendMetric";
            break;
        default:
            DBG(NULL);
            return;
        }

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(net->fd),
            HOST_FIELD(g_proc.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            UNIT_FIELD(units),
            FIELDEND
        };

        {
            event_t evt = INT_EVENT(metric, value->evt, CURRENT, fields);
            cmdSendEvent(g_ctl, &evt, net->uid, &g_proc);
            // Don't reset the info if we tried to report.  It's a gauge.
            //atomicSwapU64(value->evt, 0ULL);
        }

        // Only report if enabled
        if ((g_summary.net.open_close) && (source == EVENT_BASED)) {
            return;
        }

        event_t evt = INT_EVENT(metric, value->mtc, CURRENT, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLog(err_str, net->fd, CFG_LOG_ERROR);
        }
        // Don't reset the info if we tried to report.  It's a gauge.
        // atomicSwapU64(value, 0);

        break;
    }

    case CONNECTION_DURATION:
    {
        uint64_t dur = 0ULL;
        int cachedDurationNum = net->numDuration.evt; // avoid div by zero
        if (cachedDurationNum >= 1 ) {
            // factor of 1000000 converts ns to ms.
            dur = net->totalDuration.evt / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (dur != 0ULL) {
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(net->fd),
                HOST_FIELD(g_proc.hostname),
                PROTO_FIELD(proto),
                PORT_FIELD(localPort),
                NUMOPS_FIELD(cachedDurationNum),
                UNIT_FIELD("millisecond"),
                FIELDEND
            };
            event_t evt = INT_EVENT("net.conn_duration", dur, DELTA_MS, fields);
            cmdSendEvent(g_ctl, &evt, net->uid, &g_proc);
            atomicSwapU64(&net->numDuration.evt, 0);
            atomicSwapU64(&net->totalDuration.evt, 0);
         }

        // Only report if enabled
        if ((g_summary.net.open_close) && (source == EVENT_BASED)) {
            return;
        }

        dur = 0ULL;
        cachedDurationNum = net->numDuration.mtc; // avoid div by zero
        if (cachedDurationNum >= 1 ) {
            // factor of 1000000 converts ns to ms.
            dur = net->totalDuration.mtc / ( 1000000 * cachedDurationNum);
        }

        // Don't report zeros.
        if (dur == 0ULL) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(net->fd),
            HOST_FIELD(g_proc.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };
        event_t evt = INT_EVENT("net.conn_duration", dur, DELTA_MS, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLog("ERROR: doNetMetric:CONNECTION_DURATION:cmdSendMetric", net->fd, CFG_LOG_ERROR);
        }
        atomicSwapU64(&net->numDuration.mtc, 0);
        atomicSwapU64(&net->totalDuration.mtc, 0);
        //atomicSwapU64(&g_ctrs.connDurationNum, 0);
        //atomicSwapU64(&g_ctrs.connDurationTotal, 0);
        break;
    }

    case NETRX:
    {
        event_t rxMetric;
        event_field_t rxFields[20];
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        // Don't report zeros.
        if (net->rxBytes.evt == 0ULL) return;

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        // Do we need to define domain=LOCAL or NETLINK?
        if (addrIsUnixDomain(&net->remoteConn) ||
            addrIsUnixDomain(&net->localConn)) {
            localPort = net->lnode;
            remotePort = net->rnode;

            if (net->localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(net->fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("UNIX"),
                PROTO_FIELD(proto),
                LOCALN_FIELD(localPort),
                REMOTEN_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(net->numRX.evt),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&rxFields, &fields, sizeof(fields));
            event_t rxUnixMetric = INT_EVENT("net.rx", net->rxBytes.evt, DELTA, rxFields);
            memmove(&rxMetric, &rxUnixMetric, sizeof(event_t));
        } else {
            if (net->localConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }
            } else if (net->localConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }

            } else {
                strncpy(lip, " ", sizeof(lip));
            }

            if (net->remoteConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else if (net->remoteConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else {
                strncpy(rip, " ", sizeof(rip));
            }
            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(net->fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("AF_INET"),
                PROTO_FIELD(proto),
                LOCALIP_FIELD(lip),
                LOCALP_FIELD(localPort),
                REMOTEIP_FIELD(rip),
                REMOTEP_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(net->numRX.evt),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&rxFields, &fields, sizeof(fields));
            event_t rxNetMetric = INT_EVENT("net.rx", net->rxBytes.evt, DELTA, rxFields);
            memmove(&rxMetric, &rxNetMetric, sizeof(event_t));
        }

        // Don't report zeros.
        if (net->rxBytes.evt != 0ULL) {

             cmdSendEvent(g_ctl, &rxMetric, net->uid, &g_proc);
             atomicSwapU64(&net->numRX.evt, 0);
             atomicSwapU64(&net->rxBytes.evt, 0);
        }

        if ((g_summary.net.rx_tx) && (source == EVENT_BASED)) {
            return;
        }

        // Don't report zeros.
        if (net->rxBytes.mtc == 0ULL) return;

        if ((g_summary.net.rx_tx) && (source == EVENT_BASED)) {
            return;
        }

        event_t rxNetMetric = INT_EVENT("net.rx", net->rxBytes.mtc, DELTA, rxFields);
        memmove(&rxMetric, &rxNetMetric, sizeof(event_t));
        if (cmdSendMetric(g_mtc, &rxMetric)) {
            scopeLog("ERROR: doNetMetric:NETRX:cmdSendMetric", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        sock_summary_bucket_t bucket = getNetRxTxBucket(net);
        subFromInterfaceCounts(&g_ctrs.netrxBytes[bucket], net->rxBytes.mtc);
        atomicSwapU64(&net->numRX.mtc, 0);
        atomicSwapU64(&net->rxBytes.mtc, 0);
        break;
    }

    case NETTX:
    {
        event_t txMetric;
        event_field_t txFields[20];
        char lip[INET6_ADDRSTRLEN];
        char rip[INET6_ADDRSTRLEN];
        char data[16];

        // Don't report zeros.
        if (net->txBytes.evt == 0ULL) return;

        if ((localPort == 443) || (remotePort == 443)) {
            strncpy(data, "ssl", sizeof(data));
        } else {
            strncpy(data, "clear", sizeof(data));
        }

        if (addrIsUnixDomain(&net->remoteConn) ||
            addrIsUnixDomain(&net->localConn)) {
            localPort = net->lnode;
            remotePort = net->rnode;

            if (net->localConn.ss_family == AF_NETLINK) {
                strncpy(proto, "NETLINK", sizeof(proto));
            }

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(net->fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("UNIX"),
                PROTO_FIELD(proto),
                LOCALN_FIELD(localPort),
                REMOTEN_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(net->numRX.evt),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&txFields, &fields, sizeof(fields));
            event_t txUnixMetric = INT_EVENT("net.tx", net->txBytes.evt, DELTA, txFields);
            memmove(&txMetric, &txUnixMetric, sizeof(event_t));
        } else {
            if (net->localConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }
            } else if (net->localConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    strncpy(lip, " ", sizeof(lip));
                }

            } else {
                strncpy(lip, " ", sizeof(lip));
            }

            if (net->remoteConn.ss_family == AF_INET) {
                if (inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else if (net->remoteConn.ss_family == AF_INET6) {
                if (inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    strncpy(rip, " ", sizeof(rip));
                }
            } else {
                strncpy(rip, " ", sizeof(rip));
            }

            event_field_t fields[] = {
                PROC_FIELD(g_proc.procname),
                PID_FIELD(g_proc.pid),
                FD_FIELD(net->fd),
                HOST_FIELD(g_proc.hostname),
                DOMAIN_FIELD("AF_INET"),
                PROTO_FIELD(proto),
                LOCALIP_FIELD(lip),
                LOCALP_FIELD(localPort),
                REMOTEIP_FIELD(rip),
                REMOTEP_FIELD(remotePort),
                DATA_FIELD(data),
                NUMOPS_FIELD(net->numRX.evt),
                UNIT_FIELD("byte"),
                FIELDEND
            };
            memmove(&txFields, &fields, sizeof(fields));
            event_t txNetMetric = INT_EVENT("net.tx", net->txBytes.evt, DELTA, txFields);
            memmove(&txMetric, &txNetMetric, sizeof(event_t));
        }

        // Don't report zeros.
        if (net->txBytes.evt != 0ULL) {

            cmdSendEvent(g_ctl, &txMetric, net->uid, &g_proc);
            atomicSwapU64(&net->numTX.evt, 0);
            atomicSwapU64(&net->txBytes.evt, 0);
        }

        // Don't report zeros.
        if (net->txBytes.mtc == 0ULL) return;

        if ((g_summary.net.rx_tx) && (source == EVENT_BASED)) {
            return;
        }

        event_t txNetMetric = INT_EVENT("net.tx", net->txBytes.mtc, DELTA, txFields);
        memmove(&txMetric, &txNetMetric, sizeof(event_t));
        if (cmdSendMetric(g_mtc, &txMetric)) {
            scopeLog("ERROR: doNetMetric:NETTX:cmdSendMetric", -1, CFG_LOG_ERROR);
        }

        // Reset the info if we tried to report
        sock_summary_bucket_t bucket = getNetRxTxBucket(net);
        subFromInterfaceCounts(&g_ctrs.nettxBytes[bucket], net->txBytes.mtc);
        atomicSwapU64(&net->numTX.mtc, 0);
        atomicSwapU64(&net->txBytes.mtc, 0);

        break;
    }

    case DNS:
    {
        if (net->dnsSend == FALSE) {
            break;
        }

        // For next time
        net->dnsSend = FALSE;

        // TBD - this is only called by doSend.  Consider calling this directly
        // from there?
        doDNSMetricName(DNS, net->dnsName, 0, NULL);

        break;
    }

    default:
        scopeLog("ERROR: doNetMetric:metric type", -1, CFG_LOG_ERROR);
    }
}

void
doEvent()
{
    uint64_t data;
    while ((data = msgEventGet(g_ctl)) != -1) {
        if (data) {
            evt_type *event = (evt_type *)data;
            net_info *net;
            fs_info *fs;
            stat_err_info *staterr;
            protocol_info *proto;

            if (event->evtype == EVT_NET) {
                net = (net_info *)data;
                doNetMetric(net->data_type, net, EVENT_BASED, 0);
            } else if (event->evtype == EVT_FS) {
                fs = (fs_info *)data;
                doFSMetric(fs->data_type, fs, EVENT_BASED, fs->funcop, 0, fs->path);
            } else if (event->evtype == EVT_ERR) {
                staterr = (stat_err_info *)data;
                doErrorMetric(staterr->data_type, EVENT_BASED, staterr->funcop, staterr->name, &staterr->counters);
            } else if (event->evtype == EVT_STAT) {
                staterr = (stat_err_info *)data;
                doStatMetric(staterr->funcop, staterr->name, &staterr->counters);
            } else if (event->evtype == EVT_DNS) {
                net = (net_info *)data;
                doDNSMetricName(net->data_type, net->dnsName, &net->totalDuration, &net->counters);
            } else if (event->evtype == EVT_PROTO) {
                proto = (protocol_info *)data;
                doProtocolMetric(proto);
            } else {
                DBG(NULL);
                return;
            }

            free(event);
        }
    }
    ctlFlush(g_ctl);
}
