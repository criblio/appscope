#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>

#include "atomic.h"
#include "com.h"
#include "dbg.h"
#include "dns.h"
#include "httpstate.h"
#include "mtcformat.h"
#include "plattime.h"
#include "search.h"
#include "state.h"
#include "state_private.h"
#include "pcre2.h"
#include "fn.h"
#include "os.h"
#include "utils.h"

#define NET_ENTRIES 1024
#define FS_ENTRIES 1024
#define NUM_ATTEMPTS 100
#define MAX_CONVERT (size_t)256

extern rtconfig g_cfg;

int g_numNinfo = NET_ENTRIES;
int g_numFSinfo = FS_ENTRIES;
int g_http_guard_enabled = TRUE;
uint64_t g_http_guard[NET_ENTRIES];

// These would all be declared static, but the some functions that need
// this data have been moved into report.c.  This is managed with the
// include of state_private.h above.
summary_t g_summary = {{0}};
net_info *g_netinfo;
fs_info *g_fsinfo;
metric_counters g_ctrs = {{0}};
int g_mtc_addr_output = TRUE;
static search_t* g_http_redirect = NULL;

static protocol_def_t *g_tls_protocol_def = NULL;
static protocol_def_t *g_http_protocol_def = NULL;

// Linked list, indexed by channel ID, of net_info pointers used in
// doProtocol() when it's not provided with a valid file descriptor.
static list_t *g_extra_net_info_list = NULL;

#define REDIRECTURL "fluentd"
#define OVERURL "<!DOCTYPE html>\r\n<html>\r\n<head>\r\n<meta http-equiv=\"refresh\" content=\"3; URL='http://cribl.io'\" />\r\n</head>\r\n<body>\r\n<h1>Welcome to Cribl!</h1>\r\n</body>\r\n</html>\r\n\r\n"

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

static void
destroyNetInfo(void *data)
{
    if (!data) return;
    net_info *net = (net_info *)data;

    free(net->http.hdr);
    free(net->http.http2Buf[0].buf);
    free(net->http.http2Buf[1].buf);

    free(net);
}

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

int
get_port_net(net_info *net, int type, control_type_t which) {
    in_port_t port;
    switch (type) {
    case AF_INET:
        if (which == LOCAL) {
            port = ((struct sockaddr_in *)&net->localConn)->sin_port;
        } else {
            port = ((struct sockaddr_in *)&net->remoteConn)->sin_port;
        }
        break;
    case AF_INET6:
        if (which == LOCAL) {
            port = ((struct sockaddr_in6 *)&net->localConn)->sin6_port;
        } else {
            port = ((struct sockaddr_in6 *)&net->remoteConn)->sin6_port;
        }
        break;
    default:
        port = (in_port_t)0;
        break;
    }
    return htons(port);
}

bool
delProtocol(request_t *req)
{
    unsigned int ptype;
    protocol_def_t *protoreq, *protolist;

    if (!req) return FALSE;

    protoreq = req->protocol;

    for (ptype = 0; ptype <= g_prot_sequence; ptype++) {
        if ((protolist = lstFind(g_protlist, ptype)) != NULL) {
            if (strncmp(protoreq->protname, protolist->protname, strlen(protolist->protname)) == 0) {
                // decrement g_prot_sequence?: values are assigned to an entry, used as a key
                lstDelete(g_protlist, ptype);
            }
        }
    }

    if (protoreq && protoreq->protname) free(protoreq->protname);
    if (protoreq) free(protoreq);
    return TRUE;
}

bool
addProtocol(request_t *req)
{
    int errornumber;
    PCRE2_SIZE erroroffset;
    protocol_def_t *proto;

    if (!req) return FALSE;

    proto = req->protocol;

    proto->re = pcre2_compile((PCRE2_SPTR)proto->regex, PCRE2_ZERO_TERMINATED,
                              0, &errornumber, &erroroffset, NULL);

    if (proto->re == NULL) {
        destroyProtEntry(proto);
        return FALSE;
    }

    proto->type = ++g_prot_sequence;

    if (lstInsert(g_protlist, proto->type, proto) == FALSE) {
        destroyProtEntry(proto);
        --g_prot_sequence;
        return FALSE;
    }

    return TRUE;
}

static void
initPayloadDetect()
{
    int errornumber = 0;
    PCRE2_SIZE erroroffset = 0;

    // Setup the TLS protocol-detect regex
    errornumber = 0;
    erroroffset = 0;
    if ((g_tls_protocol_def = calloc(1, sizeof(protocol_def_t))) == NULL) return;
    g_tls_protocol_def->protname = "TLS";
    g_tls_protocol_def->binary = TRUE;
    g_tls_protocol_def->len = PAYLOAD_BYTESRC;
    g_tls_protocol_def->regex = PAYLOAD_REGEX;
    g_tls_protocol_def->re = pcre2_compile((PCRE2_SPTR)g_tls_protocol_def->regex,
                                           PCRE2_ZERO_TERMINATED, 0,
                                           &errornumber, &erroroffset, NULL);
    if (g_tls_protocol_def->re == NULL) {
        goto error;
    }
    g_tls_protocol_def->match_data = pcre2_match_data_create_from_pattern(g_tls_protocol_def->re, NULL);
    if (g_tls_protocol_def->match_data == NULL) {
        goto error;
    }

    // Setup the HTTP protocol-detect regex
    errornumber = 0;
    erroroffset = 0;
    if ((g_http_protocol_def = calloc(1, sizeof(protocol_def_t))) == NULL) return;
    g_http_protocol_def->protname = "HTTP";
    g_http_protocol_def->regex = "(?: HTTP\\/1\\.[0-2]|PRI \\* HTTP\\/2\\.0\r\n\r\nSM\r\n\r\n)";
    g_http_protocol_def->detect = TRUE;
    g_http_protocol_def->re = pcre2_compile((PCRE2_SPTR)g_http_protocol_def->regex,
                                            PCRE2_ZERO_TERMINATED, 0,
                                            &errornumber, &erroroffset, NULL);
    if (g_http_protocol_def->re == NULL) {
        goto error;
    }
    g_http_protocol_def->match_data = pcre2_match_data_create_from_pattern(g_http_protocol_def->re, NULL);
    if (g_http_protocol_def->match_data == NULL) {
        goto error;
    }

    return;

error:
    destroyProtEntry(g_tls_protocol_def);
    g_tls_protocol_def = NULL;
    destroyProtEntry(g_http_protocol_def);
    g_http_protocol_def = NULL;
}

void
initState()
{
    // Per a Read Update & Change (RUC) model; now that the object is ready assign the global
    if ((g_netinfo = (net_info *)calloc(1, sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog(CFG_LOG_ERROR, "ERROR: Constructor:Calloc");
    }

    // Per RUC...
    if ((g_fsinfo = (fs_info *)calloc(1, sizeof(struct fs_info_t) * FS_ENTRIES)) == NULL) {
        scopeLog(CFG_LOG_ERROR, "ERROR: Constructor:Calloc");
    }

    initHttpState();
    // the http guard array is static while the net fs array is dynamically allocated
    // will need to change if we want to re-size at runtime
    memset(g_http_guard, 0, sizeof(g_http_guard));
    {
        // g_http_guard_enable is always false unless
        // SCOPE_HTTP_SERIALIZE_ENABLE is defined and is "true"
        char *spin_env = getenv("SCOPE_HTTP_SERIALIZE_ENABLE");
        g_http_guard_enabled = (spin_env && !strcmp(spin_env, "true"));
    }

    g_http_redirect = searchComp(REDIRECTURL);

    g_protlist = lstCreate(destroyProtEntry);
    initPayloadDetect();

    g_extra_net_info_list = lstCreate(destroyNetInfo);

    initReporting();
}

void
resetState()
{
    memset(&g_ctrs, 0, sizeof(struct metric_counters_t));
}

// DEBUG
#if 0
static void
dumpAddrs(int sd)
{
    in_port_t port;
    char ip[INET6_ADDRSTRLEN];

    inet_ntop(AF_INET,
              &((struct sockaddr_in *)&g_netinfo[sd].localConn)->sin_addr,
              ip, sizeof(ip));
    port = get_port(sd, g_netinfo[sd].localConn.ss_family, LOCAL);
    scopeLog(CFG_LOG_DEBUG, "fd:%d %s:%d LOCAL: %s:%d", sd, __FUNCTION__, __LINE__, ip, port);

    inet_ntop(AF_INET,
              &((struct sockaddr_in *)&g_netinfo[sd].remoteConn)->sin_addr,
              ip, sizeof(ip));
    port = get_port(sd, g_netinfo[sd].remoteConn.ss_family, REMOTE);
    scopeLog(CFG_LOG_DEBUG, "fd:%d %s:%d REMOTE:%s:%d", sd, __FUNCTION__, __LINE__, ip, port);

    if (get_port(sd, g_netinfo[sd].localConn.ss_family, REMOTE) == DNS_PORT) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d DNS", sd);
    }
}
#endif

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

static int
postStatErrState(metric_t stat_err, metric_t type, const char *funcop, const char *pathname)
{
    // something passed in a param that is not a viable address; ltp does this
    if ((stat_err == EVT_ERR) && (errno == EFAULT)) return FALSE;

    int *summarize = NULL;
    switch (type) {
        case NET_ERR_CONN:
        case NET_ERR_RX_TX:
            summarize = &g_summary.net.error;
            break;
        case FS_ERR_OPEN_CLOSE:
        case FS_ERR_READ_WRITE:
        case FS_ERR_STAT:
            summarize = &g_summary.fs.error;
            break;
        case NET_ERR_DNS:
            summarize = &g_summary.net.dnserror;
            break;
        case EVT_STAT:
        case FS_STAT:
            summarize = &g_summary.fs.stat;
            break;
        default:
            break;
    }

    // Bail if we don't need to post
    int mtc_needs_reporting = summarize && !*summarize;
    int need_to_post =
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) ||
        (mtcEnabled(g_mtc) && mtc_needs_reporting);
    if (!need_to_post) return FALSE;

    size_t len = sizeof(struct stat_err_info_t);
    stat_err_info *sep = calloc(1, len);
    if (!sep) return FALSE;

    sep->evtype = stat_err;
    sep->data_type = type;

    if (pathname) {
        strncpy(sep->name, pathname, strnlen(pathname, sizeof(sep->name)));
    }

    if (funcop) {
        strncpy(sep->funcop, funcop, strnlen(funcop, sizeof(sep->funcop)));
    }

    memmove(&sep->counters, &g_ctrs, sizeof(g_ctrs));

    cmdPostEvent(g_ctl, (char *)sep);

    return mtc_needs_reporting;
}

static int
postFSState(int fd, metric_t type, fs_info *fs, const char *funcop, const char *pathname)
{
    int *summarize = NULL;
    switch (type) {
        case FS_READ:
        case FS_WRITE:
        case FS_DURATION:
            summarize = &g_summary.fs.read_write;
            break;
        case FS_OPEN:
        case FS_CLOSE:
            summarize = &g_summary.fs.open_close;
            break;
        case FS_SEEK:
            summarize = &g_summary.fs.seek;
            break;
        default:
            break;
    }

    // Bail if we don't need to post
    int mtc_needs_reporting = summarize && !*summarize;
    int need_to_post =
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) ||
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS) ||
        (mtcEnabled(g_mtc) && mtc_needs_reporting);
    if (!need_to_post) return FALSE;

    size_t len = sizeof(struct fs_info_t);
    fs_info *fsp = calloc(1, len);
    if (!fsp) return FALSE;

    memmove(fsp, fs, len);
    fsp->fd = fd;
    fsp->evtype = EVT_FS;
    fsp->data_type = type;

    if (pathname && (fs->path[0] == '\0')) {
        strncpy(fsp->path, pathname, strnlen(pathname, sizeof(fsp->path)));
    }

    if (funcop && (fs->funcop[0] == '\0')) {
        strncpy(fsp->funcop, funcop, strnlen(funcop, sizeof(fsp->funcop)));
    }

    cmdPostEvent(g_ctl, (char *)fsp);

    return mtc_needs_reporting;
}

static int
postDNSState(int fd, metric_t type, net_info *net, uint64_t duration, const char *domain)
{
    // Bail if we don't need to post
    int mtc_needs_reporting = !g_summary.net.dns;
    int need_to_post =
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) ||
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_DNS) ||
        (mtcEnabled(g_mtc) && mtc_needs_reporting);
    if (!need_to_post) return FALSE;

    size_t len = sizeof(struct net_info_t);
    net_info *netp = calloc(1, len);
    if (!netp) return FALSE;

    if (net) memmove(netp, net, len);
    netp->fd = fd;
    netp->evtype = EVT_DNS;
    netp->data_type = type;

    if (duration > 0) {
        addToInterfaceCounts(&netp->totalDuration, duration);
    }

    if (domain) {
        strncpy(netp->dnsName, domain, strnlen(domain, sizeof(netp->dnsName)));
    }

    memmove(&netp->counters, &g_ctrs, sizeof(g_ctrs));

    cmdPostEvent(g_ctl, (char *)netp);

    return mtc_needs_reporting;
}

static int
postNetState(int fd, metric_t type, net_info *net)
{
    int *summarize = NULL;
    switch (type) {
        case OPEN_PORTS:
        case NET_CONNECTIONS:
        case CONNECTION_DURATION:
            summarize = &g_summary.net.open_close;
            break;
        case NETRX:
        case NETTX:
            summarize = &g_summary.net.rx_tx;
            break;
        default:
            break;
    }

    // Bail if we don't need to post
    int mtc_needs_reporting = summarize && !*summarize;
    int need_to_post =
        // if raw metrics are enabled
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) ||
        // if NET events are enabled
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_NET) ||
        // if it's a closed HTTP channel (so we can cleanup)
        (type==CONNECTION_DURATION && net && (net->http.version[0] || net->http.version[1])) ||
        // if metrics are enabled and it's one we report
        (mtcEnabled(g_mtc) && mtc_needs_reporting);
    if (!need_to_post) return FALSE;

    size_t len = sizeof(struct net_info_t);
    net_info *netp = calloc(1, len);
    if (!netp) return FALSE;

    memmove(netp, net, len);
    netp->fd = fd;
    netp->evtype = EVT_NET;
    netp->data_type = type;

    cmdPostEvent(g_ctl, (char *)netp);
    return mtc_needs_reporting;
}

void
doUpdateState(metric_t type, int fd, ssize_t size, const char *funcop, const char *pathname)
{
    switch (type) {
    case OPEN_PORTS:
    {
        if (!checkNetEntry(fd)) break;
        if (size < 0) {
            subFromInterfaceCounts(&g_ctrs.openPorts, labs(size));
        } else if (size > 0) {
            addToInterfaceCounts(&g_ctrs.openPorts, size);
        }

        if (size && !g_netinfo[fd].startTime) {
            g_netinfo[fd].startTime = getTime();
        }
        if (postNetState(fd, type, &g_netinfo[fd])) {
            // Don't reset the info.  It's a gauge.
        }
        break;
    }

    case NET_CONNECTIONS:
    {
        if (!checkNetEntry(fd)) break;
        counters_element_t* value = NULL;

        if (g_netinfo[fd].type == SOCK_STREAM) {
            value = &g_ctrs.netConnectionsTcp;
        } else if (g_netinfo[fd].type == SOCK_DGRAM) {
            value = &g_ctrs.netConnectionsUdp;
        } else {
            value = &g_ctrs.netConnectionsOther;
        }

        if (size < 0) {
            subFromInterfaceCounts(value, labs(size));
        } else if (size > 0) {
            addToInterfaceCounts(value, size);
        }

        if (size && !g_netinfo[fd].startTime) {
            g_netinfo[fd].startTime = getTime();
        }
        if (postNetState(fd, type, &g_netinfo[fd])) {
            // Don't reset the info.  It's a gauge.
        }
        break;
    }

    case CONNECTION_DURATION:
    {
        if (!checkNetEntry(fd)) break;
        uint64_t new_duration = 0ULL;
        if (g_netinfo[fd].startTime != 0ULL) {
            new_duration = getDuration(g_netinfo[fd].startTime);
            g_netinfo[fd].startTime = 0ULL;
        }

        if (new_duration) {
            addToInterfaceCounts(&g_netinfo[fd].numDuration, 1);
            addToInterfaceCounts(&g_netinfo[fd].totalDuration, new_duration);
            addToInterfaceCounts(&g_ctrs.connDurationNum, 1);
            addToInterfaceCounts(&g_ctrs.connDurationTotal, new_duration);
        }

        if ((g_netinfo[fd].rxBytes.evt > 0) || (g_netinfo[fd].txBytes.evt > 0) ||
            (g_netinfo[fd].rxBytes.mtc > 0) || (g_netinfo[fd].txBytes.mtc > 0)) {
            postNetState(fd, type, &g_netinfo[fd]);
            atomicSwapU64(&g_netinfo[fd].numDuration.mtc, 0);
            atomicSwapU64(&g_netinfo[fd].totalDuration.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.connDurationNum, 1);
            //subFromInterfaceCounts(&g_ctrs.connDurationTotal, new_duration);
        }
        atomicSwapU64(&g_netinfo[fd].numDuration.evt, 0);
        atomicSwapU64(&g_netinfo[fd].totalDuration.evt, 0);
        break;
    }

    case CONNECTION_OPEN:
    {
        if (checkNetEntry(fd) && ctlEvtSourceEnabled(g_ctl, CFG_SRC_NET) &&
            (((g_netinfo[fd].addrSetRemote == TRUE) && (g_netinfo[fd].addrSetLocal == TRUE)) ||
             (funcop && !strncmp(funcop, "dup", 3)))) {
                postNetState(fd, type, &g_netinfo[fd]);
        }
        break;
    }

    case NETRX:
    {
        if (!checkNetEntry(fd)) break;
        addToInterfaceCounts(&g_netinfo[fd].numRX, 1);
        addToInterfaceCounts(&g_netinfo[fd].rxBytes, size);
        sock_summary_bucket_t bucket = getNetRxTxBucket(&g_netinfo[fd]);
        addToInterfaceCounts(&g_ctrs.netrxBytes[bucket], size);
        if (postNetState(fd, type, &g_netinfo[fd])) {
            atomicSwapU64(&g_netinfo[fd].numRX.mtc, 0);
            atomicSwapU64(&g_netinfo[fd].rxBytes.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.netrxBytes, size);
        }
        //atomicSwapU64(&g_netinfo[fd].numRX.evt, 0);
        //atomicSwapU64(&g_netinfo[fd].rxBytes.evt, 0);
        break;
    }

    case NETTX:
    {
        if (!checkNetEntry(fd)) break;
        addToInterfaceCounts(&g_netinfo[fd].numTX, 1);
        addToInterfaceCounts(&g_netinfo[fd].txBytes, size);
        sock_summary_bucket_t bucket = getNetRxTxBucket(&g_netinfo[fd]);
        addToInterfaceCounts(&g_ctrs.nettxBytes[bucket], size);
        if (postNetState(fd, type, &g_netinfo[fd])) {
            atomicSwapU64(&g_netinfo[fd].numTX.mtc, 0);
            atomicSwapU64(&g_netinfo[fd].txBytes.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.nettxBytes, size);
        }
        //atomicSwapU64(&g_netinfo[fd].numTX.evt, 0);
        //atomicSwapU64(&g_netinfo[fd].txBytes.evt, 0);
        break;
    }

    case DNS:
    {
        int rc;

        // on DNS resp only inc events
        if (size > 0) {
            atomicAddU64(&g_ctrs.numDNS.evt, 1);
        } else {
            addToInterfaceCounts(&g_ctrs.numDNS, 1);
        }

        if (checkNetEntry(fd)) {
            rc = postDNSState(fd, type, &g_netinfo[fd], (uint64_t)size, pathname);
        } else {
            rc = postDNSState(fd, type, NULL, (uint64_t)size, pathname);
        }

        if (rc && (size == 0)) atomicSubU64(&g_ctrs.numDNS.mtc, 1);
        atomicSubU64(&g_ctrs.numDNS.evt, 1);
        break;
    }

    case DNS_DURATION:
    {
        int rc;

        addToInterfaceCounts(&g_ctrs.dnsDurationNum, 1);
        addToInterfaceCounts(&g_ctrs.dnsDurationTotal, 0);

        if (checkNetEntry(fd)) {
            rc = postDNSState(fd, type, &g_netinfo[fd], size, pathname);
        } else {
            rc = postDNSState(fd, type, NULL, size, pathname);
        }

        if (rc) {
            atomicSwapU64(&g_ctrs.dnsDurationNum.mtc, 0);
            atomicSwapU64(&g_ctrs.dnsDurationTotal.mtc, 0);
        }

        atomicSwapU64(&g_ctrs.dnsDurationNum.evt, 0);
        atomicSwapU64(&g_ctrs.dnsDurationTotal.evt, 0);
        break;
    }

    case FS_DURATION:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numDuration, 1);
        addToInterfaceCounts(&g_fsinfo[fd].totalDuration, size);
        addToInterfaceCounts(&g_ctrs.fsDurationNum, 1);
        addToInterfaceCounts(&g_ctrs.fsDurationTotal, size);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numDuration.mtc, 0);
            atomicSwapU64(&g_fsinfo[fd].totalDuration.mtc, 0);
        }
        //atomicSwapU64(&g_fsinfo[fd].numDuration.evt, 0);
        //atomicSwapU64(&g_fsinfo[fd].totalDuration.evt, 0);
        break;
    }

    case FS_READ:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numRead, 1);
        addToInterfaceCounts(&g_fsinfo[fd].readBytes, size);
        addToInterfaceCounts(&g_ctrs.readBytes, size);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numRead.mtc, 0);
            atomicSwapU64(&g_fsinfo[fd].readBytes.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.readBytes, size);
        }
        //atomicSwapU64(&g_fsinfo[fd].numRead.evt, 0);
        //atomicSwapU64(&g_fsinfo[fd].readBytes.evt, 0);
        break;
    }

    case FS_WRITE:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numWrite, 1);
        addToInterfaceCounts(&g_fsinfo[fd].writeBytes, size);
        addToInterfaceCounts(&g_ctrs.writeBytes, size);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numWrite.mtc, 0);
            atomicSwapU64(&g_fsinfo[fd].writeBytes.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.writeBytes, size);
        }
        //atomicSwapU64(&g_fsinfo[fd].numWrite.evt, 0);
        //atomicSwapU64(&g_fsinfo[fd].writeBytes.evt, 0);
        break;
    }

    case FS_OPEN:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numOpen, 1);
        addToInterfaceCounts(&g_ctrs.numOpen, 1);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numOpen.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.numOpen, 1);
        }
        atomicSwapU64(&g_fsinfo[fd].numOpen.evt, 0);
        break;
    }

    case FS_CLOSE:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numClose, 1);
        addToInterfaceCounts(&g_ctrs.numClose, 1);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numClose.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.numClose, 1);
        }
        atomicSwapU64(&g_fsinfo[fd].numClose.evt, 0);
        break;
    }

    case FS_SEEK:
    {
        if (!checkFSEntry(fd)) break;
        addToInterfaceCounts(&g_fsinfo[fd].numSeek, 1);
        addToInterfaceCounts(&g_ctrs.numSeek, 1);
        if (postFSState(fd, type, &g_fsinfo[fd], funcop, pathname)) {
            atomicSwapU64(&g_fsinfo[fd].numSeek.mtc, 0);
            //subFromInterfaceCounts(&g_ctrs.numSeek, 1);
        }
        atomicSwapU64(&g_fsinfo[fd].numSeek.evt, 0);
        break;
    }

    case NET_ERR_CONN:
    {
        addToInterfaceCounts(&g_ctrs.netConnectErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.netConnectErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.netConnectErrors.evt, 0);
        break;
    }

    case NET_ERR_RX_TX:
    {
        addToInterfaceCounts(&g_ctrs.netTxRxErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.netTxRxErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.netTxRxErrors.evt, 0);
        break;
    }

    case FS_ERR_OPEN_CLOSE:
    {
        addToInterfaceCounts(&g_ctrs.fsOpenCloseErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.fsOpenCloseErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.fsOpenCloseErrors.evt, 0);
        break;
    }

    case FS_ERR_READ_WRITE:
    {
        addToInterfaceCounts(&g_ctrs.fsRdWrErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.fsRdWrErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.fsRdWrErrors.evt, 0);
        break;
    }

    case FS_ERR_STAT:
    {
        addToInterfaceCounts(&g_ctrs.fsStatErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.fsStatErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.fsStatErrors.evt, 0);
        break;
    }

    case NET_ERR_DNS:
    {
        addToInterfaceCounts(&g_ctrs.netDNSErrors, 1);
        if (postStatErrState(EVT_ERR, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.netDNSErrors.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.netDNSErrors.evt, 0);
        break;
    }

    case FS_STAT:
    {
        addToInterfaceCounts(&g_ctrs.numStat, 1);
        if (postStatErrState(EVT_STAT, type, funcop, pathname)) {
            atomicSwapU64(&g_ctrs.numStat.mtc, 0);
        }
        atomicSwapU64(&g_ctrs.numStat.evt, 0);
        break;
    }
    default:
         return;
    }
}

static bool
setProtocol(int sockfd, protocol_def_t *protoDef, net_info *net, char *buf, size_t len)
{
    char *data, *cpdata = NULL;
    pcre2_match_data *match_data;
    protocol_info *proto;
    bool ret = FALSE;

    // HACK: See issue #600.
    //   We see what appear to be invalid "runt" payloads sometimes from our
    //   interpositions of Java's SSL/TLS routines. In the example log below,
    //   notice the first payload is a single `H` byte while the second is the
    //   actual request.
    //
    //   DEBUG: Java_sun_security_ssl_SSLEngineImpl_unwrap
    //   DEBUG: doProtocol(id=-63, fd=93, len=1, src=TLSRX, dtyp=BUF) TLS=TRUE PROTO=PENDING (1 bytes)
    //     0000:  48                                                H
    //   DEBUG: fd:93 Protocol detection result is FALSE
    //   DEBUG: Java_sun_security_ssl_SSLEngineImpl_unwrap
    //   DEBUG: doProtocol(id=-63, fd=93, len=79, src=TLSRX, dtyp=BUF) TLS=TRUE PROTO=FALSE (64 bytes)
    //     0000:  48 45 41 44 20 2f 20 48  54 54 50 2f 31 2e 31 0d  HEAD / HTTP/1.1.
    //     0010:  0a 55 73 65 72 2d 41 67  65 6e 74 3a 20 63 75 72  .User-Agent: cur
    //     0020:  6c 2f 37 2e 32 39 2e 30  0d 0a 48 6f 73 74 3a 20  l/7.29.0..Host:
    //     0030:  6c 6f 63 61 6c 68 6f 73  74 3a 38 34 34 33 0d 0a  localhost:8443..
    //
    // In this case, as a temporary hack until we can fix the Java
    // interpositions, we're leaving protocol detection PENDING if we don't get
    // enough buffer to match. If we're not doing binary matching (as is the
    // case with HTTP) then we don't get a length to use here. As a complete
    // hack, we simply require more than one byte. Ugly...
    if ((protoDef->len > 0 && protoDef->len > len) || (protoDef->len <= 0 && len < 2)) {
        scopeLogDebug("DEBUG: skipping protocol detection on runt payload");
        return FALSE;
    }

    // nothing we can do; don't risk reading past end of a buffer
    size_t cvlen = (len < MAX_CONVERT) ? len : MAX_CONVERT;
    if (((len <= 0) && (protoDef->len <= 0)) ||   // no len
        !protoDef->re ||                          // no regex
        (protoDef->len > cvlen)) {                // not enough buf for protoDef->len
        if (net) net->protoDetect = DETECT_FALSE;
        return FALSE;
    }

    // precedence to a len defined with the protocol definition
    // if a len was not provided in the definition use the one passed
    // therefore, len is now protoDef->len
    if (protoDef->len > 0) {
        cvlen = protoDef->len;
    }

    if (protoDef->binary == FALSE) {
        // non-binary data so we can preg-match against it as is.
        data = buf;
    } else {
        // otherwise, hexdump it
        int i;
        size_t alen = (cvlen * 2) + 1;

        if ((cpdata = calloc(1, alen)) == NULL) {
            if (net) net->protoDetect = DETECT_FALSE;
            return FALSE;
        }

        for (i = 0; i < cvlen; i++) {
            snprintf(&cpdata[i<<1], 3, "%02x", (unsigned char)buf[i]);
        }

        data = cpdata;
        cvlen = cvlen * 2;
    }

    match_data = pcre2_match_data_create_from_pattern(protoDef->re, NULL);
    if (pcre2_match_wrapper(protoDef->re, (PCRE2_SPTR)data, (PCRE2_SIZE)cvlen, 0, 0,
                            match_data, NULL) > 0) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d detected %s", sockfd, protoDef->protname);

        if (net) {
            net->protoDetect = DETECT_TRUE;
            net->protoProtoDef = protoDef;
        }

        if (protoDef->detect && ctlEvtSourceEnabled(g_ctl, CFG_SRC_NET)) {
            if ((proto = calloc(1, sizeof(struct protocol_info_t))) == NULL)
            {
                if (cpdata)
                    free(cpdata);
                if (match_data)
                    pcre2_match_data_free(match_data);
                return FALSE;
            }
            proto->evtype = EVT_PROTO;
            proto->ptype = EVT_DETECT;
            proto->len = sizeof(protocol_def_t);
            proto->fd = sockfd;
            if (net) proto->uid = net->uid;
            proto->data = (char *)strdup(protoDef->protname);
            cmdPostEvent(g_ctl, (char *)proto);
        }

        ret = TRUE; // matched
    } else {
        // the regex for protocol DID NOT match
        if (net) net->protoDetect = DETECT_FALSE;
    }

    if (match_data) pcre2_match_data_free(match_data);
    if (cpdata) free(cpdata);

    return ret;
}

// This is just calling setProtocol() different ways depending on buffer type.
static bool
setProtocolByType(int sockfd, protocol_def_t *protoDef, net_info *net, char *buf, size_t len, src_data_t dtype)
{
    bool ret = FALSE;

    if (dtype == BUF) {
        // simple buffer, pass it through
        ret = ret || setProtocol(sockfd, protoDef, net, buf, len);
    } else if (dtype == MSG) {
        // buffer is a msghdr for sendmsg/recvmsg
        int i;
        struct msghdr *msg = (struct msghdr *)buf;
        struct iovec *iov;
        for (i = 0; i < msg->msg_iovlen; i++) {
            iov = &msg->msg_iov[i];
            if (iov && iov->iov_base && (iov->iov_len > 0)) {
                // check every vector?
                ret = ret || setProtocol(sockfd, protoDef, net, iov->iov_base, iov->iov_len);
            }
        }
    } else if ( dtype == IOV) {
        // buffer is an iovec, len is the iovcnt
        int i;
        struct iovec *iov = (struct iovec *)buf;
        for (i = 0; i < len; i++) {
            if (iov[i].iov_base && (iov[i].iov_len > 0)) {
                // check every vector?
                ret = ret || setProtocol(sockfd, protoDef, net, iov[i].iov_base, iov[i].iov_len);
            }
        }
    } else {
        DBG(NULL); // how do we even get here?
    }

    return ret;
}

static int
extractPayload(int sockfd, net_info *net, void *buf, size_t len, metric_t src, src_data_t dtype)
{
    if (!buf || (len <= 0)) {
        DBG(NULL); // why would we ever get here?
        return -1;
    }

    payload_info *pinfo = calloc(1, sizeof(struct payload_info_t));
    if (!pinfo) {
        return -1;
    }

    if (dtype == BUF) {
        pinfo->data = calloc(1, len);
        if (!pinfo->data) {
            free(pinfo);
            return -1;
        }
        memmove(pinfo->data, buf, len);
    } else if (dtype == MSG) {
        int i;
        size_t blen = 0;
        struct msghdr *msg = (struct msghdr *)buf;
        struct iovec *iov;
        for (i = 0; i < msg->msg_iovlen; i++) {
            iov = &msg->msg_iov[i];
            if (iov && iov->iov_base && (iov->iov_len > 0)) {
                char *temp = realloc(pinfo->data, blen + iov->iov_len);
                if (!temp) {
                    if (pinfo->data) free(pinfo->data);
                    free(pinfo);
                    return -1;
                }

                pinfo->data = temp;
                memmove(&pinfo->data[blen], iov->iov_base, iov->iov_len);
                blen += iov->iov_len;
            }
        }
        len = blen;
    } else if (dtype == IOV) {
        int i;
        size_t blen = 0;
        struct iovec *iov = (struct iovec *)buf;
        for (i = 0; i < len; i++) {
            if (iov[i].iov_base && (iov[i].iov_len > 0)) {
                char *temp = realloc(pinfo->data, blen + iov[i].iov_len);
                if (!temp) {
                    if (pinfo->data) free(pinfo->data);
                    free(pinfo);
                    return -1;
                }

                pinfo->data = temp;
                memmove(&pinfo->data[blen], iov[i].iov_base, iov[i].iov_len);
                blen += iov[i].iov_len;
            }
        }
        len = blen;
    } else {
        // no data, no need to continue
        free(pinfo);
        return -1;
    }

    if (net) {
        memmove(&pinfo->net, net, sizeof(net_info));
    } else {
        pinfo->net.active = 0;
    }

    pinfo->evtype = EVT_PAYLOAD;
    pinfo->src = src;
    pinfo->sockfd = sockfd;
    pinfo->len = len;

    if (net && net->tlsDetect) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d posting TLS payload", sockfd);
    } else if (net && net->protoProtoDef) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d posting %s payload", sockfd, net->protoProtoDef->protname);
    } else {
        scopeLog(CFG_LOG_DEBUG, "fd:%d posting payload", sockfd);
    }

    if (cmdPostPayload(g_ctl, (char *)pinfo) == -1) {
        if (pinfo->data) free(pinfo->data);
        if (pinfo) free(pinfo);
        return -1;
    }

    return 0;
}

/**
 * Apply our TLS protocol-detector to a channel.
 * 
 * Sets net->tlsDetect and net->tlsProtoDef.
 */
static void
detectTLS(int sockfd, net_info *net, void *buf, size_t len, metric_t src, src_data_t dtype)
{
#if 1
    unsigned char *data = buf;
    if (len >= 5 && data[0] == 0x16 && data[1] == 0x03 && data[2] >= 0x00 && data[2] <= 0x03) {
        // it's TLS
        net->tlsDetect = DETECT_TRUE;
        net->tlsProtoDef = g_tls_protocol_def;
    } else if (len >= 5 && (data[0]&0xf0) == 0x80 && data[2] == 0x01) {
        // it SSL2
        net->tlsDetect = DETECT_TRUE;
        net->tlsProtoDef = g_tls_protocol_def;
    } else {
        // not TLS/SSL
        net->tlsDetect = DETECT_FALSE;
    }
#else
    int rc;
    unsigned char *data = buf;
    unsigned int ptype;
    protocol_def_t *tls_proto_def = g_tls_protocol_def; // use ours by default

    // Look for an overridden TLS entry from the protocol detector configs
    for (ptype = 0; ptype <= g_prot_sequence; ptype++)
    {
        protocol_def_t *tmp_proto_def;
        if ((tmp_proto_def = lstFind(g_protlist, ptype))
            && (strcmp(tmp_proto_def->protname, "TLS") == 0)) {
            // Use the user-provided one instead of ours
            tls_proto_def = tmp_proto_def;
            break;
        }
    }

    // Extract some of the payload to a hex string since it's binary.
    int i;
    size_t alen = (tls_proto_def->len * 2) + 1;
    char cpdata[alen];
    memset(cpdata, 0, alen);
    for (i = 0; i < tls_proto_def->len; i++)
    {
        snprintf(&cpdata[i << 1], 3, "%02x", data[i]);
    }

    // Apply the regex to the hex-string payload
    pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(tls_proto_def->re, NULL);
    if ((rc = pcre2_match_wrapper(tls_proto_def->re, (PCRE2_SPTR)cpdata,
                                  (PCRE2_SIZE)alen, 0, 0,
                                  match_data, NULL)) > 0)
    {
        // matched, set the detect-state to TRUE
        net->tlsDetect = DETECT_TRUE;
        net->tlsProtoDef = tls_proto_def;
        scopeLog(CFG_LOG_DEBUG, "fd:%d detected TLS", sockfd);
        if (tls_proto_def->detect) {
            // TODO send TLS protocol-detect event
        }
    }
    else
    {
        // didn't match, set the detect-state to FALSE
        net->tlsDetect = DETECT_FALSE;
        if (rc != PCRE2_ERROR_NOMATCH)
        {
            DBG(NULL);
            scopeLog(CFG_LOG_DEBUG, "fd:%d doProtocol: TLS regex failed", sockfd);
        }
    }
    pcre2_match_data_free(match_data);
#endif
}

static void
detectProtocol(int sockfd, net_info *net, void *buf, size_t len, metric_t src, src_data_t dtype)
{
    unsigned int ptype;
    protocol_def_t *protoDef;
    bool sawHTTP = FALSE;

    // No need to try protocol detection in raw TLS data
    if (net && net->tlsDetect == DETECT_TRUE     // TLS detected already
        && (src == NETRX || src == NETTX)) {     // raw payload
        return;
    }

    // Check first against the protocol entries in the configs.
    for (ptype = 0; ptype <= g_prot_sequence; ptype++) {
        if ((protoDef = lstFind(g_protlist, ptype)) != NULL) {
            if (strcasecmp(protoDef->protname, "HTTP") == 0) {
                // Remember we saw an HTTP entry in the configs.
                sawHTTP = TRUE;
            }
            if (setProtocolByType(sockfd, protoDef, net, buf, len, dtype)) {
                // We're done since it matched.
                return;
            }
        }
    }

    // Try our HTTP detection if we've not seen one 
    if (!sawHTTP) {
        setProtocolByType(sockfd, g_http_protocol_def, net, buf, len, dtype);
    }
}

// Alternative to getNetEntry() that returns a net_info for the given channel
// ID instead of for a socket descriptor. We fallback to using this when we
// can't get the descriptor in TLS/SSL read/write operations.
static net_info *
getChannelNetEntry(uint64_t id)
{
    net_info *net = lstFind(g_extra_net_info_list, id);
    if (!net) {
        net = calloc(1, sizeof(net_info));
        if (!net) {
            scopeLogError("ERROR: failed to allocate channel's net_info");
            DBG(NULL);
        } else {
            if (lstInsert(g_extra_net_info_list, id, net) != TRUE) {
                free(net);
                net = NULL;
                scopeLogError("ERROR: failed to save channel's net_info");
                DBG(NULL);
            } else {
                // populate the new net_info
                net->active = TRUE;
                net->type = SOCK_STREAM; // assumption needed for doHttp()
                //net->localConn.ss_family = ???;
                net->uid = id;
            }
        }
    }
    return net;
}

int
doProtocol(uint64_t id, int sockfd, void *buf, size_t len, metric_t src, src_data_t dtype)
{
    // HACK: See issue #600
    //   We're ignoring payloads where we don't get a reasonable ID or FD which
    //   seems to happen with the interpositions of Java's SSL operations.
    if ((int64_t)id <= 0 && sockfd < 0) {
        return 0;
    }

    // Find the net_info for the channel
    net_info *net = getNetEntry(sockfd);    // first try by descriptor
    if (!net) net = getChannelNetEntry(id); // fallback to using channel ID

    scopeLogHexDebug(buf, len > 64 ? 64 : len, // limit hexdump to 64
            "DEBUG: doProtocol(id=%ld, fd=%d, len=%ld, src=%s, dtyp=%s) TLS=%s PROTO=%s",
            id, sockfd, len,
            src == NETRX ? "NETRX" :
            src == NETTX ? "NETTX" :
            src == TLSRX ? "TLSRX" :
            src == TLSTX ? "TLSTX" : "?",
            dtype == BUF  ? "BUF" :
            dtype == MSG  ? "MSG" :
            dtype == IOV  ? "IOV" :
            dtype == NONE ? "NONE" : "?",
            net == NULL ? "NULL" :
            net->tlsDetect == DETECT_PENDING ? "PENDING" :
            net->tlsDetect == DETECT_TRUE    ? "TRUE" :
            net->tlsDetect == DETECT_FALSE   ? "FALSE" : "INVALID",
            net == NULL ? "NULL" :
            net->protoDetect == DETECT_PENDING ? "PENDING" :
            net->protoDetect == DETECT_TRUE    ? "TRUE" :
            net->protoDetect == DETECT_FALSE   ? "FALSE" : "INVALID"
            );

    // Ignore empty payloads that should have been blocked by our interpositions
    if (!len) {
        scopeLogDebug("DEBUG: fd:%d ignoring empty payload", sockfd);
        return 0;
    }

    // HACK: See issue #600
    //   We ignore all-zeros payloads we get from our intepositions of Java's
    //   SSL operations.
    if (net && net->protoDetect == DETECT_PENDING) {
        bool foundNonZero = FALSE;
        for (size_t n = 0; n < len; ++n) {
            if (((char*)buf)[n] != 0) {
                foundNonZero = TRUE;
                break;
            }
        }
        if (!foundNonZero) {
            scopeLogDebug("DEBUG: fd:%d ignoring all-zero payload", sockfd);
            return 0;
        }
    }

    // Do TLS detection if not already done
    if (net && net->tlsDetect == DETECT_PENDING) {
        detectTLS(sockfd, net, buf, len, src, dtype);
    }

    // Only process unencrypted payloads
    if ((net && net->tlsDetect == DETECT_FALSE) || (src == TLSTX || src == TLSRX)) {

        // Do protocol-detection if not already done
        if (net && net->protoDetect == DETECT_PENDING) {
            detectProtocol(sockfd, net, buf, len, src, dtype);
        }

        // Send payloads if enabled globally or by the detected protocol
        if (cfgPayEnable(g_cfg.staticfg)
            || (net && net->protoProtoDef && net->protoProtoDef->payload)) {
            extractPayload(sockfd, net, buf, len, src, dtype);
        }

        // Process HTTP if detected and events are enabled
        if (net && net->protoProtoDef
                && !strcasecmp(net->protoProtoDef->protname, "HTTP")
                && cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_HTTP)) {
            doHttp(id, sockfd, net, buf, len, src, dtype);
        }
    }

    return 0;
}

void
setVerbosity(unsigned verbosity)
{
    summary_t *summarize = &g_summary;
    g_mtc_addr_output = verbosity >= DEFAULT_MTC_IPPORT_VERBOSITY;

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

bool
checkNetEntry(int fd)
{
    if (g_netinfo && (fd >= 0) && (fd < g_numNinfo)) {
        return TRUE;
    }

    return FALSE;
}

bool
checkFSEntry(int fd)
{
    if (g_fsinfo && (fd >= 0) && (fd < g_numFSinfo)) {
        return TRUE;
    }

    return FALSE;
}

net_info *
getNetEntry(int fd)
{
    if (g_netinfo && (fd >= 0) && (fd < g_numNinfo) &&
        g_netinfo[fd].active) {
        return &g_netinfo[fd];
    }
    return NULL;
}

fs_info *
getFSEntry(int fd)
{
    if (g_fsinfo && (fd >= 0) && (fd < g_numFSinfo) &&
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
setRemoteClose(int sd, int err)
{
    net_info *net;

    if ((err == EPIPE) && ((net = getNetEntry(sd)))) {
        net->remoteClose = TRUE;
    }
}

fs_content_type_t
getFSContentType(int fd)
{
    struct fs_info_t *fs = getFSEntry(fd);
    if (fs) {
       return fs->content_type;
    }
    DBG(NULL);
    return FS_CONTENT_UNKNOWN;
}

void
setFSContentType(int fd, fs_content_type_t type)
{
    struct fs_info_t *fs = getFSEntry(fd);
    if (fs) {
        fs->content_type = type;
    } else {
        DBG(NULL);
    }
}

void
addSock(int fd, int type, int family)
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
        if ((fd > g_numNinfo) && (fd < MAX_FDS))  {
            int increase;
            net_info *temp;

            if (fd < (MAX_FDS / 2)) {
                increase = MAX_FDS / 2;
            } else {
                increase = MAX_FDS;
            }

            // Need to realloc
            if ((temp = realloc(g_netinfo, sizeof(struct net_info_t) * increase)) == NULL) {
                scopeLog(CFG_LOG_ERROR, "fd:%d ERROR: addSock:realloc", fd);
                DBG("re-alloc on Net table failed");
            } else {
                memset(&temp[g_numNinfo], 0, sizeof(struct net_info_t) * (increase - g_numNinfo));
                g_numNinfo = increase;
                g_netinfo = temp;
            }
        }
*/

        memset(&g_netinfo[fd], 0, sizeof(struct net_info_t));
        g_netinfo[fd].active = TRUE;
        g_netinfo[fd].type = type;
        g_netinfo[fd].localConn.ss_family = family;
        g_netinfo[fd].uid = getTime();
#ifdef __linux__
        // Clear these bits so comparisons of type will work
        g_netinfo[fd].type &= ~SOCK_CLOEXEC;
        g_netinfo[fd].type &= ~SOCK_NONBLOCK;
#endif // __linux__
    }
}

int
doBlockConnection(int fd, const struct sockaddr *addr_arg)
{
    in_port_t port;

    if (g_cfg.blockconn == DEFAULT_PORTBLOCK) return 0;

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

    if (g_cfg.blockconn == htons(port)) {
        scopeLog(CFG_LOG_INFO, "fd:%d doBlockConnection: blocked connection", fd);
        return 1;
    }

    return 0;
}

void
doSetConnection(int sd, const struct sockaddr *addr, socklen_t len, control_type_t endp)
{
    net_info *net;

    if (!addr || (len <= 0)) {
        return;
    }

    // Should we check for at least the size of sockaddr_in?
    if (((net = getNetEntry(sd)) != NULL) && addr && (len > 0)) {
        if (endp == LOCAL) {
            if ((net->type == SOCK_STREAM) && (net->addrSetLocal == TRUE)) return;
            memmove(&g_netinfo[sd].localConn, addr, len);
            if (net->type == SOCK_STREAM) net->addrSetLocal = TRUE;
        } else {
            if ((net->type == SOCK_STREAM) && (net->addrSetRemote == TRUE)) return;
            memmove(&g_netinfo[sd].remoteConn, addr, len);
            if (net->type == SOCK_STREAM) net->addrSetRemote = TRUE;
        }

        if (addrIsNetDomain(&g_netinfo[sd].localConn)) {
            doUpdateState(CONNECTION_OPEN, sd, 1, NULL, NULL);
        }
    }
}

int
doSetAddrs(int sockfd)
{
    struct sockaddr_storage addr;
    socklen_t addrlen = sizeof(struct sockaddr_storage);
    net_info *net;

    // Only do this if output is enabled
    int need_to_track_addrs =
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) ||
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_NET) ||
        ctlEvtSourceEnabled(g_ctl, CFG_SRC_HTTP) ||
        (mtcEnabled(g_mtc) && g_mtc_addr_output) ||
        ctlPayEnable(g_ctl);
    if (!need_to_track_addrs) return 0;

    /*
     * Do this for TCP, UDP or UNIX sockets
     * Not doing connection details for other socket types
     * If a TCP socket only set the addrs once
     * It's possible for UDP sockets to change addrs as needed
     */
    if ((net = getNetEntry(sockfd)) == NULL) return 0;

    if (addrIsUnixDomain(&net->remoteConn) ||
        addrIsUnixDomain(&net->localConn)) {
        if (net->addrSetUnix == TRUE) return 0;
        doUnixEndpoint(sockfd, net);
        net->addrSetUnix = TRUE;
        doUpdateState(CONNECTION_OPEN, sockfd, 1, NULL, NULL);
        return 0;
    }

    if ((net->type == SOCK_STREAM) || (net->type == SOCK_DGRAM)) {
        if ((net->type == SOCK_STREAM) && (net->addrSetLocal == FALSE)) {
            if (getsockname(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
                doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, LOCAL);
            }
        }

        if ((net->type == SOCK_STREAM) && (net->addrSetRemote == FALSE)) {
            if (getpeername(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
                doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, REMOTE);
            }
        }
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
    struct sockaddr_storage addr;
    socklen_t addrlen = sizeof(addr);

    if (getsockname(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
        if (addrIsNetDomain(&addr) || addrIsUnixDomain(&addr)) {
            int type;
            socklen_t len = sizeof(type);

            if (getsockopt(sockfd, SOL_SOCKET, SO_TYPE, &type, &len) == 0) {
                addSock(sockfd, type, addr.ss_family);
            } else {
                // Really can't add the socket at this point
                scopeLog(CFG_LOG_ERROR, "fd:%d ERROR: doAddNewSock:getsockopt", sockfd);
            }
        } else {
            // is RAW a viable default?
            addSock(sockfd, SOCK_RAW, addr.ss_family);
        }
        doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, LOCAL);
    } else {
        addSock(sockfd, SOCK_RAW, 0);
    }

    addrlen = sizeof(addr);
    if (getpeername(sockfd, (struct sockaddr *)&addr, &addrlen) != -1) {
        doSetConnection(sockfd, (struct sockaddr *)&addr, addrlen, REMOTE);
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
    struct net_info_t *net = getNetEntry(sd);

    if (net == NULL) {
        return -1;
    }

    net->startTime = getTime();

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
    char *pkt_end = (char *)pkt + pktlen;

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
        g_netinfo[sd].dnsSend = TRUE;
    } else {
        strncpy(g_netinfo[sd].dnsName, dnsName, dnsNameBytesUsed);
        g_netinfo[sd].dnsSend = FALSE;
    }

    return 0;
}

/*
 * Name server functions are available in a glibc distro
 * from libresolv. They are available in libmusl on a
 * musl based distro, no libresolv is used. So, if we
 * link to libresolv we work on a glibc distro, but
 * we fail on a musl distro. To avoid dealing with
 * that we use function pointers and populate them
 * when they are needed.
 *
 * For clarification, when the executable has a
 * dependency on libresolv the function pointers
 * are resolved at libscope constructor time.
 * Otherwise, we do an explicit dlopen to load
 * libresolv and get the addrs of the functions
 * we need. The dlopen works as libresolv is
 * located in a default lib search path.
 */
static bool
getNSFuncs(void)
{
    if (!g_fn.dlopen) return FALSE;

    void *handle = g_fn.dlopen("libresolv.so", RTLD_LAZY | RTLD_NODELETE);
    if (handle == NULL) {
        scopeLog(CFG_LOG_WARN,
                    "WARNING: could not locate libresolv, DNS events will be affected");
        return FALSE;
    }

    if (!g_fn.ns_initparse) g_fn.ns_initparse = dlsym(handle, "ns_initparse");
    if (!g_fn.ns_parserr) g_fn.ns_parserr = dlsym(handle, "ns_parserr");
    dlclose(handle);

    if (!g_fn.ns_initparse || !g_fn.ns_parserr) {
        scopeLog(CFG_LOG_WARN,
                    "WARNING: could not locate name server functions, DNS events will be affected");
        return FALSE;
    }

    return TRUE;
}

#define DNSDONE(var1, var2) {if (var1) cJSON_Delete(var1); if (var2) cJSON_Delete(var2); return FALSE;}

static bool
parseDNSAnswer(char *buf, size_t len, cJSON *json, cJSON *addrs, int first)
{
    int i, nmsg;
    ns_rr rr;
    ns_msg handle;

    if (!g_fn.ns_initparse || !g_fn.ns_parserr) {
        if (getNSFuncs() == FALSE) return FALSE;
    }

    // init ns lib
    if (g_fn.ns_initparse((const unsigned char *)buf, len, &handle) == -1) {
        scopeLog(CFG_LOG_ERROR, "ERROR:init parse");
        return FALSE;
    }

    nmsg = ns_msg_count(handle, ns_s_an);

    if (nmsg > 0) {
        for (i = 0; i < nmsg; i++) {
            char ipaddr[128];
            //char dispbuf[4096];

            if (g_fn.ns_parserr(&handle, ns_s_an, i, &rr) == -1) {
                scopeLog(CFG_LOG_ERROR, "ERROR:parse rr");
                return FALSE;
            }

            // do this once, after we get the rr
            if (first == 0) {
                if (!cJSON_AddStringToObjLN(json, "domain", ns_rr_name(rr))) {
                    continue;
                }
                first = 1;
            }

            //ns_sprintrr(&handle, &rr, NULL, NULL, dispbuf, sizeof (dispbuf));
            //scopeLog(CFG_LOG_DEBUG, "%s", dispbuf);

            // type A is IPv4, AAA is IPv6
            if (ns_rr_type(rr) == ns_t_a) {
                if (!inet_ntop(AF_INET, (struct sockaddr_in *)rr.rdata,
                               ipaddr, sizeof(ipaddr))) {
                    continue;
                }
            } else if (ns_rr_type(rr) == ns_t_aaaa) {
                if (!inet_ntop(AF_INET6, (struct sockaddr_in6 *)rr.rdata,
                               ipaddr, sizeof(ipaddr))) {
                    continue;
                }
            } else {
                DBG("DNS response received without an IP address");
                continue;
            }

            //snprintf(dispbuf, sizeof(dispbuf), "resolved addr is %s\n", ipaddr);
            //scopeLog(CFG_LOG_DEBUG, "%s", dispbuf);

            if (!cJSON_AddStringToObjLN(addrs, "addr", ipaddr)) {
                continue;
            }
        }
        return TRUE;
    }

    return FALSE;
}

bool
getDNSAnswer(int sockfd, char *buf, size_t len, src_data_t dtype)
{
    bool result;
    struct net_info_t *net = getNetEntry(sockfd);

    if (!buf || !net || (len <= 0)) return FALSE;

    cJSON *json = cJSON_CreateObject();
    if (!json) return FALSE;

    cJSON *addrs = cJSON_CreateArray();
    if (!addrs) DNSDONE(json, addrs);

    if (net->startTime == 0) {
        if (!cJSON_AddNumberToObjLN(json, "duration", 0)) {
            DNSDONE(json, addrs);
        }
    } else {
        if (!cJSON_AddNumberToObjLN(json, "duration",
                                    getDuration(net->startTime) / 1000000)) {
            DNSDONE(json, addrs);
        }
    }

    switch (dtype) {
    case BUF:
        result = parseDNSAnswer(buf, len, json, addrs, 0);
        break;

    case MSG:
    {
        int i;
        struct msghdr *msg = (struct msghdr *)buf;
        struct iovec *iov;

        for (i = 0; i < msg->msg_iovlen; i++) {
            iov = &msg->msg_iov[i];
            if (iov && iov->iov_base && (iov->iov_len > 0)) {
                // do we have at least one good pass?
                if (parseDNSAnswer((char *)iov->iov_base, len, json, addrs, i) == TRUE) {
                    result = TRUE;
                } else {
                    // should we stop if an iov doesn't parse? probably.
                    break;
                }
            }
        }
        break;
    }

    case IOV:
    {
        int i;
        struct iovec *iov = (struct iovec *)buf;

        for (i = 0; i < len; i++) {
            if (iov[i].iov_base && (iov[i].iov_len > 0)) {
                if (parseDNSAnswer((char *)iov[i].iov_base, len, json, addrs, i)  == TRUE) {
                    result = TRUE;
                } else {
                    break;
                }
            }
        }
        break;
    }

    default:
        DNSDONE(json, addrs);
    }

    if (result == TRUE) {
        cJSON_AddItemToObject(json, "addrs", addrs);
        net->dnsAnswer = json;
    } else {
        net->dnsAnswer = NULL;
        if (json) cJSON_Delete(json);
        if (addrs) cJSON_Delete(addrs);
    }

    return TRUE;
}

int
doURL(int sockfd, const void *buf, size_t len, metric_t src)
{
    if (g_cfg.urls == 0) return 0;

    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
    }


    if ((src == NETTX) && (searchExec(g_http_redirect, (char *)buf, len) != -1)) {
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
doRecv(int sockfd, ssize_t rc, const void *buf, size_t len, src_data_t src)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);

        /*
         * If a netrx operation returns a len of 0
         * it means that the remote end has disconnected.
         * This is the the traditional "end-of-file"
         */
        if (len == 0) {
            g_netinfo[sockfd].remoteClose = TRUE;
            // Seems that returning here makes sense with a len of 0
            return 0;
        }

        doUpdateState(NETRX, sockfd, rc, NULL, NULL);

        if ((g_netinfo[sockfd].dnsRecv == FALSE) &&
            remotePortIsDNS(sockfd) &&
            (g_netinfo[sockfd].dnsName[0])) {
            g_netinfo[sockfd].dnsRecv = TRUE;
            doUpdateState(DNS, sockfd, (ssize_t)1, NULL, g_netinfo[sockfd].dnsName);
        }

        if ((sockfd != -1) && buf) {
            doProtocol((uint64_t)-1, sockfd, (void *)buf, len, NETRX, src);
        }
    }
    return 0;
}

int
doSend(int sockfd, ssize_t rc, const void *buf, size_t len, src_data_t src)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doUpdateState(NETTX, sockfd, rc, NULL, NULL);

        if ((g_netinfo[sockfd].dnsSend == FALSE) &&
            remotePortIsDNS(sockfd) &&
            (g_netinfo[sockfd].dnsName[0])) {
            doUpdateState(DNS, sockfd, (ssize_t)0, NULL, NULL);
            g_netinfo[sockfd].dnsSend = TRUE;
        }

        if ((sockfd != -1) && buf && (len > 0)) {
            doProtocol((uint64_t)-1, sockfd, (void *)buf, len, NETTX, src);
        }
    }
    return 0;
}

void
doAccept(int sd, struct sockaddr *addr, socklen_t *addrlen, char *func)
{
    scopeLog(CFG_LOG_DEBUG, "fd:%d %s", sd, func);
    if (addr) {
        addSock(sd, SOCK_STREAM, addr->sa_family);
    } else {
        doAddNewSock(sd);
    }


    if (getNetEntry(sd) != NULL) {
        if (addr && addrlen) doSetConnection(sd, addr, *addrlen, REMOTE);
        doUpdateState(OPEN_PORTS, sd, 1, func, NULL);
        doUpdateState(NET_CONNECTIONS, sd, 1, func, NULL);
    }
}


//
// reportFD is called in two cases:
//   1) when a socket or file is being closed
//   2) during periodic reporting
void
reportFD(int fd, control_type_t source)
{
    if (source == EVENT_BASED) return;

    struct net_info_t *ninfo = getNetEntry(fd);
    if (ninfo) {
        ninfo->fd = fd;
        if (!g_summary.net.rx_tx) {
            doNetMetric(NETTX, ninfo, source, 0);
            doNetMetric(NETRX, ninfo, source, 0);
        }
        if (!g_summary.net.open_close) {
            doNetMetric(OPEN_PORTS, ninfo, source, 0);
            doNetMetric(NET_CONNECTIONS, ninfo, source, 0);
            doNetMetric(CONNECTION_DURATION, ninfo, source, 0);
        }
    }

    struct fs_info_t *finfo = getFSEntry(fd);
    if (finfo) {
        finfo->fd = fd;
        if (!g_summary.fs.read_write) {
            doFSMetric(FS_DURATION, finfo, source, "read/write", 0, NULL);
            doFSMetric(FS_READ, finfo, source, "read", 0, NULL);
            doFSMetric(FS_WRITE, finfo, source, "write", 0, NULL);
        }
        if (!g_summary.fs.seek) {
            doFSMetric(FS_SEEK, finfo, source, "seek", 0, NULL);
        }
    }
}

void
reportAllFds(control_type_t source)
{
    int i;
    for (i = 0; i < MAX(g_numNinfo, g_numFSinfo); i++) {
        reportFD(i, source);
    }
}

void
doRead(int fd, uint64_t initialTime, int success, const void *buf, ssize_t bytes,
       const char *func, src_data_t src, size_t cnt)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    if (success) {
        scopeLog(CFG_LOG_TRACE, "fd:%d %s", fd, func);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            if (src == IOV) {
                doRecv(fd, bytes, buf, cnt, src);
            } else {
                doRecv(fd, bytes, buf, bytes, src);
            }
        } else if (fs) {
            // Don't count data from stdin
            if ((fd > 2) || strncmp(fs->path, "std", 3)) {
                uint64_t duration = getDuration(initialTime);
                doUpdateState(FS_DURATION, fd, duration, func, NULL);
                doUpdateState(FS_READ, fd, bytes, func, NULL);
            }
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_READ_WRITE, fd, bytes, func, fs->path);
        } else if (net) {
            doUpdateState(NET_ERR_RX_TX, fd, bytes, func, "nopath");
        }
    }
}

void
doWrite(int fd, uint64_t initialTime, int success, const void *buf, ssize_t bytes,
        const char *func, src_data_t src, size_t cnt)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);

    if (success) {
        scopeLog(CFG_LOG_TRACE, "fd:%d %s", fd, func);
        if (net) {
            // This is a network descriptor
            doSetAddrs(fd);
            if (src == IOV) {
                doSend(fd, bytes, buf, cnt, src);
            } else {
                doSend(fd, bytes, buf, bytes, src);
            }
        } else if (fs) {
            // Don't count data from stdout, stderr
            if ((fd > 2) || strncmp(fs->path, "std", 3)) {
                uint64_t duration = getDuration(initialTime);
                doUpdateState(FS_DURATION, fd, duration, func, NULL);
                doUpdateState(FS_WRITE, fd, bytes, func, NULL);
            }

            if (src == IOV) {
                int i;
                struct iovec *iov = (struct iovec *)buf;

                for (i = 0; i < cnt; i++) {
                    if (iov[i].iov_base && (iov[i].iov_len > 0)) {
                        ctlSendLog(g_ctl, fd, fs->path, iov[i].iov_base, iov[i].iov_len, fs->uid, &g_proc);
                    }
                }

                return;
            }

            ctlSendLog(g_ctl, fd, fs->path, buf, bytes, fs->uid, &g_proc);
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_READ_WRITE, fd, bytes, func, fs->path);
        } else if (net) {
            doUpdateState(NET_ERR_RX_TX, fd, bytes, func, "nopath");
        }
    }
}

void
doSeek(int fd, int success, const char *func)
{
    struct fs_info_t *fs = getFSEntry(fd);
    if (success) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d %s", fd, func);
        if (fs) {
            doUpdateState(FS_SEEK, fd, 0, func, NULL);
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_READ_WRITE, fd, (size_t)0, func, fs->path);
        }
    }
}

#ifdef __linux__
void
doStatPath(const char *path, int rc, const char *func)
{
    if (rc != -1) {
        scopeLog(CFG_LOG_DEBUG, "%s", func);
        doUpdateState(FS_STAT, -1, 0, func, path);
    } else {
        doUpdateState(FS_ERR_STAT, -1, (size_t)0, func, path);
    }
}

void
doStatFd(int fd, int rc, const char* func)
{
    struct fs_info_t *fs = getFSEntry(fd);

    if (rc != -1) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d %s", fd, func);
        if (fs) {
            doUpdateState(FS_STAT, fd, 0, func, fs->path);
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_STAT, fd, (size_t)0, func, fs->path);
        }
    }
}
#endif // __linux__

int
doDupFile(int oldfd, int newfd, const char *func)
{
    if (!checkFSEntry(newfd) || !checkFSEntry(oldfd)) {
        return -1;
    }

    doOpen(newfd, g_fsinfo[oldfd].path, g_fsinfo[oldfd].type, func);
    return 0;
}

int
doDupSock(int oldfd, int newfd)
{
    if (!checkNetEntry(newfd) || !checkNetEntry(oldfd)) {
        return -1;
    }

    memmove(&g_netinfo[newfd], &g_netinfo[oldfd], sizeof(struct fs_info_t));
    g_netinfo[newfd].active = TRUE;
    g_netinfo[newfd].numTX = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].numRX = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].txBytes = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].rxBytes = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].startTime = 0ULL;
    g_netinfo[newfd].totalDuration = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].numDuration = (counters_element_t){.mtc=0, .evt=0};

    // don't dup the HTTP/2 frame stashes
    memset(&g_netinfo[newfd].http.http2Buf[0], 0, sizeof(http_buf_t));
    memset(&g_netinfo[newfd].http.http2Buf[1], 0, sizeof(http_buf_t));

    doUpdateState(CONNECTION_OPEN, newfd, 1, "dup", NULL);
    return 0;
}

void
doDup(int fd, int rc, const char *func, int copyNet)
{
    struct fs_info_t *fs = getFSEntry(fd);
    struct net_info_t *net = getNetEntry(fd);
    if (rc != -1) {
        if (net) {
            // This is a network descriptor
            scopeLog(CFG_LOG_DEBUG, "fd:%d %s", rc, func);
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
            doUpdateState(FS_ERR_OPEN_CLOSE, fd, (size_t)0, func, fs->path);
        } else if (net) {
            doUpdateState(NET_ERR_CONN, fd, (size_t)0, func, "nopath");
        }
    }
}

void
doDup2(int oldfd, int newfd, int rc, const char *func)
{
    struct fs_info_t *fs = getFSEntry(oldfd);
    struct net_info_t *net = getNetEntry(oldfd);

    if ((rc != -1) && (oldfd != newfd)) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d %s", rc, func);
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
            doUpdateState(FS_ERR_OPEN_CLOSE, oldfd, (size_t)0, func, fs->path);
        } else if (net) {
            doUpdateState(NET_ERR_CONN, oldfd, (size_t)0, func, "nopath");
        }
    }
}

void
doClose(int fd, const char *func)
{
    struct net_info_t *ninfo;
    struct fs_info_t *fsinfo;

    ninfo = getNetEntry(fd);

    int guard_enabled = g_http_guard_enabled && ninfo;
    if (guard_enabled) while (!atomicCasU64(&g_http_guard[fd], 0ULL, 1ULL));

    if (ninfo != NULL) {
        doUpdateState(OPEN_PORTS, fd, -1, func, NULL);
        doUpdateState(NET_CONNECTIONS, fd, -1, func, NULL);
        doUpdateState(CONNECTION_DURATION, fd, -1, func, NULL);
        resetHttp(&ninfo->http);
    }

    // Check both file descriptor tables
    if ((fsinfo = getFSEntry(fd)) != NULL) {

        doUpdateState(FS_CLOSE, fd, 0, func, NULL);
    }

    // report everything before the info is lost
    reportFD(fd, EVENT_BASED);

    if (ninfo) memset(ninfo, 0, sizeof(struct net_info_t));
    if (fsinfo) memset(fsinfo, 0, sizeof(struct fs_info_t));

    if (guard_enabled) while (!atomicCasU64(&g_http_guard[fd], 1ULL, 0ULL));
}

void
doOpen(int fd, const char *path, fs_type_t type, const char *func)
{
    if (checkFSEntry(fd) == TRUE) {
        if (g_fsinfo[fd].active) {
            scopeLog(CFG_LOG_DEBUG, "fd:%d doOpen: duplicate", fd);
            DBG(NULL);
            doClose(fd, func);
        }
/*
 * We need to do this realloc.
 * However, it needs to be done in such a way as to not
 * free the previous object that may be in use by a thread.
 * Possibly not use realloc. Leaving the code in place and this
 * comment as a reminder.

        if ((fd > g_numFSinfo) && (fd < MAX_FDS))  {
            int increase;
            fs_info *temp;

            if (fd < (MAX_FDS / 2)) {
                increase = MAX_FDS / 2;
            } else {
                increase = MAX_FDS;
            }

            // Need to realloc
            if ((temp = realloc(g_fsinfo, sizeof(struct fs_info_t) * increase)) == NULL) {
                scopeLog(CFG_LOG_ERROR, "fd:%d ERROR: doOpen:realloc", fd);
                DBG("re-alloc on FS table failed");
            } else {
                memset(&temp[g_numFSinfo], 0, sizeof(struct fs_info_t) * (increase - g_numFSinfo));
                g_fsinfo = temp;
                g_numFSinfo = increase;
            }
        }
*/
        memset(&g_fsinfo[fd], 0, sizeof(struct fs_info_t));
        g_fsinfo[fd].active = TRUE;
        g_fsinfo[fd].type = type;
        g_fsinfo[fd].uid = getTime();
        strncpy(g_fsinfo[fd].path, path, sizeof(g_fsinfo[fd].path));

        if (ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS) && ctlEnhanceFs(g_ctl)) {
            struct stat sbuf;
            int errsave = errno;

            if ((g_fn.__xstat) && (g_fn.__xstat(1, g_fsinfo[fd].path, &sbuf) == 0)) {
                g_fsinfo[fd].fuid = sbuf.st_uid;
                g_fsinfo[fd].fgid = sbuf.st_gid;
                g_fsinfo[fd].mode = sbuf.st_mode;
            }
            errno = errsave;
        }

        doUpdateState(FS_OPEN, fd, 0, func, path);
        scopeLog(CFG_LOG_TRACE, "fd:%d %s", fd, func);
    }
}

void
doSendFile(int out_fd, int in_fd, uint64_t initialTime, int rc, const char *func)
{
    struct fs_info_t *fsrd = getFSEntry(in_fd);
    struct net_info_t *nettx = getNetEntry(out_fd);

    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d %s", in_fd, func);
        if (nettx) {
            doSetAddrs(out_fd);
            doSend(out_fd, rc, NULL, 0, NONE);
        }

        if (fsrd) {
            uint64_t duration = getDuration(initialTime);
            doUpdateState(FS_DURATION, in_fd, duration, func, NULL);
            doUpdateState(FS_WRITE, in_fd, rc, func, NULL);
        }
    } else {
        /*
         * We don't want to increment an error twice
         * We don't know which fd the error is associated with
         * We emit one metric with the input pathname
         */
        if (fsrd) {
            doUpdateState(FS_ERR_READ_WRITE, in_fd, (size_t)0, func, fsrd->path);
        }

        if (nettx) {
            doUpdateState(NET_ERR_RX_TX, out_fd, (size_t)0, func, "nopath");
        }
    }
}

void
doCloseAndReportFailures(int fd, int success, const char *func)
{
    struct fs_info_t *fs;
    if (success) {
        doClose(fd, func);
    } else {
        if ((fs = getFSEntry(fd))) {
            doUpdateState(FS_ERR_OPEN_CLOSE, fd, (size_t)0, func, fs->path);
        }
    }
}

void
doCloseAllStreams()
{
    if (!g_fsinfo) return;
    int i;
    for (i = 0; i < g_numFSinfo; i++) {
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

bool
addrIsNetDomain(struct sockaddr_storage *sock)
{
    if (!sock) return FALSE;

    return  ((sock->ss_family == AF_INET) ||
             (sock->ss_family == AF_INET6));
}

bool
addrIsUnixDomain(struct sockaddr_storage *sock)
{
    if (!sock) return FALSE;

    return  ((sock->ss_family == AF_UNIX) ||
             (sock->ss_family == AF_LOCAL));
}

sock_summary_bucket_t
getNetRxTxBucket(net_info *net)
{
    if (!net) return SOCK_OTHER;

    sock_summary_bucket_t bucket = SOCK_OTHER;

    if (addrIsNetDomain(&net->localConn) ||
        addrIsNetDomain(&net->remoteConn)) {

        if (net->type == SOCK_STREAM) {
            bucket = INET_TCP;
        } else if (net->type == SOCK_DGRAM) {
            bucket = INET_UDP;
        }

    } else if (addrIsUnixDomain(&net->localConn) ||
               addrIsUnixDomain(&net->remoteConn)) {

        if (net->type == SOCK_STREAM) {
            bucket = UNIX_TCP;
        } else if (net->type == SOCK_DGRAM) {
            bucket = UNIX_UDP;
        }

    }

    return bucket;
}
