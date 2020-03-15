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
#include "dbg.h"
#include "dns.h"
#include "mtcformat.h"
#include "plattime.h"
#include "state.h"
#include "state_private.h"
#include "runtimecfg.h"

#define NET_ENTRIES 1024
#define FS_ENTRIES 1024

extern rtconfig g_cfg;

int g_numNinfo = NET_ENTRIES;
int g_numFSinfo = FS_ENTRIES;

// These would all be declared static, but the some functions that need
// this data have been moved into report.c.  This is managed with the
// include of state_private.h above.
summary_t g_summary = {0};
net_info *g_netinfo;
fs_info *g_fsinfo;
metric_counters g_ctrs = {0};


// interfaces
mtc_t *g_mtc = NULL;
ctl_t *g_ctl = NULL;

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

void
initState()
{
    net_info *netinfoLocal;
    fs_info *fsinfoLocal;
    if ((netinfoLocal = (net_info *)malloc(sizeof(struct net_info_t) * NET_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    if (netinfoLocal) memset(netinfoLocal, 0, sizeof(struct net_info_t) * NET_ENTRIES);

    // Per a Read Update & Change (RUC) model; now that the object is ready assign the global
    g_netinfo = netinfoLocal;

    if ((fsinfoLocal = (fs_info *)malloc(sizeof(struct fs_info_t) * FS_ENTRIES)) == NULL) {
        scopeLog("ERROR: Constructor:Malloc", -1, CFG_LOG_ERROR);
    }

    if (fsinfoLocal) memset(fsinfoLocal, 0, sizeof(struct fs_info_t) * FS_ENTRIES);

    // Per RUC...
    g_fsinfo = fsinfoLocal;
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

static void
postStatErrState(metric_t stat_err, metric_t type, const char *funcop, const char *pathname)
{
    // something passed in a param that is not a viable address; ltp does this
    if ((stat_err == EVT_ERR) && (errno == EFAULT)) return;

    size_t len = sizeof(struct stat_err_info_t);
    stat_err_info *sep = calloc(1, len);
    if (!sep) return;

    sep->evtype = stat_err;
    sep->data_type = type;

    if (pathname) {
        strncpy(sep->name, pathname, strnlen(pathname, sizeof(sep->name)));
    }

    if (funcop) {
        strncpy(sep->funcop, funcop, strnlen(funcop, sizeof(sep->funcop)));
    }

    cmdPostEvent(g_ctl, (char *)sep);
}

static void
postFSState(int fd, metric_t type, fs_info *fs, const char *funcop, const char *pathname)
{
    if (!cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_METRIC) &&
        !cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_FILE) &&
        !cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_CONSOLE) &&
        !cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_SYSLOG)) {
        switch (type) {
        case FS_READ:
        case FS_WRITE:
        case FS_DURATION:
            if (g_summary.fs.read_write) return;
            break;

        case FS_OPEN:
        case FS_CLOSE:
            if (g_summary.fs.open_close) return;
            break;

        case FS_SEEK:
            if (g_summary.fs.seek) return;
            break;

        default:
            break;
        }
    }

    size_t len = sizeof(struct fs_info_t);
    fs_info *fsp = calloc(1, len);
    if (!fsp) return;

    bcopy(fs, fsp, len);
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
}

static void
postDNSState(int fd, metric_t type, net_info *net, uint64_t duration, const char *domain)
{
    size_t len = sizeof(struct net_info_t);
    net_info *netp = calloc(1, len);
    if (!netp) return;

    bcopy(net, netp, len);
    netp->fd = fd;
    netp->evtype = EVT_DNS;
    netp->data_type = type;

    if (duration > 0) {
        netp->totalDuration = duration;
    }

    if (domain) {
        strncpy(netp->dnsName, domain, strnlen(domain, sizeof(netp->dnsName)));
    }

    cmdPostEvent(g_ctl, (char *)netp);
}

static void
postNetState(int fd, metric_t type, net_info *net)
{
    if (!cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_METRIC)) {
        switch (type) {
        case OPEN_PORTS:
        case NET_CONNECTIONS:
        case CONNECTION_DURATION:
            if (g_summary.net.open_close) return;
            break;

        case NETRX:
        case NETTX:
            if (g_summary.net.rx_tx) return;
            break;

        default:
            break;
        }
    }

    size_t len = sizeof(struct net_info_t);
    net_info *netp = calloc(1, len);
    if (!netp) return;

    bcopy(net, netp, len);
    netp->fd = fd;
    netp->evtype = EVT_NET;
    netp->data_type = type;

    cmdPostEvent(g_ctl, (char *)netp);
}

void
doUpdateState(metric_t type, int fd, ssize_t size, const char *funcop, const char *pathname)
{
    switch (type) {
    case OPEN_PORTS:
    {
        if (size < 0) {
            atomicSubU64(&g_ctrs.openPorts, labs(size));
        } else {
            atomicAddU64(&g_ctrs.openPorts, size);
        }
        postNetState(fd, type, &g_netinfo[fd]);
        break;
    }

    case NET_CONNECTIONS:
    {
        uint64_t* value = NULL;

        if (g_netinfo[fd].type == SOCK_STREAM) {
            value = &g_ctrs.netConnectionsTcp;
        } else if (g_netinfo[fd].type == SOCK_DGRAM) {
            value = &g_ctrs.netConnectionsUdp;
        } else {
            value = &g_ctrs.netConnectionsOther;
        }

        if (size < 0) {
            atomicSubU64(value, labs(size));
        } else {
            atomicAddU64(value, size);
        }

        if (!g_netinfo[fd].startTime) {
            g_netinfo[fd].startTime = getTime();
        }
        postNetState(fd, type, &g_netinfo[fd]);
        break;
    }

    case CONNECTION_DURATION:
    {
        uint64_t new_duration = 0ULL;
        if (g_netinfo[fd].startTime != 0ULL) {
            new_duration = getDuration(g_netinfo[fd].startTime);
            g_netinfo[fd].startTime = 0ULL;
        }

        if (new_duration) {
            atomicAddU64(&g_netinfo[fd].numDuration, 1);
            atomicAddU64(&g_netinfo[fd].totalDuration, new_duration);
            atomicAddU64(&g_ctrs.connDurationNum, 1);
            atomicAddU64(&g_ctrs.connDurationTotal, new_duration);
        }
        postNetState(fd, type, &g_netinfo[fd]);
        break;
    }

    case NETRX:
    {
        atomicAddU64(&g_netinfo[fd].numRX, 1);
        atomicAddU64(&g_netinfo[fd].rxBytes, size);
        atomicAddU64(&g_ctrs.netrxBytes, size);
        postNetState(fd, type, &g_netinfo[fd]);
        break;
    }

    case NETTX:
    {
        atomicAddU64(&g_netinfo[fd].numTX, 1);
        atomicAddU64(&g_netinfo[fd].txBytes, size);
        atomicAddU64(&g_ctrs.nettxBytes, size);
        postNetState(fd, type, &g_netinfo[fd]);
        break;
    }

    case DNS:
    {
        atomicAddU64(&g_ctrs.numDNS, 1);
        postDNSState(fd, type, &g_netinfo[fd], (uint64_t)size, pathname);
        break;
    }

    case DNS_DURATION:
    {
        atomicAddU64(&g_ctrs.dnsDurationNum, 1);
        atomicAddU64(&g_ctrs.dnsDurationTotal, 0);
        postDNSState(fd, type, &g_netinfo[fd], (uint64_t)size, pathname);
        break;
    }

    case FS_DURATION:
    {
        atomicAddU64(&g_fsinfo[fd].numDuration, 1);
        atomicAddU64(&g_fsinfo[fd].totalDuration, size);
        atomicAddU64(&g_ctrs.fsDurationNum, 1);
        atomicAddU64(&g_ctrs.fsDurationTotal, size);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case FS_READ:
    {
        atomicAddU64(&g_fsinfo[fd].numRead, 1);
        atomicAddU64(&g_fsinfo[fd].readBytes, size);
        atomicAddU64(&g_ctrs.readBytes, size);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case FS_WRITE:
    {
        atomicAddU64(&g_fsinfo[fd].numWrite, 1);
        atomicAddU64(&g_fsinfo[fd].writeBytes, size);
        atomicAddU64(&g_ctrs.writeBytes, size);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case FS_OPEN:
    {
        atomicAddU64(&g_fsinfo[fd].numOpen, 1);
        atomicAddU64(&g_ctrs.numOpen, 1);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case FS_CLOSE:
    {
        atomicAddU64(&g_fsinfo[fd].numClose, 1);
        atomicAddU64(&g_ctrs.numClose, 1);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case FS_SEEK:
    {
        atomicAddU64(&g_fsinfo[fd].numSeek, 1);
        atomicAddU64(&g_ctrs.numSeek, 1);
        postFSState(fd, type, &g_fsinfo[fd], funcop, pathname);
        break;
    }

    case NET_ERR_CONN:
    {
        atomicAddU64(&g_ctrs.netConnectErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case NET_ERR_RX_TX:
    {
        atomicAddU64(&g_ctrs.netTxRxErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case FS_ERR_OPEN_CLOSE:
    {
        atomicAddU64(&g_ctrs.fsOpenCloseErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case FS_ERR_READ_WRITE:
    {
        atomicAddU64(&g_ctrs.fsRdWrErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case FS_ERR_STAT:
    {
        atomicAddU64(&g_ctrs.fsStatErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case NET_ERR_DNS:
    {
        atomicAddU64(&g_ctrs.netDNSErrors, 1);
        postStatErrState(EVT_ERR, type, funcop, pathname);
        break;
    }

    case FS_STAT:
        atomicAddU64(&g_ctrs.numStat, 1);
        postStatErrState(EVT_STAT, type, funcop, pathname);
    default:
         return;
    }
}

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
                scopeLog("ERROR: addSock:realloc", fd, CFG_LOG_ERROR);
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
        scopeLog("doBlockConnection: blocked connection", fd, CFG_LOG_INFO);
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
    }
}

int
doSetAddrs(int sockfd)
{
    struct sockaddr_storage addr;
    socklen_t addrlen = sizeof(struct sockaddr_storage);
    net_info *net;

    // Only do this if output is enabled
    if ((cfgMtcVerbosity(g_cfg.staticfg) < DEFAULT_MTC_IPPORT_VERBOSITY) &&
        !cfgEvtFormatSourceEnabled(g_cfg.staticfg, CFG_SRC_METRIC)) return 0;

    /*
     * Do this for TCP, UDP or UNIX sockets
     * Not doing connection details for other socket types
     * If a TCP socket only set the addrs once
     * It's possible for UDP sockets to change addrs as needed
     */
    if ((net = getNetEntry(sockfd)) == NULL) return 0;

    // TODO: dont think LOCAL is correct?
    if ((net->type == SCOPE_UNIX) ||
        (net->remoteConn.ss_family == AF_UNIX) ||
        (net->remoteConn.ss_family == AF_LOCAL) ||
        (net->remoteConn.ss_family == AF_NETLINK) ||
        (net->localConn.ss_family == AF_UNIX) ||
        (net->localConn.ss_family == AF_LOCAL) ||
        (net->localConn.ss_family == AF_NETLINK)) {
        if (net->addrSetUnix == TRUE) return 0;
        doUnixEndpoint(sockfd, net);
        net->addrSetUnix = TRUE;
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
    if (g_cfg.urls == 0) return 0;

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
doRecv(int sockfd, ssize_t rc)
{
    if (checkNetEntry(sockfd) == TRUE) {
        if (!g_netinfo[sockfd].active) {
            doAddNewSock(sockfd);
        }

        doSetAddrs(sockfd);
        doUpdateState(NETRX, sockfd, rc, NULL, NULL);

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
        doUpdateState(NETTX, sockfd, rc, NULL, NULL);

        if (get_port(sockfd, g_netinfo[sockfd].remoteConn.ss_family, REMOTE) == DNS_PORT) {
            // tbd - consider calling doDNSMetricName instead...
            doUpdateState(DNS, sockfd, (ssize_t)0, NULL, NULL);
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
doRead(int fd, uint64_t initialTime, int success, ssize_t bytes, const char *func)
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
doWrite(int fd, uint64_t initialTime, int success, const void *buf, ssize_t bytes, const char *func)
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
                doUpdateState(FS_DURATION, fd, duration, func, NULL);
                doUpdateState(FS_WRITE, fd, bytes, func, NULL);
            }
            ctlSendLog(g_ctl, fs->path, buf, bytes, fs->uid, &g_proc);
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
        scopeLog(func, fd, CFG_LOG_DEBUG);
        if (fs) {
            doUpdateState(FS_SEEK, fd, 0, func, NULL);
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_READ_WRITE, fd, (size_t)0, func, fs->path);
        }
    }
}

#ifdef __LINUX__
void
doStatPath(const char *path, int rc, const char *func)
{
    if (rc != -1) {
        scopeLog(func, -1, CFG_LOG_DEBUG);
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
        scopeLog(func, fd, CFG_LOG_DEBUG);
        if (fs) {
            doUpdateState(FS_STAT, fd, 0, func, fs->path);
        }
    } else {
        if (fs) {
            doUpdateState(FS_ERR_STAT, fd, (size_t)0, func, fs->path);
        }
    }
}
#endif // __LINUX__

int
doDupFile(int newfd, int oldfd, const char *func)
{
    if ((newfd >= g_numFSinfo) || (oldfd >= g_numFSinfo)) {
        return -1;
    }

    doOpen(newfd, g_fsinfo[oldfd].path, g_fsinfo[oldfd].type, func);
    return 0;
}

int
doDupSock(int oldfd, int newfd)
{
    if ((newfd >= g_numFSinfo) || (oldfd >= g_numFSinfo)) {
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
doDup(int fd, int rc, const char *func, int copyNet)
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

    if ((ninfo = getNetEntry(fd)) != NULL) {

        doUpdateState(OPEN_PORTS, fd, -1, func, NULL);
        doUpdateState(NET_CONNECTIONS, fd, -1, func, NULL);
        doUpdateState(CONNECTION_DURATION, fd, -1, func, NULL);
    }

    // Check both file desriptor tables
    if ((fsinfo = getFSEntry(fd)) != NULL) {

        doUpdateState(FS_CLOSE, fd, 0, func, NULL);
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
            scopeLog("doOpen: duplicate", fd, CFG_LOG_DEBUG);
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
                scopeLog("ERROR: doOpen:realloc", fd, CFG_LOG_ERROR);
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

        doUpdateState(FS_OPEN, fd, 0, func, path);
        scopeLog(func, fd, CFG_LOG_TRACE);
    }
}

void
doSendFile(int out_fd, int in_fd, uint64_t initialTime, int rc, const char *func)
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

