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

#define NET_ENTRIES 1024
#define FS_ENTRIES 1024

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
mtc_t* g_mtc = NULL;
ctl_t* g_ctl = NULL;

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
    for (i = 0; i < MAX(g_numNinfo, g_numFSinfo); i++) {
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
            ctlSendLog(g_ctl, fs->path, buf, bytes, fs->uid, &g_proc);
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

    bcopy(&g_netinfo[newfd], &g_netinfo[oldfd], sizeof(struct fs_info_t));
    g_netinfo[newfd].active = TRUE;
    g_netinfo[newfd].numTX = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].numRX = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].txBytes = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].rxBytes = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].startTime = 0ULL;
    g_netinfo[newfd].totalDuration = (counters_element_t){.mtc=0, .evt=0};
    g_netinfo[newfd].numDuration = (counters_element_t){.mtc=0, .evt=0};

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

