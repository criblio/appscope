#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>

#include "atomic.h"
#include "com.h"
#include "dbg.h"
#include "fn.h"
#include "httpagg.h"
#include "mtcformat.h"
#include "os.h"
#include "plattime.h"
#include "report.h"
#include "search.h"
#include "state_private.h"
#include "linklist.h"
#include "dns.h"

#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif


#define DATA_FIELD(val)         STRFIELD("data",           (val), 1, TRUE)
#define UNIT_FIELD(val)         STRFIELD("unit",           (val), 1, TRUE)
#define CLASS_FIELD(val)        STRFIELD("class",          (val), 2, TRUE)
#define PROTO_FIELD(val)        STRFIELD("proto",          (val), 2, TRUE)
#define OP_FIELD(val)           STRFIELD("op",             (val), 3, TRUE)
#define PID_FIELD(val)          NUMFIELD("pid",            (val), 4, TRUE)
#define PROC_UID(val)           NUMFIELD("proc.uid",       (val), 4, TRUE)
#define PROC_GID(val)           NUMFIELD("proc.gid",       (val), 4, TRUE)
#define PROC_CGROUP(val)        STRFIELD("proc.cgroup",    (val), 4, TRUE)
#define HOST_FIELD(val)         STRFIELD("host",           (val), 4, FALSE)
#define PROC_FIELD(val)         STRFIELD("proc",           (val), 4, FALSE)
#define HTTPSTAT_FIELD(val)     NUMFIELD("http_status",    (val), 4, TRUE)
#define DOMAIN_FIELD(val)       STRFIELD("domain",         (val), 5, TRUE)

#define FILE_FIELD(val)      STRFIELD("file",              (val), 5, TRUE)
#define FILE_EV_NAME(val)    STRFIELD("file.name",         (val), 5, TRUE)
#define FILE_EV_MODE(val)    STRFIELD("file.perms",        (val), 5, TRUE)
#define FILE_OWNER(val)      NUMFIELD("file.owner",        (val), 5, TRUE)
#define FILE_GROUP(val)      NUMFIELD("file.group",        (val), 5, TRUE)
#define FILE_RD_BYTES(val)   NUMFIELD("file.read_bytes",   (val), 5, TRUE)
#define FILE_RD_OPS(val)     NUMFIELD("file.read_ops",     (val), 5, TRUE)
#define FILE_WR_BYTES(val)   NUMFIELD("file.write_bytes",  (val), 5, TRUE)
#define FILE_WR_OPS(val)     NUMFIELD("file.write_ops",    (val), 5, TRUE)
#define FILE_ERRS(val)       NUMFIELD("file.errors",       (val), 5, TRUE)

#define LOCALIP_FIELD(val)      STRFIELD("localip",        (val), 6, TRUE)
#define REMOTEIP_FIELD(val)     STRFIELD("remoteip",       (val), 6, TRUE)
#define LOCALP_FIELD(val)       NUMFIELD("localp",         (val), 6, TRUE)
#define LOCALN_FIELD(val)       NUMFIELD("localn",         (val), 6, TRUE)
#define PORT_FIELD(val)         NUMFIELD("port",           (val), 6, TRUE)
#define REMOTEP_FIELD(val)      NUMFIELD("remotep",        (val), 6, TRUE)
#define REMOTEN_FIELD(val)      NUMFIELD("remoten",        (val), 6, TRUE)
#define FD_FIELD(val)           NUMFIELD("fd",             (val), 7, TRUE)
#define ARGS_FIELD(val)         STRFIELD("args",           (val), 7, TRUE)
#define DURATION_FIELD(val)     NUMFIELD("duration",       (val), 8, TRUE)
#define NUMOPS_FIELD(val)       NUMFIELD("numops",         (val), 8, TRUE)
#define RATE_FIELD(val)         NUMFIELD("req_per_sec",    (val), 8, TRUE)
#define HREQ_FIELD(val)         STRFIELD("req",            (val), 8, TRUE)
#define HRES_FIELD(val)         STRFIELD("resp",           (val), 8, TRUE)
#define DETECT_PROTO(val)       STRFIELD("protocol",       (val), 8, TRUE)

#define EVENT_ONLY_ATTR (CFG_MAX_VERBOSITY+1)
#define HTTP_MAX_FIELDS 30
#define NET_MAX_FIELDS 16
#define H_ATTRIB(field, att, val, verbosity) \
    field.name = att; \
    field.value_type = FMT_STR; \
    field.event_usage = TRUE; \
    field.value.str = val; \
    field.cardinality = verbosity;

#define H_VALUE(field, att, val, verbosity) \
    field.name = att; \
    field.value_type = FMT_NUM; \
    field.event_usage = TRUE;   \
    field.value.num = val; \
    field.cardinality = verbosity;

#define HTTP_NEXT_FLD(n) if (n < HTTP_MAX_FIELDS-1) {n++;}else{DBG(NULL);}
#define NEXT_FLD(n, max) if (n < max-1) {n+=1;}else{DBG(NULL);}

#define HTTP_STATUS "HTTP/1."

typedef struct http_report_t {
    char *hreq;
    char *hres;
    metric_t ptype;
    int ix;
    size_t clen;
    char rport[8];
    char lport[8];
    char raddr[INET6_ADDRSTRLEN];
    char laddr[INET6_ADDRSTRLEN];
} http_report;

// TBD - Ideally, we'd remove this dependency on the configured interval
// and replace it with a measured value.  It'd be one less dependency
// and could be more accurate.
int g_interval = DEFAULT_SUMMARY_PERIOD;
static list_t *g_maplist;
static search_t *g_http_status = NULL;
static http_agg_t *g_http_agg;

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
    g_http_status = searchComp(HTTP_STATUS);
    g_http_agg = httpAggCreate();
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
        scopeLog("ERROR: sendEvent:cmdSendMetric", -1, CFG_LOG_ERROR);
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

static void
destroyProto(protocol_info *proto)
{
    if (!proto) return;

    /*
     * for future reference;
     * proto is freed as event in doEvent().
     * post->data is the http header and is freed in destroyHttpMap()
     * when the list entry is deleted.
     */
    if (proto->data) free (proto->data);
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

static bool
getConn(struct sockaddr_storage *conn, char *addr, size_t alen, char *port, size_t plen)
{
    if (!conn || !addr || !port) return FALSE;

    if (conn->ss_family == AF_INET) {
        if (inet_ntop(AF_INET, &((struct sockaddr_in *)conn)->sin_addr,
                      addr, alen) == NULL) {
                return FALSE;
        }

        snprintf(port, plen, "%d", htons(((struct sockaddr_in *)conn)->sin_port));
    } else if (conn->ss_family == AF_INET6) {
        if (inet_ntop(AF_INET6, &((struct sockaddr_in6 *)conn)->sin6_addr,
                      addr, alen) == NULL) {
                return  FALSE;
        }

        snprintf(port, plen, "%d", htons(((struct sockaddr_in6 *)conn)->sin6_port));
    } else {
        return FALSE;
    }
    return TRUE;
}

// yeah, a lot of params. but, it's generic.
static bool
getNetInternals(int type, struct sockaddr_storage *lconn, struct sockaddr_storage *rconn,
                char *laddr, char *raddr, size_t alen, char *lport, char *rport, size_t plen,
                event_field_t *fields, int *ix, int maxfld)
{
    if (!lconn || !rconn || !laddr || !raddr || !fields || !ix) return FALSE;

    if (addrIsNetDomain(lconn)) {
        switch (type) {
        case SOCK_STREAM:
            H_ATTRIB(fields[*ix], "net.transport", "IP.TCP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_DGRAM:
            H_ATTRIB(fields[*ix], "net.transport", "IP.UDP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RAW:
            H_ATTRIB(fields[*ix], "net.transport", "IP.RAW", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RDM:
            H_ATTRIB(fields[*ix], "net.transport", "IP.RDM", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_SEQPACKET:
            H_ATTRIB(fields[*ix], "net.transport", "IP.SEQPACKET", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        default:
            break;
        }
    } else if (addrIsUnixDomain(lconn)) {
        switch (type) {
        case SOCK_STREAM:
            H_ATTRIB(fields[*ix], "net.transport", "UNIX.TCP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_DGRAM:
            H_ATTRIB(fields[*ix], "net.transport", "UNIX.UDP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RAW:
            H_ATTRIB(fields[*ix], "net.transport", "UNIX.RAW", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RDM:
            H_ATTRIB(fields[*ix], "net.transport", "UNIX.RDM", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_SEQPACKET:
            H_ATTRIB(fields[*ix], "net.transport", "UNIX.SEQPACKET", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        default:
            break;
        }
    }

    // Connection details, where we know the file descriptor
    if (getConn(rconn, raddr, alen, rport, plen) == TRUE) {
        H_ATTRIB(fields[*ix], "net.peer.ip", raddr, 5);
        NEXT_FLD(*ix, maxfld);
        H_ATTRIB(fields[*ix], "net.peer.port", rport, 5);
        NEXT_FLD(*ix, maxfld);
    }

    if (getConn(lconn, laddr, alen, lport, plen) == TRUE) {
        H_ATTRIB(fields[*ix], "net.host.ip", laddr, 1);
        NEXT_FLD(*ix, maxfld);
        H_ATTRIB(fields[*ix], "net.host.port", lport, 1);
        NEXT_FLD(*ix, maxfld);
    }

    return TRUE;
}

static size_t
getHttpStatus(char *header, size_t len, char **stext)
{
    size_t ix;
    size_t rc;
    char *val;

    // ex: HTTP/1.1 200 OK\r\n
    if ((ix = searchExec(g_http_status, header, len)) == -1) return -1;

    if ((ix < 0) || (ix > len) || ((ix + strlen(HTTP_STATUS) + 1) > len)) return -1;

    val = &header[ix + strlen(HTTP_STATUS) + 1];
    // note that the spec defines the status code to be exactly 3 chars/digits
    *stext = &header[ix + strlen(HTTP_STATUS) + 6];

    errno = 0;
    rc = strtoull(val, NULL, 0);
    if ((errno != 0) || (rc == 0)) {
        return -1;
    }
    return rc;
}

static void
httpFieldEnd(event_field_t *fields, http_report *hreport)
{
    fields[hreport->ix].name = NULL;
    fields[hreport->ix].value_type = FMT_END;
    fields[hreport->ix].value.str = NULL;
    fields[hreport->ix].cardinality = 0;
}

static bool
httpFields(event_field_t *fields, http_report *hreport, char *hdr, size_t hdr_len, protocol_info *proto)
{
    if (!fields || !hreport || !proto || !hdr) return FALSE;

    // Start with fields from the header
    char *savea = NULL, *header;
    hreport->clen = -1;

    if ((hreport->ptype == EVT_HREQ) && (hreport->hreq)) {
        strncpy(hreport->hreq, hdr, hdr_len);
        header = hreport->hreq;
    } else if ((hreport->ptype == EVT_HRES) && (hreport->hres)) {
        strncpy(hreport->hres, hdr, hdr_len);
        header = hreport->hres;
    } else {
        scopeLog("WARN: httpFields: proto ptype is not req or resp", proto->fd, CFG_LOG_WARN);
        return FALSE;
    }

    char *reqh = strtok_r(header, "\r\n", &savea);
    if (!reqh) {
        scopeLog("WARN: httpFields: parse an http request header", proto->fd, CFG_LOG_WARN);
        return FALSE;
    }

    while ((reqh = strtok_r(NULL, "\r\n", &savea)) != NULL) {
        // From RFC 2616 Section 4.2 "Field names are case-insensitive."
        if (strcasestr(reqh, "Host:")) {
            H_ATTRIB(fields[hreport->ix], "http.host", strchr(reqh, ':') + 2, 1);
            HTTP_NEXT_FLD(hreport->ix);
        } else if (strcasestr(reqh, "User-Agent:")) {
            H_ATTRIB(fields[hreport->ix], "http.user_agent", strchr(reqh, ':') + 2, 5);
            HTTP_NEXT_FLD(hreport->ix);
        } else if(strcasestr(reqh, "X-Forwarded-For:")) {
            H_ATTRIB(fields[hreport->ix], "http.client_ip", strchr(reqh, ':') + 2, 5);
            HTTP_NEXT_FLD(hreport->ix);
        } else if(strcasestr(reqh, "Content-Length:")) {
            errno = 0;
            if (((hreport->clen = strtoull(strchr(reqh, ':') + 2, NULL, 0)) == 0) || (errno != 0)) {
                hreport->clen = -1;
            }
        }
    }
    return TRUE;
}

static bool
httpFieldsInternal(event_field_t *fields, http_report *hreport, protocol_info *proto)
{
    /*
    Compression and getting to an attribute with compressed and uncompressed lenghts.
    https://www.w3.org/Protocols/rfc2616/rfc2616-sec3.html
    https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.11
    https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.13
    https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.41
    Content-Encoding: gzip, compress, deflate (identity is N/A)
    Transfer-Encoding: chunked (N/A?), identity (N/A?), gzip, compress, and deflate
    */

    H_VALUE(fields[hreport->ix], "pid", g_proc.pid, 4);
    fields[hreport->ix].event_usage = FALSE;            // for http metrics only
    HTTP_NEXT_FLD(hreport->ix);
    H_ATTRIB(fields[hreport->ix], "host", g_proc.hostname, 4);
    fields[hreport->ix].event_usage = FALSE;            // for http metrics only
    HTTP_NEXT_FLD(hreport->ix);
    H_ATTRIB(fields[hreport->ix], "proc", g_proc.procname, 4);
    fields[hreport->ix].event_usage = FALSE;            // for http metrics only
    HTTP_NEXT_FLD(hreport->ix);

    // Next, add net fields from internal state
    H_ATTRIB(fields[hreport->ix], "net.host.name", g_proc.hostname, 1);
    HTTP_NEXT_FLD(hreport->ix);

    if (proto->sock_type != -1) {
#if 0
        getNetInternals(proto->sock_type, &proto->localConn, &proto->remoteConn,
                        hreport->laddr, hreport->raddr, sizeof(hreport->raddr),
                        hreport->lport, hreport->rport, sizeof(hreport->rport),
                        fields, &hreport->ix, HTTP_MAX_FIELDS);
#else
        if (addrIsNetDomain(&proto->localConn)) {
            if (proto->sock_type == SOCK_STREAM) {
                H_ATTRIB(fields[hreport->ix], "net.transport", "IP.TCP", 1);
                HTTP_NEXT_FLD(hreport->ix);
            } else if (proto->sock_type == SOCK_DGRAM) {
                H_ATTRIB(fields[hreport->ix], "net.transport", "IP.UDP", 1);
                HTTP_NEXT_FLD(hreport->ix);
            } else {
                H_ATTRIB(fields[hreport->ix], "net.transport", "IP", 1);
                HTTP_NEXT_FLD(hreport->ix);
            }
        } else if (addrIsUnixDomain(&proto->localConn)) { // TODO: more than unix
            if (proto->sock_type == SOCK_STREAM) {
                H_ATTRIB(fields[hreport->ix], "net.transport", "Unix.TCP", 1);
                HTTP_NEXT_FLD(hreport->ix);
            } else if (proto->sock_type == SOCK_DGRAM) {
                H_ATTRIB(fields[hreport->ix], "net.transport", "Unix.UDP", 1);
                HTTP_NEXT_FLD(hreport->ix);
            } else {
                H_ATTRIB(fields[hreport->ix], "net.transport", "Unix", 1);
                HTTP_NEXT_FLD(hreport->ix);
            }
        }

        // Connection details, where we know the file descriptor
        if (getConn(&proto->remoteConn, hreport->raddr, sizeof(hreport->raddr),
                    hreport->rport, sizeof(hreport->rport)) == TRUE) {
            H_ATTRIB(fields[hreport->ix], "net.peer.ip", hreport->raddr, 5);
            HTTP_NEXT_FLD(hreport->ix);
            H_ATTRIB(fields[hreport->ix], "net.peer.port", hreport->rport, 7);
            HTTP_NEXT_FLD(hreport->ix);
        }

        if (getConn(&proto->localConn, hreport->laddr, sizeof(hreport->laddr),
                    hreport->lport, sizeof(hreport->lport)) == TRUE) {
            H_ATTRIB(fields[hreport->ix], "net.host.ip", hreport->laddr, 1);
            HTTP_NEXT_FLD(hreport->ix);
            H_ATTRIB(fields[hreport->ix], "net.host.port", hreport->lport, 1);
            HTTP_NEXT_FLD(hreport->ix);
        }
#endif
    }

    return TRUE;
}

static void
doHttpHeader(protocol_info *proto)
{
    if (!proto || !proto->data) {
        destroyProto(proto);
        return;
    }

    char *ssl;
    event_field_t fields[HTTP_MAX_FIELDS];
    http_report hreport;
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
        map->req = NULL;
        map->req_len = 0;
    }

    map->frequency++;
    ssl = (post->ssl) ? "https" : "http";
    hreport.ix = 0;
    hreport.hreq = NULL;
    hreport.hres = NULL;

 /*
     * RFC 2616 Section 5 Request
     * The Request-Line begins with a method token, followed by the Request-URI
     * and the protocol version, and ending with CRLF. The elements are separated
     * by SP characters. No CR or LF is allowed except in the final CRLF sequence.
     *
     *  Request-Line   = Method SP Request-URI SP HTTP-Version CRLF
     */
    if (proto->ptype == EVT_HREQ) {
        map->start_time = post->start_duration;
        map->req = (char *)post->hdr;
        map->req_len = proto->len;
    }

    char header[map->req_len];
    // we're either building a new req or we have a previous req
    if (map->req) {
        if ((hreport.hreq = calloc(1, map->req_len)) == NULL) {
            scopeLog("ERROR: doHttpHeader: hreq memory allocation failure", proto->fd, CFG_LOG_ERROR);
            return;
        }

        char *savea = NULL;
        strncpy(header, map->req, map->req_len);

        char *headertok = strtok_r(header, "\r\n", &savea);
        if (!headertok) {
            scopeLog("WARN: doHttpHeader: parse an http request header", proto->fd, CFG_LOG_WARN);
            return;
        }

        // The request specific values from Request-Line
        char *method_str = strtok_r(headertok, " ", &savea);
        if (method_str) {
            H_ATTRIB(fields[hreport.ix], "http.method", method_str, 1);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLog("WARN: doHttpHeader: no method in an http request header", proto->fd, CFG_LOG_WARN);
        }

        char *target_str = strtok_r(NULL, " ", &savea);
        if (target_str) {
            H_ATTRIB(fields[hreport.ix], "http.target", target_str, 4);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLog("WARN: doHttpHeader: no target in an http request header", proto->fd, CFG_LOG_WARN);
        }

        char *flavor_str = strtok_r(NULL, " ", &savea);
        if (flavor_str &&
            ((flavor_str = strtok_r(flavor_str, "/", &savea))) &&
            ((flavor_str = strtok_r(NULL, "\r", &savea)))) {
            if (proto->ptype == EVT_HREQ) {
                H_ATTRIB(fields[hreport.ix], "http.flavor", flavor_str, 1);
                HTTP_NEXT_FLD(hreport.ix);
            }
        } else {
            scopeLog("WARN: doHttpHeader: no http version in an http request header", proto->fd, CFG_LOG_WARN);
        }

        H_ATTRIB(fields[hreport.ix], "http.scheme", ssl, 1);
        HTTP_NEXT_FLD(hreport.ix);

        if (proto->ptype == EVT_HREQ) {
            hreport.ptype = EVT_HREQ;
            // Fields common to request & response
            httpFields(fields, &hreport, map->req, map->req_len, proto);
            httpFieldsInternal(fields, &hreport, proto);

            if (hreport.clen != -1) {
                H_VALUE(fields[hreport.ix], "http.request_content_length", hreport.clen, EVENT_ONLY_ATTR);
                HTTP_NEXT_FLD(hreport.ix);
            }
            map->clen = hreport.clen;

            httpFieldEnd(fields, &hreport);

            event_t sendEvent = INT_EVENT("http-req", proto->len, SET, fields);
            cmdSendHttp(g_ctl, &sendEvent, map->id, &g_proc);
        }
    }

    /*
    * RFC 2616 Section 6 Response
    * After receiving and interpreting a request message, a server responds with an HTTP response message.
    * Response = Status-Line               ; Section 6.1
    *            *(( general-header        ; Section 4.5
    *            | response-header         ; Section 6.2
    *            | entity-header ) CRLF)   ; Section 7.1
    *            CRLF
    *            [ message-body ]          ; Section 7.2
    *
    *
    * Status-Line = HTTP-Version SP Status-Code SP Reason-Phrase CRLF
    */
    if (proto->ptype == EVT_HRES) {
        if ((hreport.hres = calloc(1, proto->len)) == NULL) {
            scopeLog("ERROR: doHttpHeader: hres memory allocation failure", proto->fd, CFG_LOG_ERROR);
            return;
        }

        int rps = map->frequency;
        int sec = (map->first_time > 0) ? (int)time(NULL) - map->first_time : 1;
        if (sec > 0) {
            rps = map->frequency / sec;
        }

        map->resp = (char *)post->hdr;

        if (!map->req) {
            map->duration = 0;
        } else {
            map->duration = getDurationNow(post->start_duration, map->start_time);
            map->duration = map->duration / 1000000;
        }

        char *stext;
        size_t status = getHttpStatus((char *)map->resp, proto->len, &stext);

        // The response specific values from Status-Line
        char *savea;
        char reqheader[proto->len];
        strncpy(reqheader, map->resp, proto->len);

        char *headertok = strtok_r(reqheader, "\r\n", &savea);
        char *flavor_str = strtok_r(headertok, " ", &savea);
        if (flavor_str &&
            ((flavor_str = strtok_r(flavor_str, "/", &savea))) &&
            ((flavor_str = strtok_r(NULL, "", &savea)))) {
            H_ATTRIB(fields[hreport.ix], "http.flavor", flavor_str, 1);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLog("WARN: doHttpHeader: no version string in an http request header", proto->fd, CFG_LOG_WARN);
        }

        H_VALUE(fields[hreport.ix], "http.status_code", status, 1);
        HTTP_NEXT_FLD(hreport.ix);

        // point past the status code
        char st[strlen(stext)];
        strncpy(st, stext, strlen(stext));
        char *status_str = strtok_r(st, "\r", &savea);
        H_ATTRIB(fields[hreport.ix], "http.status_text", status_str, 1);
        HTTP_NEXT_FLD(hreport.ix);

        H_VALUE(fields[hreport.ix], "http.server.duration", map->duration, EVENT_ONLY_ATTR);
        HTTP_NEXT_FLD(hreport.ix);

        // Fields common to request & response
        if (map->req) {
            hreport.ptype = EVT_HREQ;
            httpFields(fields, &hreport, map->req, map->req_len, proto);
            if (hreport.clen != -1) {
                H_VALUE(fields[hreport.ix], "http.request_content_length", hreport.clen, EVENT_ONLY_ATTR);
                HTTP_NEXT_FLD(hreport.ix);
            }
            map->clen = hreport.clen;
        }

        hreport.ptype = EVT_HRES;
        httpFields(fields, &hreport, map->resp, proto->len, proto);
        httpFieldsInternal(fields, &hreport, proto);
        if (hreport.clen != -1) {
            H_VALUE(fields[hreport.ix], "http.response_content_length", hreport.clen, EVENT_ONLY_ATTR);
            HTTP_NEXT_FLD(hreport.ix);
        }

        httpFieldEnd(fields, &hreport);

        event_t hevent = INT_EVENT("http-resp", proto->len, SET, fields);
        cmdSendHttp(g_ctl, &hevent, map->id, &g_proc);

        // Are we doing a metric event?
        event_field_t mfields[] = {
            DURATION_FIELD(map->duration),
            RATE_FIELD(rps),
            HTTPSTAT_FIELD(status),
            PROC_FIELD(g_proc.procname),
            FD_FIELD(proto->fd),
            PID_FIELD(g_proc.pid),
            UNIT_FIELD("byte"),
            FIELDEND
        };

        event_t mevent = INT_EVENT("http-metrics", proto->len, SET, mfields);
        cmdSendHttp(g_ctl, &mevent, map->id, &g_proc);

        // emit statsd metrics, if enabled.
        if (mtcEnabled(g_mtc)) {

            char *mtx_name = (proto->isServer) ? "http.server.duration" : "http.client.duration";
            event_t http_dur = INT_EVENT(mtx_name, map->duration, DELTA, fields);
            // TBD AGG Only cmdSendMetric(g_mtc, &http_dur);
            httpAggAddMetric(g_http_agg, &http_dur, map->clen, hreport.clen);

            /* TBD AGG Only
            if (map->clen != -1) {
                event_t http_req_len = INT_EVENT("http.request.content_length", map->clen, DELTA, fields);
                cmdSendMetric(g_mtc, &http_req_len);
            }

            if (hreport.clen != -1) {
                event_t http_rsp_len = INT_EVENT("http.response.content_length", hreport.clen, DELTA, fields);
                cmdSendMetric(g_mtc, &http_rsp_len);
            }
            */

        }

        // Done; we remove the list entry; complete when reported
        if (lstDelete(g_maplist, post->id) == FALSE) DBG(NULL);
    }

    if (hreport.hreq) free(hreport.hreq);
    if (hreport.hres) free(hreport.hres);
    destroyProto(proto);
}

static void
doDetection(protocol_info *proto)
{
    char *protname;

    if (!proto) return;

    protname = (char *)proto->data;
    if (!protname) {
        destroyProto(proto);
        return;
    }

    event_field_t fields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        FD_FIELD(proto->fd),
        HOST_FIELD(g_proc.hostname),
        DETECT_PROTO(protname),
        FIELDEND
    };

    event_t evt = INT_EVENT("remote_protocol", proto->fd, SET, fields);
    cmdSendEvent(g_ctl, &evt, proto->uid, &g_proc);
    destroyProto(proto);
}

void
doProtocolMetric(protocol_info *proto)
{
    if (!proto) return;

    if ((proto->ptype == EVT_HREQ) || (proto->ptype == EVT_HRES)) {
        doHttpHeader(proto);
    } else if (proto->ptype == EVT_DETECT) {
        doDetection(proto);
    }
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
doDNSMetricName(metric_t type, const char *domain, counters_element_t *duration, void *ctr)
{
    if (!domain || !domain[0]) return;

    metric_counters* ctrs = (ctr) ? (metric_counters*) ctr : &g_ctrs;

    switch (type) {
    case DNS:
    {
        // Don't report zeros.
        if (ctrs->numDNS.evt != 0) {
            if (duration && (duration->evt > 0)) {
                event_field_t resp[] = {
                    PROC_FIELD(g_proc.procname),
                    PID_FIELD(g_proc.pid),
                    HOST_FIELD(g_proc.hostname),
                    DOMAIN_FIELD(domain),
                    UNIT_FIELD("response"),
                    FIELDEND
                };
                event_t dnsMetric = INT_EVENT("net.dns.resp", ctrs->numDNS.evt, DELTA, resp);
                cmdSendEvent(g_ctl, &dnsMetric, getTime(), &g_proc);
            } else {
                event_field_t req[] = {
                    PROC_FIELD(g_proc.procname),
                    PID_FIELD(g_proc.pid),
                    HOST_FIELD(g_proc.hostname),
                    DOMAIN_FIELD(domain),
                    UNIT_FIELD("request"),
                    FIELDEND
                };
                event_t dnsMetric = INT_EVENT("net.dns.req", ctrs->numDNS.evt, DELTA, req);
                cmdSendEvent(g_ctl, &dnsMetric, getTime(), &g_proc);
            }
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
            // default to at least 1ms as opposed to reporting nothing
            if (dur == 0) dur = 1;
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
            // default to at least 1ms as opposed to reporting nothing
            if (dur == 0) dur = 1;
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

static uint64_t
getFSDuration(fs_info *fs)
{
    uint64_t dur = 0ULL;
    int cachedDurationNum = fs->numDuration.evt; // avoid div by zero
    if (cachedDurationNum >= 1) {
        // factor of 1000 converts ns to us.
        dur = fs->totalDuration.evt / ( 1000 * cachedDurationNum);
    }
    return dur;
}

/*
{
  "sourcetype": "net",
  "source": "net.conn.open",
  "cmd": "foo",
  "pid": 10831,
  "host": "hostname",
  "data" {
    "net.transport": "IP.TCP",
    "net.peer.ip": "5.9.243.187",
    "net.peer.port": 443,
    "net.peer.name": wttr.in,
    "net.host.ip": "172.17.0.2",
    "net.host.port": 49202,
    "net.host.name": "scope-vm",
    "net.protocol": "http",
  },
  "_time": timestamp
}
 */
static void
doNetOpenEvent(net_info *net)
{
    int nix = 0;
    const char *metric = "net.conn.open";
    char rport[8];
    char lport[8];
    char raddr[INET6_ADDRSTRLEN];
    char laddr[INET6_ADDRSTRLEN];

    event_field_t nevent[NET_MAX_FIELDS];

    getNetInternals(net->type, &net->localConn, &net->remoteConn,
                    laddr, raddr, sizeof(raddr),
                    lport, rport, sizeof(rport),
                    nevent, &nix, NET_MAX_FIELDS);

    if (net->dnsName[0] != 0) {
        H_ATTRIB(nevent[nix], "net.host.name", net->dnsName, 1);
        NEXT_FLD(nix, NET_MAX_FIELDS);
    }

    nevent[nix].name = NULL;
    nevent[nix].value_type = FMT_END;
    nevent[nix].value.str = NULL;
    nevent[nix].cardinality = 0;

    event_t evt = INT_EVENT(metric, g_ctrs.openPorts.evt, CURRENT, nevent);
    evt.src = CFG_SRC_NET;
    cmdSendEvent(g_ctl, &evt, net->uid, &g_proc);
}

/* Example FS Events
   {
   "sourcetype": "fs",
   "source": "fs.open",
   "cmd": "foo",
   "pid": 10831,
   "host": "hostname",
   "_time": timestamp,
   "data": {
   "file.name": "/usr/lib/ssl/openssl.cnf",
   "file.perms": 0755,
   "file.owner": 0,
   "file.group": 0,
   "proc.uid": 1000,
   "proc.gid": 1000,
   "proc.cgroup": "foo"
   }
   }

   {
   "sourcetype": "fs",
   "source": "fs.close",
   "cmd": "foo",
   "pid": 10831,
   "host": "hostname",
   "_time": timestamp,
   "data": {
   "file.name": "/usr/lib/ssl/openssl.cnf",
   "file.perms": 0755,
   "file.owner": 0,
   "file.group": 0,
   "file.read_bytes": 18343,
   "file.read_ops": 5,
   "file.write_bytes": 1823,
   "file.write_ops": 9,
   "file.errors": 1,
   "proc.uid": 1000,
   "proc.gid": 1000,
   "proc.cgroup": "foo",
   "duration": 1833
   }
   }
*/
static void
doFSOpenEvent(fs_info *fs, const char *op)
{
    const char *metric = "fs.open";
    counters_element_t *numops = &fs->numOpen;

    if (ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS) &&
        (fs->fd > 2) && strncmp(fs->path, "std", 3)) {
        char *mode = osGetFileMode(fs->mode);

        event_field_t fevent[] = {
            FILE_EV_NAME(fs->path),
            PROC_UID(g_proc.uid),
            PROC_GID(g_proc.gid),
            PROC_CGROUP(g_proc.cgroup),
            FILE_EV_MODE((mode == NULL) ? "---" : mode),
            FILE_OWNER(fs->fuid),
            FILE_GROUP(fs->fgid),
            OP_FIELD(op),
            FIELDEND
        };

        event_t evt = INT_EVENT(metric, numops->evt, DELTA, fevent);
        evt.src = CFG_SRC_FS;
        cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);

        if (mode) free(mode);
    }
}

static void
doFSCloseEvent(fs_info *fs, const char *op)
{
    const char *metric = "fs.close";

    if (ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS) &&
        (fs->fd > 2) && strncmp(fs->path, "std", 3)) {
        char *mode = osGetFileMode(fs->mode);

        event_field_t fevent[] = {
            FILE_EV_NAME(fs->path),
            PROC_UID(g_proc.uid),
            PROC_GID(g_proc.gid),
            PROC_CGROUP(g_proc.cgroup),
            FILE_EV_MODE((mode == NULL) ? "---" : mode),
            FILE_OWNER(fs->fuid),
            FILE_GROUP(fs->fgid),
            FILE_RD_BYTES(fs->readBytes.evt),
            FILE_RD_OPS(fs->numRead.evt),
            FILE_WR_BYTES(fs->writeBytes.evt),
            FILE_WR_OPS(fs->numWrite.evt),
            //FILE_ERRS(g_ctrs.fsRdWrErrors.evt), we don't track errs per fd
            DURATION_FIELD(getFSDuration(fs)),
            OP_FIELD(op),
            FIELDEND
        };

        event_t evt = INT_EVENT(metric, fs->numClose.evt, DELTA, fevent);
        evt.src = CFG_SRC_FS;
        cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);

        if (mode) free(mode);
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
            //atomicSwapU64(&fs->numDuration.evt, 0);
            //atomicSwapU64(&fs->totalDuration.evt, 0);
            ////atomicSwapU64(&g_ctrs.fsDurationNum.evt, 0);
            ////atomicSwapU64(&g_ctrs.fsDurationTotal.evt, 0);
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
            //atomicSwapU64(&numops->evt, 0);
            //atomicSwapU64(&sizebytes->evt, 0);
            ////atomicSwapU64(global_counter->evt, 0);
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
        bool reported = FALSE;
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
        if (ctlEvtSourceEnabled(g_ctl, CFG_SRC_METRIC) && (numops->evt != 0ULL)) {
            event_t evt = INT_EVENT(metric, numops->evt, DELTA, fields);
            cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
            reported = TRUE;
        }

        if ((type == FS_OPEN) && (numops->evt != 0ULL)) {
            doFSOpenEvent(fs, op);
            reported = TRUE;
        }

        if ((type == FS_CLOSE) && (numops->evt != 0ULL)) {
            doFSCloseEvent(fs, op);
            reported = TRUE;
            //atomicSwapU64(&fs->numWrite.evt, 0);
            //atomicSwapU64(&fs->writeBytes.evt, 0);
            //atomicSwapU64(&fs->numRead.evt, 0);
            //atomicSwapU64(&fs->readBytes.evt, 0);
            //atomicSwapU64(&fs->numDuration.evt, 0);
            //atomicSwapU64(&fs->totalDuration.evt, 0);
        }

        if (reported == TRUE) atomicSwapU64(&numops->evt, 0);

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

        if (g_summary.net.rx_tx) {

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

    case CONNECTION_OPEN:
    {
        doNetOpenEvent(net);
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
                NUMOPS_FIELD(net->numTX.evt),
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
                NUMOPS_FIELD(net->numTX.evt),
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
    httpAggSendReport(g_http_agg, g_mtc);
    httpAggReset(g_http_agg);
    ctlFlush(g_ctl);
}

void
doPayload()
{
    bool lsbin = TRUE;
    bool filebin = TRUE;
    uint64_t data;

    while ((data = msgPayloadGet(g_ctl)) != -1) {
        if (data) {
            payload_info *pinfo = (payload_info *)data;
            net_info *net = &pinfo->net;
            size_t hlen = 1024;
            char pay[hlen];
            char *srcstr = NULL, rx[]="rx", tx[]="tx", none[]="none";

            switch (pinfo->src) {
            case NETTX:
            case TLSTX:
                srcstr = tx;
                break;

            case NETRX:
            case TLSRX:
                srcstr = rx;
                break;

            default:
                srcstr = none;
                break;
            }

            char lport[8], rport[8];
            char lip[INET6_ADDRSTRLEN];
            char rip[INET6_ADDRSTRLEN];

            if (net) {
                if (getConn(&net->localConn, lip, sizeof(lip), lport, sizeof(lport)) == FALSE) {
                    strncpy(lip, "af_int_err", sizeof(lip));
                    strncpy(lport, "0", sizeof(lport));
                }

                if (getConn(&net->remoteConn, rip, sizeof(rip), rport, sizeof(rport)) == FALSE) {
                    strncpy(rip, "af_int_err", sizeof(rip));
                    strncpy(rport, "0", sizeof(rport));
                }
            }

            uint64_t netid = (net != NULL) ? net->uid : 0;
            int rc = snprintf(pay, hlen,
                              "{\"id\":\"%s\",\"pid\":%d,\"ppid\":%d,\"fd\":%d,\"src\":\"%s\",\"_channel\":%ld,\"len\":%ld,\"localip\":\"%s\",\"localp\":%s,\"remoteip\":\"%s\",\"remotep\":%s}",
                              g_proc.id, g_proc.pid, g_proc.ppid, pinfo->sockfd, srcstr, netid, pinfo->len, lip, lport, rip, rport);
            if (rc < 0) {
                // unlikley
                if (pinfo->data) free(pinfo->data);
                if (pinfo) free(pinfo);
                DBG(NULL);
                return;
            }

            if (rc < hlen) {
                hlen = rc + 1;
            } else {
                hlen--;
                scopeLog("WARN: payload header was truncated", pinfo->sockfd, CFG_LOG_WARN);
            }

            char *bdata = NULL;

            if (lsbin == TRUE) {
                bdata = calloc(1, hlen + pinfo->len);
                if (bdata) {
                    memmove(bdata, pay, hlen);
                    strncat(bdata, "\n", hlen);
                    memmove(&bdata[hlen], pinfo->data, pinfo->len);
                    cmdSendPayload(g_ctl, bdata, hlen + pinfo->len);
                }
            }

            if (filebin == TRUE) {
                int fd;
                char path[PATH_MAX];

                ///tmp/<splunk-pid>/<src_host:src_port:dst_port>.in
                switch (pinfo->src) {
                case NETTX:
                case TLSTX:
                    snprintf(path, PATH_MAX, "/tmp/%d_%s:%s:%s.out", g_proc.pid, rip, lport, rport);
                    break;

                case NETRX:
                case TLSRX:
                    snprintf(path, PATH_MAX, "/tmp/%d_%s:%s:%s.in", g_proc.pid, rip, rport, lport);
                    break;

                default:
                    snprintf(path, PATH_MAX, "/tmp/%d.na", g_proc.pid);
                    break;
                }

                if ((fd = g_fn.open(path, O_WRONLY | O_CREAT | O_APPEND, 0666)) != -1) {
                    g_fn.write(fd, pinfo->data, pinfo->len);
                    g_fn.close(fd);
                }
            }

            if (bdata) free(bdata);
            if (pinfo->data) free(pinfo->data);
            if (pinfo) free(pinfo);
        }
    }
}
