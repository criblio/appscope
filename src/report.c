#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <sys/time.h>
#include <lshpack.h>

#include "atomic.h"
#include "com.h"
#include "dbg.h"
#include "fn.h"
#include "httpagg.h"
#include "metriccapture.h"
#include "mtcformat.h"
#include "plattime.h"
#include "report.h"
#include "search.h"
#include "state.h"
#include "state_private.h"
#include "linklist.h"
#include "dns.h"
#include "utils.h"
#include "runtimecfg.h"
#include "cfg.h"
#include "scopestdlib.h"

#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif


#define DATA_FIELD(val)         STRFIELD("data",           (val), 1, TRUE)
#define UNIT_FIELD(val)         STRFIELD("unit",           (val), 1, TRUE)
#define SUMMARY_FIELD(val)      STRFIELD("summary",        (val), 1, TRUE)
#define CLASS_FIELD(val)        STRFIELD("class",          (val), 2, TRUE)
#define PROTO_FIELD(val)        STRFIELD("proto",          (val), 2, TRUE)
#define OP_FIELD(val)           STRFIELD("op",             (val), 3, TRUE)
#define PID_FIELD(val)          NUMFIELD("pid",            (val), 4, TRUE)
#define PROC_UID(val)           NUMFIELD("proc_uid",       (val), 4, TRUE)
#define PROC_GID(val)           NUMFIELD("proc_gid",       (val), 4, TRUE)
#define PROC_CGROUP(val)        STRFIELD("proc_cgroup",    (val), 4, TRUE)
#define HOST_FIELD(val)         STRFIELD("host",           (val), 4, TRUE)
#define PROC_FIELD(val)         STRFIELD("proc",           (val), 4, TRUE)
#define HTTPSTAT_FIELD(val)     NUMFIELD("http_status",    (val), 4, TRUE)
#define DOMAIN_FIELD(val)       STRFIELD("domain",         (val), 5, TRUE)

#define FILE_FIELD(val)      STRFIELD("file",              (val), 5, TRUE)
#define FILE_EV_NAME(val)    STRFIELD("file",              (val), 5, TRUE)
#define FILE_EV_MODE(val)    NUMFIELD("file_perms",        (val), 5, TRUE)
#define FILE_OWNER(val)      NUMFIELD("file_owner",        (val), 5, TRUE)
#define FILE_GROUP(val)      NUMFIELD("file_group",        (val), 5, TRUE)
#define FILE_RD_BYTES(val)   NUMFIELD("file_read_bytes",   (val), 5, TRUE)
#define FILE_RD_OPS(val)     NUMFIELD("file_read_ops",     (val), 5, TRUE)
#define FILE_WR_BYTES(val)   NUMFIELD("file_write_bytes",  (val), 5, TRUE)
#define FILE_WR_OPS(val)     NUMFIELD("file_write_ops",    (val), 5, TRUE)
#define FILE_ERRS(val)       NUMFIELD("file_errors",       (val), 5, TRUE)

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
#define NUM_DYNS 4

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

// saved state for an HTTP/2 channel
typedef struct http2Channel {
    // HPAC decoder
    struct lshpack_dec decoder;

    // list of http2Stream_t indexed by stream
    list_t *streams;
} http2Channel_t;

// saved state for an HTTP/2 stream within a channel
typedef struct http2Stream {
    // type of the current message being processed
    uint8_t msgType; // 0=unset, 1=request, 2=response

    // cJSON node for the content for the event's body.data
    cJSON *jsonData;

    // data from the last HTTP/2 request on the stream
    uint64_t lastRequestAt;       // hi-res timer value (nsecs)
    int      lastStatus;          // ":status" integer value; 200, 404
    char     lastHost[256];       // ":authority" value; server's hostname
    char     lastMethod[8];       // ":method" value; GET, POST, etc.
    char     lastTarget[2048];    // ":target" value; the URI
    char     lastUserAgent[1024]; // "user-agent" value"; mozilla

    // req/resp content-length values
    int lastReqLen;
    int lastRespLen;
} http2Stream_t;

// list of http2Channel_t indexed by channel
static list_t *g_http2_channels = NULL;

#define DEFAULT_MIN_DURATION_TIME (1)

static void
destroyHttpMap(void *data)
{
    if (!data) return;
    http_map *map = (http_map *)data;

    if (map->req) scope_free(map->req);
    if (map->resp) scope_free(map->resp);
    if (map) scope_free(map);
}

static void
destroyHttp2Channel(void *data)
{
    if (!data) return;
    http2Channel_t *info = (http2Channel_t *)data;

    lshpack_dec_cleanup(&info->decoder);
    if (info->streams) {
        lstDestroy(&info->streams);
    }

    scope_free(info);
}

static void
destroyHttp2Stream(void *data)
{
    if (!data) return;
    http2Stream_t *info = (http2Stream_t *)data;

    if (info->jsonData) {
        cJSON_Delete(info->jsonData);
    }

    scope_free(info);
}

void
initReporting()
{
    // NB: Each of these does a dynamic allocation that we are not releasing.
    //     They don't grow and would ideally be released when the reporting
    //     thread exits but we've not gotten to it yet. 
    g_maplist = lstCreate(destroyHttpMap);
    g_http2_channels = lstCreate(destroyHttp2Channel);
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
        scopeLogError("ERROR: sendEvent:cmdSendMetric");
    }
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
    if (proto->data) {
        // comment above doesn't apply to HTTP/2 protocol events
        if (proto->ptype == EVT_H2FRAME) {
            http_post *post = (http_post *)proto->data;
            scope_free(post->hdr);
        }
        scope_free(proto->data);
    }
}

static int
getProtocol(int type, char *proto, size_t len)
{
    if (!proto) {
        return -1;
    }

    if (type == SOCK_STREAM) {
        scope_strncpy(proto, "TCP", len);
    } else if (type == SOCK_DGRAM) {
        scope_strncpy(proto, "UDP", len);
    } else if (type == SOCK_RAW) {
        scope_strncpy(proto, "RAW", len);
    } else if (type == SOCK_RDM) {
        scope_strncpy(proto, "RDM", len);
    } else if (type == SOCK_SEQPACKET) {
        scope_strncpy(proto, "SEQPACKET", len);
    } else {
        scope_strncpy(proto, "OTHER", len);
    }

    return 0;
}

static bool
getConn(struct sockaddr_storage *conn, char *addr, size_t alen, char *port, size_t plen)
{
    if (!conn || !addr || !port) return FALSE;

    if (conn->ss_family == AF_INET) {
        if (scope_inet_ntop(AF_INET, &((struct sockaddr_in *)conn)->sin_addr,
                      addr, alen) == NULL) {
                return FALSE;
        }

        scope_snprintf(port, plen, "%d", scope_ntohs(((struct sockaddr_in *)conn)->sin_port));
    } else if (conn->ss_family == AF_INET6) {
        if (scope_inet_ntop(AF_INET6, &((struct sockaddr_in6 *)conn)->sin6_addr,
                      addr, alen) == NULL) {
                return  FALSE;
        }

        scope_snprintf(port, plen, "%d", scope_ntohs(((struct sockaddr_in6 *)conn)->sin6_port));
    } else {
        return FALSE;
    }
    return TRUE;
}

// yeah, a lot of params. but, it's generic.
static bool
getNetInternals(net_info *net, int type,
                struct sockaddr_storage *lconn, struct sockaddr_storage *rconn,
                char *laddr, char *raddr, size_t alen, char *lport, char *rport, size_t plen,
                event_field_t *fields, int *ix, int maxfld)
{
    if (!lconn || !rconn || !laddr || !raddr || !fields || !ix) return FALSE;

    if (addrIsNetDomain(lconn)) {
        switch (type) {
        case SOCK_STREAM:
            H_ATTRIB(fields[*ix], "net_transport", "IP.TCP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_DGRAM:
            H_ATTRIB(fields[*ix], "net_transport", "IP.UDP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RAW:
            H_ATTRIB(fields[*ix], "net_transport", "IP.RAW", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RDM:
            H_ATTRIB(fields[*ix], "net_transport", "IP.RDM", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_SEQPACKET:
            H_ATTRIB(fields[*ix], "net_transport", "IP.SEQPACKET", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        default:
            break;
        }

        if (getConn(rconn, raddr, alen, rport, plen) == TRUE) {
            in_port_t pport = scope_ntohs(((struct sockaddr_in *)rconn)->sin_port);
            H_ATTRIB(fields[*ix], "net_peer_ip", raddr, 1);
            NEXT_FLD(*ix, maxfld);
            H_VALUE(fields[*ix], "net_peer_port", pport, 1);
            NEXT_FLD(*ix, maxfld);
        }

        if (getConn(lconn, laddr, alen, lport, plen) == TRUE) {
            in_port_t hport = scope_ntohs(((struct sockaddr_in *)lconn)->sin_port);
            H_ATTRIB(fields[*ix], "net_host_ip", laddr, 1);
            NEXT_FLD(*ix, maxfld);
            H_VALUE(fields[*ix], "net_host_port", hport, 1);
            NEXT_FLD(*ix, maxfld);
        }
    } else if (addrIsUnixDomain(lconn)) {
        switch (type) {
        case SOCK_STREAM:
            H_ATTRIB(fields[*ix], "net_transport", "Unix.TCP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_DGRAM:
            H_ATTRIB(fields[*ix], "net_transport", "Unix.UDP", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RAW:
            H_ATTRIB(fields[*ix], "net_transport", "Unix.RAW", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_RDM:
            H_ATTRIB(fields[*ix], "net_transport", "Unix.RDM", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        case SOCK_SEQPACKET:
            H_ATTRIB(fields[*ix], "net_transport", "Unix.SEQPACKET", 1);
            NEXT_FLD(*ix, maxfld);
            break;
        default:
            break;
        }

        if (net) {
            H_VALUE(fields[*ix], "unix_peer_inode", net->rnode, 1);
            NEXT_FLD(*ix, maxfld);
            H_VALUE(fields[*ix], "unix_local_inode", net->lnode, 1);
            NEXT_FLD(*ix, maxfld);
        }
    }

    if (net && net->dnsName[0]) {
        H_ATTRIB(fields[*ix], "net_peer_name", net->dnsName, 1);
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

    if ((ix < 0) || (ix > len) || ((ix + scope_strlen(HTTP_STATUS) + 1) > len)) return -1;

    val = &header[ix + scope_strlen(HTTP_STATUS) + 1];
    // note that the spec defines the status code to be exactly 3 chars/digits
    *stext = &header[ix + scope_strlen(HTTP_STATUS) + 6];

    scope_errno = 0;
    rc = scope_strtoull(val, NULL, 0);
    if ((scope_errno != 0) || (rc == 0)) {
        return -1;
    }
    return rc;
}

static void
httpFieldEnd(event_field_t *fields, int ix)
{
    fields[ix].name = NULL;
    fields[ix].value_type = FMT_END;
    fields[ix].value.str = NULL;
    fields[ix].cardinality = 0;
}

static bool
headerMatch(regex_t *re, const char *match)
{
    if (!re || !match) return FALSE;

    if (!regexec_wrapper(re, match, 0, NULL, 0)) return TRUE;

    return FALSE;
}

static bool
httpFields(event_field_t *fields, http_report *hreport, char *hdr,
           size_t hdr_len, protocol_info *proto, config_t *cfg)
{
    if (!fields || !hreport || !proto || !hdr) return FALSE;

    // Start with fields from the header
    char *savea = NULL, *header;
    size_t numExtracts;

    hreport->clen = -1;

    if ((hreport->ptype == EVT_HREQ) && (hreport->hreq)) {
        scope_strncpy(hreport->hreq, hdr, hdr_len);
        header = hreport->hreq;
    } else if ((hreport->ptype == EVT_HRES) && (hreport->hres)) {
        scope_strncpy(hreport->hres, hdr, hdr_len);
        header = hreport->hres;
    } else {
        scopeLogWarn("fd:%d WARN: httpFields: proto ptype is not req or resp", proto->fd);
        return FALSE;
    }

    char *thishdr = scope_strtok_r(header, "\r\n", &savea);
    if (!thishdr) {
        scopeLogWarn("fd:%d WARN: httpFields: parse an http header", proto->fd);
        return FALSE;
    }

    while ((thishdr = scope_strtok_r(NULL, "\r\n", &savea)) != NULL) {
        // From RFC 2616 Section 4.2 "Field names are case-insensitive."
        if (scope_strcasestr(thishdr, "Host:")) {
            H_ATTRIB(fields[hreport->ix], "http_host", scope_strchr(thishdr, ':') + 2, 1);
            HTTP_NEXT_FLD(hreport->ix);
        } else if (scope_strcasestr(thishdr, "User-Agent:")) {
            H_ATTRIB(fields[hreport->ix], "http_user_agent", scope_strchr(thishdr, ':') + 2, 5);
            HTTP_NEXT_FLD(hreport->ix);
        } else if (scope_strcasestr(thishdr, "X-Forwarded-For:")) {
            H_ATTRIB(fields[hreport->ix], "http_client_ip", scope_strchr(thishdr, ':') + 2, 5);
            HTTP_NEXT_FLD(hreport->ix);
        } else if (scope_strcasestr(thishdr, "Content-Length:")) {
            scope_errno = 0;
            if (((hreport->clen = scope_strtoull(scope_strchr(thishdr, ':') + 2, NULL, 0)) == 0) || (scope_errno != 0)) {
                hreport->clen = -1;
            }
        } else if (scope_strcasestr(thishdr, "x-appscope:")) {
                H_ATTRIB(fields[hreport->ix], "x-appscope", scope_strchr(thishdr, ':') + 2, 5);
                HTTP_NEXT_FLD(hreport->ix);
        } else if ((numExtracts = cfgEvtFormatNumHeaders(cfg)) > 0) {
            int i;

            for (i = 0; i < numExtracts; i++) {
                regex_t *re;

                if (((re = cfgEvtFormatHeaderRe(cfg, i)) != NULL) &&
                    (headerMatch(re, thishdr) == TRUE)) {
                    char *evsrc = scope_strchr(thishdr, ':');

                    if (evsrc) {
                        *evsrc = '\0';
                        H_ATTRIB(fields[hreport->ix], thishdr, evsrc + 2, 5);
                        HTTP_NEXT_FLD(hreport->ix);
                    }
                }
            }
        }
    }

    return TRUE;
}

static bool
httpFieldsInternal(event_field_t *fields, http_report *hreport, protocol_info *proto)
{
    /*
    Compression and getting to an attribute with compressed and uncompressed lengths.
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
    if (proto->sock_type != -1) {
        getNetInternals(NULL, proto->sock_type,
                        &proto->localConn, &proto->remoteConn,
                        hreport->laddr, hreport->raddr, sizeof(hreport->raddr),
                        hreport->lport, hreport->rport, sizeof(hreport->rport),
                        fields, &hreport->ix, HTTP_MAX_FIELDS);
    }

    return TRUE;
}

static void
doHttp1Header(protocol_info *proto)
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
        if ((map = scope_calloc(1, sizeof(http_map))) == NULL) {
            destroyProto(proto);
            return;
        }

        if (lstInsert(g_maplist, post->id, map) == FALSE) {
            destroyHttpMap(map);
            destroyProto(proto);
            return;
        }

        struct timeval tv;
        scope_gettimeofday(&tv, NULL);

        map->id = post->id;
        map->first_time = tv.tv_sec;
        map->req = NULL;
        map->req_len = 0;
    }

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
        if ((hreport.hreq = scope_calloc(1, map->req_len)) == NULL) {
            scopeLogError("fd:%d ERROR: doHttp1Header: hreq memory allocation failure", proto->fd);
            return;
        }

        char *savea = NULL;
        scope_strncpy(header, map->req, map->req_len);

        char *headertok = scope_strtok_r(header, "\r\n", &savea);
        if (!headertok) {
            scope_free(hreport.hreq);
            scopeLogWarn("fd:%d WARN: doHttp1Header: parse an http request header", proto->fd);
            return;
        }

        // The request specific values from Request-Line
        char *method_str = scope_strtok_r(headertok, " ", &savea);
        if (method_str) {
            H_ATTRIB(fields[hreport.ix], "http_method", method_str, 1);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLogWarn("fd:%d WARN: doHttp1Header: no method in an http request header", proto->fd);
        }

        char *target_str = scope_strtok_r(NULL, " ", &savea);
        if (target_str) {
            H_ATTRIB(fields[hreport.ix], "http_target", target_str, 4);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLogWarn("fd:%d WARN: doHttp1Header: no target in an http request header", proto->fd);
        }

        char *flavor_str = scope_strtok_r(NULL, " ", &savea);
        if (flavor_str &&
            ((flavor_str = scope_strtok_r(flavor_str, "/", &savea))) &&
            ((flavor_str = scope_strtok_r(NULL, "\r", &savea)))) {
            if (proto->ptype == EVT_HREQ) {
                H_ATTRIB(fields[hreport.ix], "http_flavor", flavor_str, 1);
                HTTP_NEXT_FLD(hreport.ix);
            }
        } else {
            scopeLogWarn("fd:%d WARN: doHttp1Header: no http version in an http request header", proto->fd);
        }

        H_ATTRIB(fields[hreport.ix], "http_scheme", ssl, 1);
        HTTP_NEXT_FLD(hreport.ix);

        if (proto->ptype == EVT_HREQ) {
            hreport.ptype = EVT_HREQ;
            // Fields common to request & response
            httpFields(fields, &hreport, map->req, map->req_len, proto, g_cfg.staticfg);
            httpFieldsInternal(fields, &hreport, proto);

            if (hreport.clen != -1) {
                H_VALUE(fields[hreport.ix], "http_request_content_length", hreport.clen, EVENT_ONLY_ATTR);
                HTTP_NEXT_FLD(hreport.ix);
            }
            map->clen = hreport.clen;

            httpFieldEnd(fields, hreport.ix);

            event_t sendEvent = INT_EVENT("http.req", proto->len, SET, fields);
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
        if ((hreport.hres = scope_calloc(1, proto->len)) == NULL) {
            scopeLogError("fd:%d ERROR: doHttp1Header: hres memory allocation failure", proto->fd);
            return;
        }

        struct timeval tv;
        scope_gettimeofday(&tv, NULL);

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
        scope_strncpy(reqheader, map->resp, proto->len);

        char *headertok = scope_strtok_r(reqheader, "\r\n", &savea);
        char *flavor_str = scope_strtok_r(headertok, " ", &savea);
        if (flavor_str &&
            ((flavor_str = scope_strtok_r(flavor_str, "/", &savea))) &&
            ((flavor_str = scope_strtok_r(NULL, "", &savea)))) {
            H_ATTRIB(fields[hreport.ix], "http_flavor", flavor_str, 1);
            HTTP_NEXT_FLD(hreport.ix);
        } else {
            scopeLogWarn("fd:%d WARN: doHttp1Header: no version string in an http request header", proto->fd);
        }

        H_VALUE(fields[hreport.ix], "http_status_code", status, 1);
        HTTP_NEXT_FLD(hreport.ix);

        // point past the status code
        char st[scope_strlen(stext)];
        scope_strncpy(st, stext, scope_strlen(stext));
        char *status_str = scope_strtok_r(st, "\r\n", &savea);
        // if no Reason-Phrase is provided, st will not be equal to status_str
        if (st != status_str) status_str = "";
        H_ATTRIB(fields[hreport.ix], "http_status_text", status_str, 1);
        HTTP_NEXT_FLD(hreport.ix);

        H_VALUE(fields[hreport.ix], "http_server_duration", map->duration, EVENT_ONLY_ATTR);
        HTTP_NEXT_FLD(hreport.ix);

        // Fields common to request & response
        if (map->req) {
            hreport.ptype = EVT_HREQ;
            httpFields(fields, &hreport, map->req, map->req_len, proto, g_cfg.staticfg);
            if (hreport.clen != -1) {
                H_VALUE(fields[hreport.ix], "http_request_content_length", hreport.clen, EVENT_ONLY_ATTR);
                HTTP_NEXT_FLD(hreport.ix);
            }
            map->clen = hreport.clen;
        }

        hreport.ptype = EVT_HRES;
        httpFields(fields, &hreport, map->resp, proto->len, proto, g_cfg.staticfg);
        httpFieldsInternal(fields, &hreport, proto);
        if (hreport.clen != -1) {
            H_VALUE(fields[hreport.ix], "http_response_content_length", hreport.clen, EVENT_ONLY_ATTR);
            HTTP_NEXT_FLD(hreport.ix);
        }

        httpFieldEnd(fields, hreport.ix);

        event_t hevent = INT_EVENT("http.resp", proto->len, SET, fields);
        cmdSendHttp(g_ctl, &hevent, map->id, &g_proc);

        // emit statsd metrics, if enabled.
        if ((mtcEnabled(g_mtc)) && (cfgMtcHttpEnable(g_cfg.staticfg))) {

            char *mtx_name = (proto->isServer) ? "http_server_duration" : "http_client_duration";
            event_t http_dur = INT_EVENT(mtx_name, map->duration, DELTA_MS, fields);
            // TBD AGG Only cmdSendMetric(g_mtc, &http_dur);
            httpAggAddMetric(g_http_agg, &http_dur, map->clen, hreport.clen);

            /* TBD AGG Only
            if (map->clen != -1) {
                event_t http_req_len = INT_EVENT("http.req.content_length", map->clen, DELTA, fields);
                cmdSendMetric(g_mtc, &http_req_len);
            }

            if (hreport.clen != -1) {
                event_t http_rsp_len = INT_EVENT("http.resp.content_length", hreport.clen, DELTA, fields);
                cmdSendMetric(g_mtc, &http_rsp_len);
            }
            */

        }

        // Done; we remove the list entry; complete when reported
        if (lstDelete(g_maplist, post->id) == FALSE) DBG(NULL);
    }

    if (hreport.hreq) scope_free(hreport.hreq);
    if (hreport.hres) scope_free(hreport.hres);
    destroyProto(proto);
}

static const char *
httpStatusCode2Text(int code)
{
    switch (code) {
        case 100:
            return "Continue";
        case 101:
            return "Switching Protocols";
        case 102:
            return "Processing";
        case 200:
            return "OK";
        case 201:
            return "Created";
        case 202:
            return "Accepted";
        case 203:
            return "Non-authoritative Information";
        case 204:
            return "No Content";
        case 205:
            return "Reset Content";
        case 206:
            return "Partial Content";
        case 207:
            return "Multi-Status";
        case 208:
            return "Already Reported";
        case 226:
            return "IM Used";
        case 300:
            return "Multiple Choices";
        case 301:
            return "Moved Permanently";
        case 302:
            return "Found";
        case 303:
            return "See Other";
        case 304:
            return "Not Modified";
        case 305:
            return "Use Proxy";
        case 307:
            return "Temporary Redirect";
        case 308:
            return "Permanent Redirect";
        case 400:
            return "Bad Request";
        case 401:
            return "Unauthorized";
        case 402:
            return "Payment Required";
        case 403:
            return "Forbidden";
        case 404:
            return "Not Found";
        case 405:
            return "Method Not Allowed";
        case 406:
            return "Not Acceptable";
        case 407:
            return "Proxy Authentication Required";
        case 408:
            return "Request Timeout";
        case 409:
            return "Conflict";
        case 410:
            return "Gone";
        case 411:
            return "Length Required";
        case 412:
            return "Precondition Failed";
        case 413:
            return "Payload Too Large";
        case 414:
            return "Request-URI Too Long";
        case 415:
            return "Unsupported Media Type";
        case 416:
            return "Requested Range Not Satisfiable";
        case 417:
            return "Expectation Failed";
        case 418:
            return "I'm a teapot";
        case 421:
            return "Misdirected Request";
        case 422:
            return "Unprocessable Entity";
        case 423:
            return "Locked";
        case 424:
            return "Failed Dependency";
        case 426:
            return "Upgrade Required";
        case 428:
            return "Precondition Required";
        case 429:
            return "Too Many Requests";
        case 431:
            return "Request Header Fields Too Large";
        case 444:
            return "Connection Closed Without Response";
        case 451:
            return "Unavailable For Legal Reasons";
        case 499:
            return "Client Closed Request";
        case 500:
            return "Internal Server Error";
        case 501:
            return "Not Implemented";
        case 502:
            return "Bad Gateway";
        case 503:
            return "Service Unavailable";
        case 504:
            return "Gateway Timeout";
        case 505:
            return "HTTP Version Not Supported";
        case 506:
            return "Variant Also Negotiates";
        case 507:
            return "Insufficient Storage";
    }

    return "UNKNOWN";
}

static bool
isHttp2NameEnabled(const char* name)
{
    regex_t *nameRe = evtFormatNameFilter(ctlEvtGet(g_ctl), CFG_SRC_HTTP);
    if (!nameRe) {
        scopeLogError("ERROR: missing name filter for HTTP watch");
        DBG("Missing name filter for HTTP watch");
        return FALSE;
    }
    return headerMatch(nameRe, name);
}

static bool
isHttp2FieldEnabled(const char* field, const char *value)
{
    regex_t *fieldRe = evtFormatFieldFilter(ctlEvtGet(g_ctl), CFG_SRC_HTTP);
    if (!fieldRe) {
        scopeLogError("ERROR: missing field filter for HTTP watch");
        DBG("Missing field filter for HTTP watch");
        return FALSE;
    }
    regex_t *valueRe = evtFormatValueFilter(ctlEvtGet(g_ctl), CFG_SRC_HTTP);
    if (!valueRe) {
        scopeLogError("ERROR: missing value filter for HTTP watch");
        DBG("Missing value filter for HTTP watch");
        return FALSE;
    }
    return headerMatch(fieldRe, field) && headerMatch(valueRe, value);
}

static void
addHttp2NumField(cJSON *jsonData, const char* field, uint32_t value)
{
    char buf[16];
    if (scope_snprintf(buf, sizeof(buf), "%d", value) >= sizeof(buf)) {
        scopeLogError("ERROR: failed to convert int to string");
        DBG("buf too small for uint32");
        return;
    }
    if (isHttp2FieldEnabled(field, buf)) {
        cJSON_AddNumberToObject(jsonData, field, value);
    }
}

static void
addHttp2StrField(cJSON *jsonData, const char* field, const char *value)
{
    if (isHttp2FieldEnabled(field, value)) {
        cJSON_AddStringToObject(jsonData, field, value);
    }
}

static void
addHttp2StrFieldLN(cJSON *jsonData, const char* field, const char *value)
{
    if (isHttp2FieldEnabled(field, value)) {
        cJSON_AddStringToObjLN(jsonData, field, value);
    }
}

static void
doHttp2Frame(protocol_info *proto)
{
    // require the protocol object and it's data pointer to be set
    if (!proto || !proto->data) {
        scopeLogError("ERROR: null proto or proto->data");
        destroyProto(proto);
        return;
    }

    // require the post's hdr pointer to be set
    http_post *post = (http_post *)proto->data;
    if (!post->hdr) {
        scopeLogError("ERROR: null post->hdr");
        destroyProto(proto);
        return;
    }

    // extract the HTTP/2 frame header
    const uint8_t *frame = (uint8_t *)post->hdr;
    if (proto->len < 9) {
        scopeLogHexError(frame, proto->len,
                "ERROR: runt HTTP/2 frame; only %ld bytes long", proto->len);
        DBG(NULL);
        goto cleanup;
    }
    uint32_t fLen    = (frame[0]<<16) + (frame[1]<<8) + (frame[2]);
    uint8_t  fType   = frame[3];
    uint8_t  fFlags  = frame[4];
    uint32_t fStream = ((frame[5]&0x7F)<<24) + (frame[6]<<16) + (frame[7]<<8) + (frame[8]);
    if (proto->len != 9 + fLen) {
        scopeLogHexError(frame, proto->len,
                "ERROR: bad HTTP/2 frame size; got %ld/%d bytes", proto->len, 9+fLen);
        DBG(NULL);
        goto cleanup;
    }

    //scopeLogHexDebug(frame, proto->len,
    //        "DEBUG: HTTP/2 frame; chan=0x%lx, stream=%d, type=0x%02x, flags=0x%02x",
    //        proto->uid, fStream, fType, fFlags);

    if (fType == 0x05) {
        // PUSH_PROMISE frames are analogous to unsolicited request messages
        // for future responses the client didn't directly ask for. They are
        // typically sent by the server when it's smart enough to identify
        // assets the client will need or for async messaging.
        //
        // They will be sent in the stream that triggered them but the HEADERS
        // and DATA frames for the pushed response will be on a separate stream
        // identified in the PUSH_PROMISE frame. We're overriding fStream here
        // so the header block in the PUSH_PROMISE are processed as if it were
        // a request on the stream where the response will come later.
        //
        // The stream ID to use is right after the headers or one byte further if
        // the PADDED flag is set.
        size_t offset = (fFlags & 0x08) ? 10 : 9;
        fStream = ((frame[offset+0]&0x7F) << 24) // first bit is reserved
                + ( frame[offset+1]       << 16)
                + ( frame[offset+2]       <<  8)
                + ( frame[offset+3]            );
    }

    if (fType == 0x01 || fType == 0x05 || fType == 0x09) {
        // Process HEADERS(1), PUSH_PROMISE(5), or CONTINUATION(9) frames. All
        // three contain a header block; an HPACK-encoded lists of key/value
        // headers.
        //
        // HEADERS frames correspond to a request or response depending on the
        // fields in the header block; ":method" indicates request, ":status"
        // indicates a response.
        //
        // PUSH_PROMISE frames are like request message headers except they are
        // sent by the server when it decides to push a message without a
        // request from the client. See the notes earlier about how the stream
        // ID in the frame header and the one in the body differ.
        //
        // If the END_HEADERS flag in any of these three frame types IS NOT
        // set, they are followied immediately but a CONTINUATION frame that
        // contains additional headers. These are used when the headers don't
        // fit in the earlier frame(s). When the flag is set, it's the end of 
        // the message.

        // get/create the channel info
        http2Channel_t *channel = lstFind(g_http2_channels, proto->uid);
        if (!channel) {
            channel = scope_calloc(1, sizeof(http2Channel_t));
            if (!channel) {
                scopeLogError("ERROR: failed to create channel info");
                DBG(NULL);
                goto cleanup;
            }

            lshpack_dec_init(&channel->decoder);
            lshpack_dec_set_max_capacity(&channel->decoder, 0x4000);
            channel->streams = lstCreate(destroyHttp2Stream);

            if (lstInsert(g_http2_channels, proto->uid, channel) != TRUE) {
                destroyHttp2Channel(channel);
                scopeLogError("ERROR: failed to insert channel");
                DBG(NULL);
                goto cleanup;
            }
        }

        // get/create the stream info
        http2Stream_t *stream = lstFind(channel->streams, fStream);
        if (!stream) {
            stream = scope_calloc(1, sizeof(http2Stream_t));
            if (!stream) {
                scopeLogError("ERROR: failed to create http2Stream");
                DBG(NULL);
                goto cleanup;
            }

            if (lstInsert(channel->streams, fStream, stream) != TRUE) {
                destroyHttp2Stream(stream);
                scopeLogError("ERROR: failed to insert decoder");
                DBG(NULL);
                goto cleanup;
            }
        }

        // Rather than keep an event_field_t array like the HTTP/1 logic does,
        // we're building the cJSON object for the event directly. The way the
        // headers are unpacked one at a time here makes it combersome to save
        // off the results so we're skipping the extra layer. It means we need
        // to reproduce the watch filter logic and the http.* metrics updates
        // ourselves though. This is the cJSON object we're building.
        if (!stream->jsonData) {
            stream->jsonData = cJSON_CreateObject();
            if (!stream->jsonData) {
                scopeLogError("ERROR: failed to create jsonData");
                DBG(NULL);
                goto cleanup;
            }

            // Hard coding 2.0 for now
            cJSON_AddStringToObjLN(stream->jsonData, "http_flavor", "2.0");
        }

        // The position in the frame where the header data is depends on the
        // type and flags. Initially, we start just after the frame header and
        // end at the end of the frame then adjust. See below.
        const uint8_t *decPos = frame + 9;
        const uint8_t *decEnd = decPos + fLen;
        if (fFlags & 0x08) {
            // When the PADDED flag is set, we need to skip over one byte at
            // the start. The value of that byte is the number at the end to
            // skip too.
            decPos += 1;
            decEnd -= frame[9];
        }
        if (fFlags & 0x20) {
            // When the PRIORITY flag is set, we need to skip over 5 bytes
            // at the start.
            decPos += 5;
        }
        if (fType == 0x05) {
            // We skip the stream ID at the start of PUSH_PROMISE frames.
            decPos += 4;
        }

        // Now loop through the header data letting the HPACK decoder put the
        // field name and value into the buffer we provide. The size of the
        // buffer is taken from examples in the lshpack package. Not sure
        // exactly what it should be.
        char out[2048];
        lsxpack_header_t hdr;
        while (decPos < decEnd) {
            lsxpack_header_prepare_decode(&hdr, out, 0, sizeof(out));
            int rc = lshpack_dec_decode(&channel->decoder, &decPos, decEnd, &hdr);
            if (rc != 0) {
                scopeLogError("ERROR: HTTP/2 decoder failed; err=%d", rc);
                scopeLogHexError(frame, proto->len,
                        "  chan=0x%lx, stream=%d, type=0x%02x, flags=0x%02x",
                        proto->uid, fStream, fType, fFlags);
                scopeLogHexError(decPos, decEnd-decPos,
                        "  decoder failed here");
                DBG(NULL);
                break;
            }

            // get the field name and value as null-terminated strings
            out[hdr.name_offset + hdr.name_len] = '\0';
            out[hdr.val_offset  + hdr.val_len ] = '\0';
            char *name = out + hdr.name_offset;
            char *val  = out + hdr.val_offset;
            //scopeLogDebug("DEBUG: HTTP/2 decoded header: name=\"%s\", value=\"%s\"", name, val);

            // Update the state of the stream for the given header field. Most
            // of these become entries in the cJSON object that will eventually
            // become the body.data element in the JSON event. Some are stashed
            // into the state object for use later.
            if (!scope_strcasecmp(":method", name)) {
                // We use the presence of the :method header to indicate we're
                // processing a request message.
                stream->msgType = 1;
                addHttp2NumField(stream->jsonData, "http_stream", fStream);

                addHttp2StrField(stream->jsonData, "http_method", val);
                scope_strncpy(stream->lastMethod, val, sizeof(stream->lastMethod));

                // record the start timestamp for duration calculations
                stream->lastRequestAt = post->start_duration;

                // record the frame type in request events so we can see Server Push
                if (fType == 0x01) {
                    addHttp2StrFieldLN(stream->jsonData, "http_frame", "HEADERS");
                } else if (fType == 0x05) {
                    addHttp2StrFieldLN(stream->jsonData, "http_frame", "PUSH_PROMISE");
                }
            } else if (!scope_strcasecmp(":status", name)) {
                // We use the presence of the :status header to indicate we're
                // processing a response message.
                stream->msgType = 2; // response
                addHttp2NumField(stream->jsonData, "http_stream", fStream);

                stream->lastStatus = scope_atoi(val);
                addHttp2NumField(stream->jsonData, "http_status_code", stream->lastStatus);
                addHttp2StrField(stream->jsonData, "http_status_text", httpStatusCode2Text(stream->lastStatus));
            } else if (!scope_strcasecmp(":authority", name)) {
                addHttp2StrField(stream->jsonData, "http_host", val);
                scope_strncpy(stream->lastHost, val, sizeof(stream->lastHost));
            } else if (!scope_strcasecmp(":path", name)) {
                addHttp2StrField(stream->jsonData, "http_target", val);
                scope_strncpy(stream->lastTarget, val, sizeof(stream->lastTarget));
            } else if (!scope_strcasecmp(":scheme", name)) {
                addHttp2StrField(stream->jsonData, "http_scheme", val);
            } else if (!scope_strcasecmp("user-agent", name)) {
                addHttp2StrField(stream->jsonData, "http_user_agent", val);
                scope_strncpy(stream->lastUserAgent, val, sizeof(stream->lastUserAgent));
            } else if (!scope_strcasecmp("x-appscope", name)) {
                addHttp2StrField(stream->jsonData, "x-appscope", val);
            } else if (!scope_strcasecmp("x-forwarded-for", name)) {
                addHttp2StrField(stream->jsonData, "http_client_ip", val);
            } else if (!scope_strcasecmp("content-length", name)) {
                if (stream->msgType == 1) {
                    stream->lastReqLen = scope_atoi(val);
                    addHttp2NumField(stream->jsonData, "http_request_content_length", stream->lastReqLen);
                } else if (stream->msgType == 2) {
                    stream->lastRespLen = scope_atoi(val);
                    addHttp2NumField(stream->jsonData, "http_response_content_length", stream->lastRespLen);
                } else {
                    scopeLogError("ERROR: invalid msgType; %d", stream->msgType);
                    DBG(NULL);
                }
            } else {
                // All other header fields need to match a regex in the
                // event.watch[name=http].headers array in the runtime config.
                //
                // Note that the filter is applied to the header's `name:
                // value` form, not the name and value separately. We're munging the
                // buffer temporarily here.
                out[hdr.name_offset + hdr.name_len] = ':';
                size_t i;
                size_t numHeaders = cfgEvtFormatNumHeaders(g_cfg.staticfg);
                for (i = 0; i < numHeaders; ++i) {
                    regex_t *re = cfgEvtFormatHeaderRe(g_cfg.staticfg, i);
                    if (re) {
                        if (!regexec_wrapper(re, out, 0, NULL, 0)) {
                            // matched, add it and skip the rest of the filters
                            out[hdr.name_offset + hdr.name_len] = '\0';
                            cJSON_AddStringToObject(stream->jsonData, name, val);
                            break;
                        }
                    }
                }
                out[hdr.name_offset + hdr.name_len] = '\0';
            }
        }

        // The END_HEADERS flag is set when there are no (more) CONTINUATION
        // frames coming. It indicates to us that we've got the whole message
        // so now we need to generate the event.
        if (fFlags & 0x04) {

            // The isServer value in the protocol object is only half of the
            // answer here. See reportHttp2() in httpstate.c for details. We're
            // recreating the isServer value that the HTTP/1 logic produces
            // because we don't crack the headers on the data side and
            // therefore don't know if it's a request or response.
            bool isSend     = proto->isServer;
            bool isResponse = stream->msgType == 2;
            bool isServer   = (isSend && isResponse) || (!isSend && !isResponse);

            // Add the socket info fields
            if (addrIsNetDomain(&proto->localConn)) {
                switch (proto->sock_type) {
                    case SOCK_STREAM:
                        addHttp2StrFieldLN(stream->jsonData, "net_transport", "IP.TCP");
                        break;
                    case SOCK_DGRAM:
                        addHttp2StrFieldLN(stream->jsonData, "net_transport", "IP.UDP");
                        break;
                    case SOCK_RAW:
                        addHttp2StrFieldLN(stream->jsonData, "net_transport", "IP.RAW");
                        break;
                    case SOCK_RDM:
                        addHttp2StrFieldLN(stream->jsonData, "net_transport", "IP.RDM");
                        break;
                    case SOCK_SEQPACKET:
                        addHttp2StrFieldLN(stream->jsonData, "net_transport", "IP.SEQPACKET");
                        break;
                }

                char addr[INET6_ADDRSTRLEN];
                if (scope_inet_ntop(proto->remoteConn.ss_family,
                            &((struct sockaddr_in*)&proto->remoteConn)->sin_addr, addr, sizeof(addr))) {
                    addHttp2StrField(stream->jsonData, "net_peer_ip", addr);
                }
                if (scope_inet_ntop(proto->localConn.ss_family,
                            &((struct sockaddr_in*)&proto->localConn)->sin_addr, addr, sizeof(addr))) {
                    addHttp2StrField(stream->jsonData, "net_host_ip", addr);
                }

                addHttp2NumField(stream->jsonData, "net_peer_port", scope_ntohs(((struct sockaddr_in*)&proto->remoteConn)->sin_port));
                addHttp2NumField(stream->jsonData, "net_host_port", scope_ntohs(((struct sockaddr_in*)&proto->localConn)->sin_port));
            }

            // If it's a request message...
            if (stream->msgType == 1) {
                if (isHttp2NameEnabled("http.req")) {
                    // send the request event
                    event_t event = INT_EVENT("http.req", proto->len, SET, NULL);
                    event.data = stream->jsonData;
                    cmdSendHttp(g_ctl, &event, proto->uid, &g_proc);
                }
            } 

            // if it's a response message...
            else if (stream->msgType == 2) {
                // we may need the HTTP/1 state 
                http_map *map = lstFind(g_maplist, post->id);

                // add duration from request
                unsigned duration; // msecs
                if (stream->lastRequestAt) {
                    duration = (post->start_duration - stream->lastRequestAt) / 1000000;
                } else if (map && map->start_time) {
                    duration = (post->start_duration - map->start_time) / 1000000;
                }
                addHttp2NumField(stream->jsonData, isServer ?  "http_server_duration" : "http_client_duration", duration);

                // add host from request
                if (stream->lastHost[0]) {
                    addHttp2StrField(stream->jsonData, "http_host", stream->lastHost);
                } else if (map) {
                    // TODO: HTTP/1->2 upgrade, get value from HTTP/1 request
                }

                // add method from request
                if (stream->lastMethod[0]) {
                    addHttp2StrField(stream->jsonData, "http_method", stream->lastMethod);
                } else if (map) {
                    // TODO: HTTP/1->2 upgrade, get value from HTTP/1 request
                }

                // add target URL from request
                if (stream->lastTarget[0]) {
                    addHttp2StrField(stream->jsonData, "http_target", stream->lastTarget);
                } else if (map) {
                    // TODO: HTTP/1->2 upgrade, get value from HTTP/1 request
                }

                // add user-agent from request
                if (stream->lastUserAgent[0]) {
                    addHttp2StrField(stream->jsonData, "http_user_agent", stream->lastUserAgent);
                } else if (map) {
                    // TODO: HTTP/1->2 upgrade, get value from HTTP/1 request
                }

                if (isHttp2NameEnabled("http.resp")) {
                    // send the response event
                    event_t event = INT_EVENT("http.resp", proto->len, SET, NULL);
                    event.data = stream->jsonData;
                    cmdSendHttp(g_ctl, &event, proto->uid, &g_proc);
                }

                // if metrics are enabled...
                if (mtcEnabled(g_mtc) && (cfgMtcHttpEnable(g_cfg.staticfg))) {
                    // update HTTP metrics
                    event_field_t fields[] = {
                        STRFIELD("http_target", stream->lastTarget, 4, TRUE),
                        NUMFIELD("http_status_code", stream->lastStatus, 1, TRUE),
                        FIELDEND
                    };
                    event_t httpMetric = INT_EVENT(
                            isServer ? "http_server_duration" : "http_client_duration",
                            duration, DELTA_MS, fields);
                    httpAggAddMetric(g_http_agg, &httpMetric, 
                            (stream->lastReqLen > 0) ? stream->lastReqLen : -1,
                            (stream->lastRespLen > 0) ? stream->lastRespLen : -1);
                }
            }

            // otherwise, the message type is invalid
            else {
                scopeLogError("ERROR: HTTP/2 invalid msgType; %d", stream->msgType);
                DBG(NULL);
            }

            // reset
            stream->msgType = 0;
            if (stream->jsonData) {
                // jsonData was deleted for us down in cmdSendHttp()
                //cJSON_Delete(stream->jsonData);
                stream->jsonData = NULL;
            }
        }
    } else {
        scopeLogError("ERROR: HTTP/2 unexpected frame type; type=0x%02d", fType);
        DBG(NULL);
    }

cleanup:
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

    event_t evt = INT_EVENT("net.app", proto->fd, SET, fields);
    evt.src = CFG_SRC_NET;
    cmdSendEvent(g_ctl, &evt, proto->uid, &g_proc);
    destroyProto(proto);
}

void
doProtocolMetric(protocol_info *proto)
{
    if (!proto) return;
    if ((proto->ptype == EVT_HREQ) || (proto->ptype == EVT_HRES)) {
        doHttp1Header(proto);
    } else if (proto->ptype == EVT_H2FRAME) {
        doHttp2Frame(proto);
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
    metric_counters* ctrs = (ctr) ? (metric_counters*) ctr : &g_ctrs;

    const char err_name[] = "EFAULT";
    if (scope_errno == EFAULT) {
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

        event_field_t summary = SUMMARY_FIELD("true");
        event_field_t fieldend = FIELDEND;
        // if func and name are both null, this is a summary
        event_field_t* conditional_field = (func || name) ? &fieldend : &summary;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(func),
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            *conditional_field, // either a SUMMARY_FIELD or a FIELDEND
            FIELDEND
        };

        // Don't report zeros.
        if (value->evt != 0ULL) {
             event_t netErrMetric = INT_EVENT("net.error", value->evt, DELTA, fields);
             cmdSendEvent(g_ctl, &netErrMetric, getTime(), &g_proc);
             atomicSwapU64(&value->evt, 0);
        }

        // Only report if metrics enabled
        if ((!cfgMtcNetEnable(g_cfg.staticfg)) ||
           ((g_summary.net.error) && (source == EVENT_BASED)) ||
           ((!g_summary.net.error) && (source == PERIODIC))) {
            return;
        }
        // Don't report zeros.
        if (value->mtc == 0) return;

        event_t netErrMetric = INT_EVENT("net.error", value->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &netErrMetric)) {
            scopeLogError("ERROR: doErrorMetric:NET:cmdSendMetric");
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

        event_field_t summary = SUMMARY_FIELD("true");
        event_field_t fieldend = FIELDEND;
        // if func and name are both null, this is a summary
        event_field_t* conditional_field = (func || name) ? &fieldend : &summary;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(func),
            *name_field,
            CLASS_FIELD(class),
            UNIT_FIELD("operation"),
            *conditional_field, // either a SUMMARY_FIELD or a FIELDEND
            FIELDEND
        };

        // Don't report zeros.
        if (value->evt != 0ULL) {
            event_t fsErrMetric = INT_EVENT(metric, value->evt, DELTA, fields);
            cmdSendEvent(g_ctl, &fsErrMetric, getTime(), &g_proc);
            atomicSwapU64(&value->evt, 0);
        }

        // Only report if metrics enabled
        if ((!cfgMtcFsEnable(g_cfg.staticfg)) ||
            ((*summarize) && (source == EVENT_BASED)) ||
            ((!*summarize) && (source == PERIODIC))) {
            return;
        }

        // Don't report zeros.
        if (value->mtc == 0) return;

        event_t fsErrMetric = INT_EVENT(metric, value->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &fsErrMetric)) {
            scopeLogError("ERROR: doErrorMetric:FS_ERR:cmdSendMetric");
        }
        atomicSwapU64(&value->mtc, 0);
        break;
    }

    default:
        scopeLogError("ERROR: doErrorMetric:metric type");
    }
}

void
doDNSMetricName(metric_t type, net_info *net)
{
    if (!net || !net->dnsName || !net->dnsName[0]) return;

    metric_counters *ctrs = &net->counters;
    counters_element_t *duration = &net->totalDuration;

    switch (type) {
    case DNS:
    {
        // Don't report zeros.
        if (ctrs->numDNS.evt != 0) {
            // This creates a DNS raw event
            if (duration && (duration->evt > 0)) {
                event_field_t resp[] = {
                    PROC_FIELD(g_proc.procname),
                    PID_FIELD(g_proc.pid),
                    HOST_FIELD(g_proc.hostname),
                    DOMAIN_FIELD(net->dnsName),
                    UNIT_FIELD("response"),
                    FIELDEND
                };
                event_t dnsMetric = INT_EVENT("dns.resp", ctrs->numDNS.evt, DELTA, resp);
                cmdSendEvent(g_ctl, &dnsMetric, getTime(), &g_proc);

                // This creates a DNS event
                event_field_t evfield[] = {
                    DOMAIN_FIELD(net->dnsName),
                    DURATION_FIELD(duration->evt / 1000000), // convert ns to ms.
                    FIELDEND
                };
                event_t dnsEvent = INT_EVENT("dns.resp", ctrs->numDNS.evt, DELTA, evfield);
                dnsEvent.src = CFG_SRC_DNS;
                dnsEvent.data = net->dnsAnswer;
                cmdSendEvent(g_ctl, &dnsEvent, getTime(), &g_proc);
            } else {
                // This create a DNS raw event
                event_field_t req[] = {
                    PROC_FIELD(g_proc.procname),
                    PID_FIELD(g_proc.pid),
                    HOST_FIELD(g_proc.hostname),
                    DOMAIN_FIELD(net->dnsName),
                    UNIT_FIELD("request"),
                    FIELDEND
                };
                event_t dnsMetric = INT_EVENT("dns.req", ctrs->numDNS.evt, DELTA, req);
                cmdSendEvent(g_ctl, &dnsMetric, getTime(), &g_proc);

                // This creates a DNS event
                event_field_t evfield[] = {
                    DOMAIN_FIELD(net->dnsName),
                    FIELDEND
                };
                event_t dnsEvent = INT_EVENT("dns.req", ctrs->numDNS.evt, DELTA, evfield);
                dnsEvent.src = CFG_SRC_DNS;
                cmdSendEvent(g_ctl, &dnsEvent, getTime(), &g_proc);
            }
        }

        // Only report if metrics enabled
        if ((g_summary.net.dns) || (!cfgMtcDnsEnable(g_cfg.staticfg))) {
            return;
        }

        // Don't report zeros.
        if (ctrs->numDNS.mtc == 0) return;

        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            DOMAIN_FIELD(net->dnsName),
            DURATION_FIELD(duration->mtc / 1000000), // convert ns to ms.
            UNIT_FIELD("request"),
            FIELDEND
        };
        event_t dnsMetric = INT_EVENT("dns.req", ctrs->numDNS.mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &dnsMetric)) {
            scopeLogError("ERROR: doDNSMetricName:DNS:cmdSendMetric");
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
                DOMAIN_FIELD(net->dnsName),
                NUMOPS_FIELD(cachedDurationNum),
                UNIT_FIELD("millisecond"),
                FIELDEND
            };

            event_t dnsDurMetric = INT_EVENT("dns.duration", dur, DELTA_MS, fields);
            cmdSendEvent(g_ctl, &dnsDurMetric, getTime(), &g_proc);
            atomicSwapU64(&ctrs->dnsDurationNum.evt, 0);
            atomicSwapU64(&ctrs->dnsDurationTotal.evt, 0);
        }

        // Only report if metrics enabled
        if ((g_summary.net.dns) || (!cfgMtcDnsEnable(g_cfg.staticfg))) {
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
            DOMAIN_FIELD(net->dnsName),
            NUMOPS_FIELD(cachedDurationNum),
            UNIT_FIELD("millisecond"),
            FIELDEND
        };
        event_t dnsDurMetric = INT_EVENT("dns.duration", dur, DELTA_MS, fields);
        if (cmdSendMetric(g_mtc, &dnsDurMetric)) {
            scopeLogError("ERROR: doDNSMetricName:DNS_DURATION:cmdSendMetric");
        }
        atomicSwapU64(&ctrs->dnsDurationNum.mtc, 0);
        atomicSwapU64(&ctrs->dnsDurationTotal.mtc, 0);
        break;
    }

    default:
        scopeLogError("ERROR: doDNSMetric:metric type");
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
        event_t event = INT_EVENT("proc.mem", measurement, CURRENT, fields);
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
        scopeLogError("ERROR: doProcMetric:metric type");
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
        event_t evt = INT_EVENT("fs.stat", ctrs->numStat.evt, DELTA, fields);
        cmdSendEvent(g_ctl, &evt, getTime(), &g_proc);
    }

    // Only report if enabled
    if ((g_summary.fs.stat) || !cfgMtcFsEnable(g_cfg.staticfg)) {
        return;
    }

    // Do not report zeros
    if (ctrs->numStat.mtc == 0) return;

    event_t evt = INT_EVENT("fs.stat", ctrs->numStat.mtc, DELTA, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLogError("doStatMetric");
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

static int
decimalToOctal(int dec)
{
    int oct = 0, i = 1;

    while (dec != 0) {
        oct += (dec % 8) * i;
        dec /= 8;
        i *= 10;
    }

    return oct;
}

// The assumption being we will add more protocols
static void
getNetPtotocol(net_info *net, event_field_t *nevent, int *ix)
{
    if (!net || !nevent) return;
    in_port_t localPort, remotePort;

    localPort = get_port_net(net, net->localConn.ss_family, LOCAL);
    remotePort = get_port_net(net, net->remoteConn.ss_family, REMOTE);

    if ((localPort == 80) || (localPort == 443) ||
        (remotePort == 80) || (remotePort == 443)) {
        H_ATTRIB(nevent[*ix], "net_protocol", "http", 1);
        NEXT_FLD(*ix, NET_MAX_FIELDS);
    }

    return;
}

/*
{
  "sourcetype": "net",
  "source": "net.open",
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
    "net.host.name": "scope-vm", (removed as redundant with host)
    "net.protocol": "http",
  },
  "_time": timestamp
}
*/
static void
doNetOpenEvent(net_info *net)
{
    int nix = 0;
    const char *metric = "net.open";
    char rport[8];
    char lport[8];
    char raddr[INET6_ADDRSTRLEN];
    char laddr[INET6_ADDRSTRLEN];
    event_field_t nevent[NET_MAX_FIELDS];

    if (net->type != SOCK_STREAM) return;

    getNetInternals(net, net->type,
                    &net->localConn, &net->remoteConn,
                    laddr, raddr, sizeof(raddr),
                    lport, rport, sizeof(rport),
                    nevent, &nix, NET_MAX_FIELDS);

    getNetPtotocol(net, nevent, &nix);

    nevent[nix].name = NULL;
    nevent[nix].value_type = FMT_END;
    nevent[nix].value.str = NULL;
    nevent[nix].cardinality = 0;

    event_t evt = INT_EVENT(metric, g_ctrs.openPorts.evt, CURRENT, nevent);
    evt.src = CFG_SRC_NET;
    cmdSendEvent(g_ctl, &evt, net->uid, &g_proc);
}

/*
{
  "sourcetype": "net",
  "source": "net.close",
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
    "net.host.name": "scope-vm", (removed as redundant with host)
    "net.protocol": "http",
    "duration": 243,
    "net.close.reason": "normal",
    "net.close.origin": "peer",
    "net.bytes_sent": 4134,
    "net.bytes_recv": 123
  },
  "_time": timestamp
}
 */
static void
doNetCloseEvent(net_info *net, uint64_t dur)
{
    int nix = 0;
    const char *metric = "net.close";
    char rport[8];
    char lport[8];
    char raddr[INET6_ADDRSTRLEN];
    char laddr[INET6_ADDRSTRLEN];
    event_field_t nevent[NET_MAX_FIELDS];


    if (net->type != SOCK_STREAM) return;

    getNetInternals(net, net->type,
                    &net->localConn, &net->remoteConn,
                    laddr, raddr, sizeof(raddr),
                    lport, rport, sizeof(rport),
                    nevent, &nix, NET_MAX_FIELDS);

    if ((net->protoDetect == DETECT_TRUE) &&
        net->protoProtoDef &&
        !scope_strcasecmp(net->protoProtoDef->protname, "HTTP")) {
        H_ATTRIB(nevent[nix], "net_protocol", "http", 1);
        NEXT_FLD(nix, NET_MAX_FIELDS);
    }

    getNetPtotocol(net, nevent, &nix);

    H_VALUE(nevent[nix], "duration", dur, 1);
    NEXT_FLD(nix, NET_MAX_FIELDS);

    H_VALUE(nevent[nix], "net_bytes_sent", net->txBytes.evt, 1);
    NEXT_FLD(nix, NET_MAX_FIELDS);

    H_VALUE(nevent[nix], "net_bytes_recv", net->rxBytes.evt, 1);
    NEXT_FLD(nix, NET_MAX_FIELDS);

    if (net->remoteClose == TRUE) {
        H_ATTRIB(nevent[nix], "net_close_reason", "remote", 1);
        NEXT_FLD(nix, NET_MAX_FIELDS);
    } else {
        H_ATTRIB(nevent[nix], "net_close_reason", "local", 1);
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
   "file.errors": 1, (removed until we support errs per fd)
   "proc.uid": 1000,
   "proc.gid": 1000,
   "proc.cgroup": "foo",
   "duration": 1833
   }
   }
   Note: if we ever want mode values in rwx string format as opposed to octal
   This will function will return the rwx string: char *mode = osGetFileMode(fs->mode);
*/
static void
doFSOpenEvent(fs_info *fs, const char *op)
{
    const char *metric = "fs.open";

    if ((fs->fd > 2) && scope_strncmp(fs->path, "std", 3)) {

        event_field_t fevent[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            FILE_EV_NAME(fs->path),
            PROC_UID(g_proc.uid),
            PROC_GID(g_proc.gid),
            PROC_CGROUP(g_proc.cgroup),
            FILE_EV_MODE(decimalToOctal(fs->mode & (S_IRWXU | S_IRWXG | S_IRWXO))),
            FILE_OWNER(fs->fuid),
            FILE_GROUP(fs->fgid),
            OP_FIELD(op),
            FIELDEND
        };

        event_t evt = INT_EVENT(metric, fs->numOpen.evt, DELTA, fevent);
        evt.src = (ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS)) ? CFG_SRC_FS : CFG_SRC_METRIC;
        cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
    }
}

static void
doFSCloseEvent(fs_info *fs, const char *op)
{
    const char *metric = "fs.close";

    if ((fs->fd > 2) && scope_strncmp(fs->path, "std", 3)) {

        event_field_t fevent[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            FILE_EV_NAME(fs->path),
            PROC_UID(g_proc.uid),
            PROC_GID(g_proc.gid),
            PROC_CGROUP(g_proc.cgroup),
            FILE_EV_MODE(decimalToOctal(fs->mode & (S_IRWXU | S_IRWXG | S_IRWXO))),
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
        evt.src = (ctlEvtSourceEnabled(g_ctl, CFG_SRC_FS)) ? CFG_SRC_FS : CFG_SRC_METRIC;
        cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
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
            // with small value of totalDuration.evt we miss the fs.duration event
            // TODO handle the precision with floating point in future
            if (dur == 0ULL) {
                dur = DEFAULT_MIN_DURATION_TIME;
            }
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

        // Only report if metrics enabled
        if (((g_summary.fs.read_write) && (source == EVENT_BASED)) ||
           (!cfgMtcFsEnable(g_cfg.staticfg))) {
            return;
        }

        dur = 0ULL;
        cachedDurationNum = fs->numDuration.mtc; // avoid div by zero
        if (cachedDurationNum >= 1) {
            // factor of 1000 converts ns to us.
            dur = fs->totalDuration.mtc / ( 1000 * cachedDurationNum);
        }

        // Don't report zeros
        if (dur == 0ULL) return;

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
            scopeLogError("fd:%d ERROR: doFSMetric:FS_DURATION:cmdSendMetric", fs->fd);
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

            event_t rwMetric = INT_EVENT(metric, sizebytes->evt, DELTA, fields);
            cmdSendEvent(g_ctl, &rwMetric, fs->uid, &g_proc);
            //atomicSwapU64(&numops->evt, 0);
            //atomicSwapU64(&sizebytes->evt, 0);
            ////atomicSwapU64(global_counter->evt, 0);
        }

        // Only report if metrics enabled
        if (((g_summary.fs.read_write) && (source == EVENT_BASED)) ||
           (!cfgMtcFsEnable(g_cfg.staticfg))) {
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

        event_t rwMetric = INT_EVENT(metric, sizebytes->mtc, DELTA, fields);

        if (cmdSendMetric(g_mtc, &rwMetric)) {
            scopeLogError("fd:%d %s", fs->fd, err_str);
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
                metric = "fs.open";
                numops = &fs->numOpen;
                global_counter = &g_ctrs.numOpen;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_OPEN:cmdSendMetric";
                break;
            case FS_CLOSE:
                metric = "fs.close";
                numops = &fs->numClose;
                global_counter = &g_ctrs.numClose;
                summarize = &g_summary.fs.open_close;
                err_str = "ERROR: doFSMetric:FS_CLOSE:cmdSendMetric";
                break;
            case FS_SEEK:
                metric = "fs.seek";
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
        if ((type == FS_SEEK) && (numops->evt != 0ULL)) {
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
        }

        if (reported == TRUE) atomicSwapU64(&numops->evt, 0);


        // Only report if metrics enabled
        if ((*summarize && (source == EVENT_BASED)) ||
           (!cfgMtcFsEnable(g_cfg.staticfg))) {
            return;
        }

        // Don't report zeros.
        if (numops->mtc == 0ULL) return;

        event_t evt = INT_EVENT(metric, numops->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLogError("fd:%d %s", fs->fd, err_str);
        }
        subFromInterfaceCounts(global_counter, numops->mtc);
        atomicSwapU64(&numops->mtc, 0);
        break;
    }

    case FS_DELETE:
    {
        event_field_t fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            HOST_FIELD(g_proc.hostname),
            OP_FIELD(op),
            FILE_FIELD(pathname),
            UNIT_FIELD("operation"),
            FIELDEND
        };

        event_t evt = INT_EVENT("fs.delete", 1, DELTA, fields);
        evt.src = CFG_SRC_FS;
        cmdSendEvent(g_ctl, &evt, fs->uid, &g_proc);
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
                SUMMARY_FIELD("true"),
                FIELDEND
            };
            event_t evt = INT_EVENT(metric, (*value)[bucket].mtc, DELTA, fields);
            if (cmdSendMetric(g_mtc, &evt)) {
                scopeLogError("%s", err_str);
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
            if (!g_summary.fs.read_write || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.read";
            value = &g_ctrs.readBytes;
            err_str = "ERROR: doTotal:TOT_READ:cmdSendMetric";
            break;
        case TOT_WRITE:
            if (!g_summary.fs.read_write || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.write";
            value = &g_ctrs.writeBytes;
            err_str = "ERROR: doTotal:TOT_WRITE:cmdSendMetric";
            break;
        case TOT_RX:
        case TOT_TX:
            if (!g_summary.net.rx_tx || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            doTotalNetRxTx(type);
            return;   // <--  We're doing the work above; nothing to see here.
        case TOT_SEEK:
            if (!g_summary.fs.seek || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.seek";
            value = &g_ctrs.numSeek;
            err_str = "ERROR: doTotal:TOT_SEEK:cmdSendMetric";
            units = "operation";
            break;
        case TOT_STAT:
            if (!g_summary.fs.stat || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.stat";
            value = &g_ctrs.numStat;
            err_str = "ERROR: doTotal:TOT_STAT:cmdSendMetric";
            units = "operation";
            break;
        case TOT_OPEN:
            if (!g_summary.fs.open_close || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.open";
            value = &g_ctrs.numOpen;
            err_str = "ERROR: doTotal:TOT_OPEN:cmdSendMetric";
            units = "operation";
            break;
        case TOT_CLOSE:
            if (!g_summary.fs.open_close || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.close";
            value = &g_ctrs.numClose;
            err_str = "ERROR: doTotal:TOT_CLOSE:cmdSendMetric";
            units = "operation";
            break;
        case TOT_DNS:
            if (!g_summary.net.dns || (!cfgMtcDnsEnable(g_cfg.staticfg)))
                return;
            metric = "dns.req";
            value = &g_ctrs.numDNS;
            err_str = "ERROR: doTotal:TOT_DNS:cmdSendMetric";
            units = "request";
            break;
        case TOT_PORTS:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.port";
            value = &g_ctrs.openPorts;
            err_str = "ERROR: doTotal:TOT_PORTS:cmdSendMetric";
            units = "instance";
            aggregation_type = CURRENT;
            break;
        case TOT_TCP_CONN:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.tcp";
            value = &g_ctrs.netConnectionsTcp;
            err_str = "ERROR: doTotal:TOT_TCP_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_UDP_CONN:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.udp";
            value = &g_ctrs.netConnectionsUdp;
            err_str = "ERROR: doTotal:TOT_UDP_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_OTHER_CONN:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.other";
            value = &g_ctrs.netConnectionsOther;
            err_str = "ERROR: doTotal:TOT_OTHER_CONN:cmdSendMetric";
            units = "connection";
            aggregation_type = CURRENT;
            break;
        case TOT_NET_OPEN:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.open";
            value = &g_ctrs.netConnOpen;
            err_str = "ERROR: doTotal:TOT_NET_OPEN:cmdSendMetric";
            units = "connection";
            break;
        case TOT_NET_CLOSE:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.close";
            value = &g_ctrs.netConnClose;
            err_str = "ERROR: doTotal:TOT_NET_CLOSE:cmdSendMetric";
            units = "connection";
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
        SUMMARY_FIELD("true"),
        FIELDEND
    };
    event_t evt = INT_EVENT(metric, value->mtc, aggregation_type, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLogError("%s", err_str);
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
            if (!g_summary.fs.open_close || (!cfgMtcFsEnable(g_cfg.staticfg)))
                return;
            metric = "fs.duration";
            value = &g_ctrs.fsDurationTotal;
            num = &g_ctrs.fsDurationNum;
            aggregation_type = HISTOGRAM;
            units = "microsecond";
            factor = 1000;
            err_str = "ERROR: doTotalDuration:TOT_FS_DURATION:cmdSendMetric";
            break;
        case TOT_NET_DURATION:
            if (!g_summary.net.open_close || (!cfgMtcNetEnable(g_cfg.staticfg)))
                return;
            metric = "net.duration";
            value = &g_ctrs.connDurationTotal;
            num = &g_ctrs.connDurationNum;
            aggregation_type = DELTA_MS;
            units = "millisecond";
            factor = 1000000;
            err_str = "ERROR: doTotalDuration:TOT_NET_DURATION:cmdSendMetric";
            break;
        case TOT_DNS_DURATION:
            if (!g_summary.net.dns || (!cfgMtcDnsEnable(g_cfg.staticfg)))
                return;
            metric = "dns.duration";
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
        SUMMARY_FIELD("true"),
        FIELDEND
    };
    event_t evt = INT_EVENT(metric, dur, aggregation_type, fields);
    if (cmdSendMetric(g_mtc, &evt)) {
        scopeLogError("%s", err_str);
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
            value = &net->counters.openPorts;
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

        // Only report metric if enabled
        if (((g_summary.net.open_close) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
            return;
        }

        event_t evt = INT_EVENT(metric, value->mtc, CURRENT, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLogError("fd:%d %s", net->fd, err_str);
        }
        // Don't reset the info if we tried to report.  It's a gauge.
        // atomicSwapU64(value, 0);

        break;
    }

    case CONNECTION_OPEN:
    {
        // Report the net open event
        doNetOpenEvent(net);

        // Only report metric if enabled
        if (((g_summary.net.open_close) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
            return;
        }

        const char* metric = "net.open";
        const char* units = "connection";
        const char* err_str = "ERROR: doNetMetric:CONNECTION_OPEN:cmdSendMetric";
        counters_element_t* value = &net->counters.netConnOpen;
        counters_element_t* global_counter = &g_ctrs.netConnOpen;
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

        // Report the net open metric
        event_t evt = INT_EVENT(metric, value->mtc, DELTA, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLogError("fd:%d %s", net->fd, err_str);
        }
        subFromInterfaceCounts(global_counter, value->mtc);
        atomicSwapU64(&value->mtc, 0);

        break;
    }

    case CONNECTION_CLOSE:
    {
        // Only report metric if enabled
        if (((g_summary.net.open_close) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
            return;
        }

        // Report the net close metric
        const char* metric = "net.close";
        const char* units = "connection";
        const char* err_str = "ERROR: doNetMetric:CONNECTION_CLOSE:cmdSendMetric";
        counters_element_t* value = &net->counters.netConnClose;
        counters_element_t* global_counter = &g_ctrs.netConnClose;
        event_field_t mtc_fields[] = {
            PROC_FIELD(g_proc.procname),
            PID_FIELD(g_proc.pid),
            FD_FIELD(net->fd),
            HOST_FIELD(g_proc.hostname),
            PROTO_FIELD(proto),
            PORT_FIELD(localPort),
            UNIT_FIELD(units),
            FIELDEND
        };
        event_t mtc_evt = INT_EVENT(metric, value->mtc, DELTA, mtc_fields);
        if (cmdSendMetric(g_mtc, &mtc_evt)) {
            scopeLogError("fd:%d %s", net->fd, err_str);
        }
        subFromInterfaceCounts(global_counter, value->mtc);
        atomicSwapU64(&value->mtc, 0);

        break;
    }

    case CONNECTION_DURATION:
    {
        // Cleanup the per-channel state data for HTTP processing when the
        // channel is closed. We're not checking the results here because it's
        // possible, even likely, these operations will fail to find an entry
        // for the channel. Instead of searching first and then deleting, just
        // delete and let it fail if it's not there.
        lstDelete(g_maplist, net->uid);        // HTTP/1 saved state
        lstDelete(g_http2_channels, net->uid); // HTTP/2 saved state

        // Most NET events get blocked on the data side when the watch/source
        // is disabled but logic added in postNetState() will send this one
        // when it occurs on an HTTP channel even when the events are disabled
        // so we can cleanup on the reporting side. So, we still need to check
        // the config here.
        if (!ctlEvtSourceEnabled(g_ctl, CFG_SRC_NET)) {
            return;
        }

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
            event_t evt = INT_EVENT("net.duration", dur, DELTA_MS, fields);
            cmdSendEvent(g_ctl, &evt, net->uid, &g_proc);
            atomicSwapU64(&net->numDuration.evt, 0);
            atomicSwapU64(&net->totalDuration.evt, 0);
        }

        // Report the net close event
        doNetCloseEvent(net, dur);

        // Only report metric if enabled
        if (((g_summary.net.open_close) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
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

        // Report the net.duration metric
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
        event_t evt = INT_EVENT("net.duration", dur, DELTA_MS, fields);
        if (cmdSendMetric(g_mtc, &evt)) {
            scopeLogError("fd:%d ERROR: doNetMetric:CONNECTION_DURATION:cmdSendMetric", net->fd);
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
            scope_strncpy(data, "ssl", sizeof(data));
        } else {
            scope_strncpy(data, "clear", sizeof(data));
        }

        // Do we need to define domain=LOCAL or NETLINK?
        if (addrIsUnixDomain(&net->remoteConn) ||
            addrIsUnixDomain(&net->localConn)) {
            localPort = net->lnode;
            remotePort = net->rnode;

            if (net->localConn.ss_family == AF_NETLINK) {
                scope_strncpy(proto, "NETLINK", sizeof(proto));
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
            scope_memmove(&rxFields, &fields, sizeof(fields));
            event_t rxUnixMetric = INT_EVENT("net.rx", net->rxBytes.evt, DELTA, rxFields);
            scope_memmove(&rxMetric, &rxUnixMetric, sizeof(event_t));
        } else {
            if (net->localConn.ss_family == AF_INET) {
                if (scope_inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    scope_strncpy(lip, " ", sizeof(lip));
                }
            } else if (net->localConn.ss_family == AF_INET6) {
                if (scope_inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    scope_strncpy(lip, " ", sizeof(lip));
                }

            } else {
                scope_strncpy(lip, " ", sizeof(lip));
            }

            if (net->remoteConn.ss_family == AF_INET) {
                if (scope_inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    scope_strncpy(rip, " ", sizeof(rip));
                }
            } else if (net->remoteConn.ss_family == AF_INET6) {
                if (scope_inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    scope_strncpy(rip, " ", sizeof(rip));
                }
            } else {
                scope_strncpy(rip, " ", sizeof(rip));
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
            scope_memmove(&rxFields, &fields, sizeof(fields));
            event_t rxNetMetric = INT_EVENT("net.rx", net->rxBytes.evt, DELTA, rxFields);
            scope_memmove(&rxMetric, &rxNetMetric, sizeof(event_t));
        }

        // Don't report zeros.
        if (net->rxBytes.evt != 0ULL) {

             cmdSendEvent(g_ctl, &rxMetric, net->uid, &g_proc);
             atomicSwapU64(&net->numRX.evt, 0);
             atomicSwapU64(&net->rxBytes.evt, 0);
        }

        if (((g_summary.net.rx_tx) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
            return;
        }

        // Don't report zeros.
        if (net->rxBytes.mtc == 0ULL) return;

        event_t rxNetMetric = INT_EVENT("net.rx", net->rxBytes.mtc, DELTA, rxFields);
        scope_memmove(&rxMetric, &rxNetMetric, sizeof(event_t));
        if (cmdSendMetric(g_mtc, &rxMetric)) {
            scopeLogError("ERROR: doNetMetric:NETRX:cmdSendMetric");
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
            scope_strncpy(data, "ssl", sizeof(data));
        } else {
            scope_strncpy(data, "clear", sizeof(data));
        }

        if (addrIsUnixDomain(&net->remoteConn) ||
            addrIsUnixDomain(&net->localConn)) {
            localPort = net->lnode;
            remotePort = net->rnode;

            if (net->localConn.ss_family == AF_NETLINK) {
                scope_strncpy(proto, "NETLINK", sizeof(proto));
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
            scope_memmove(&txFields, &fields, sizeof(fields));
            event_t txUnixMetric = INT_EVENT("net.tx", net->txBytes.evt, DELTA, txFields);
            scope_memmove(&txMetric, &txUnixMetric, sizeof(event_t));
        } else {
            if (net->localConn.ss_family == AF_INET) {
                if (scope_inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->localConn)->sin_addr,
                              lip, sizeof(lip)) == NULL) {
                    scope_strncpy(lip, " ", sizeof(lip));
                }
            } else if (net->localConn.ss_family == AF_INET6) {
                if (scope_inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->localConn)->sin6_addr,
                              lip, sizeof(lip)) == NULL) {
                    scope_strncpy(lip, " ", sizeof(lip));
                }

            } else {
                scope_strncpy(lip, " ", sizeof(lip));
            }

            if (net->remoteConn.ss_family == AF_INET) {
                if (scope_inet_ntop(AF_INET,
                              &((struct sockaddr_in *)&net->remoteConn)->sin_addr,
                              rip, sizeof(rip)) == NULL) {
                    scope_strncpy(rip, " ", sizeof(rip));
                }
            } else if (net->remoteConn.ss_family == AF_INET6) {
                if (scope_inet_ntop(AF_INET6,
                              &((struct sockaddr_in6 *)&net->remoteConn)->sin6_addr,
                              rip, sizeof(rip)) == NULL) {
                    scope_strncpy(rip, " ", sizeof(rip));
                }
            } else {
                scope_strncpy(rip, " ", sizeof(rip));
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
            scope_memmove(&txFields, &fields, sizeof(fields));
            event_t txNetMetric = INT_EVENT("net.tx", net->txBytes.evt, DELTA, txFields);
            scope_memmove(&txMetric, &txNetMetric, sizeof(event_t));
        }

        // Don't report zeros.
        if (net->txBytes.evt != 0ULL) {

            cmdSendEvent(g_ctl, &txMetric, net->uid, &g_proc);
            //atomicSwapU64(&net->numTX.evt, 0);
            //atomicSwapU64(&net->txBytes.evt, 0);
        }

        // Don't report zeros.
        if (net->txBytes.mtc == 0ULL) return;

        if (((g_summary.net.rx_tx) && (source == EVENT_BASED)) ||
           (!cfgMtcNetEnable(g_cfg.staticfg))) {
            return;
        }

        event_t txNetMetric = INT_EVENT("net.tx", net->txBytes.mtc, DELTA, txFields);
        scope_memmove(&txMetric, &txNetMetric, sizeof(event_t));
        if (cmdSendMetric(g_mtc, &txMetric)) {
            scopeLogError("ERROR: doNetMetric:NETTX:cmdSendMetric");
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

        doDNSMetricName(DNS, net);

        break;
    }

    default:
        scopeLogError("ERROR: doNetMetric:metric type");
    }
}

static data_type_t
typeFromStr(const unsigned char *string)
{
    const char *str = (const char *)string;  // casting away unsigned
    if (!scope_strcmp(str, "c")) return DELTA;     // counter
    if (!scope_strcmp(str, "g")) return CURRENT;   // gauge
    if (!scope_strcmp(str, "ms")) return DELTA_MS; // timer
    if (!scope_strcmp(str, "s")) return SET;
    if (!scope_strcmp(str, "h")) return HISTOGRAM;

    return CURRENT;  // Default to CURRENT aka "gauge"
}

static event_field_t *
createFieldsForCapturedMetrics(const unsigned char *alldims)
{
    event_field_t *fields = scope_calloc(HTTP_MAX_FIELDS, sizeof(event_field_t));
    if (!fields) return NULL;
    int ix = 0;

    // Metric fields from what we intercepted
    //  eg:   #ceo:clint,date_founded:2017
    //        fieldname1->ceo           fieldval1->clint
    //        fieldname2->date_founded  fieldval2->2017
    char *dims = (char *)alldims;
    while (dims) {

        // dims starts with # and commas are overwritten to null below
        if ((dims[0] != '#') && (dims[0] != '\0')) break;

        // get pointers to the name and value pairs and
        // make sure each is null delimited
        char *fieldname = ++dims; // advance past the # or , char
        char *fieldval = scope_strchr(fieldname, ':');
        if (!fieldval) break;
        *fieldval = '\0';       // overwrite the : char to delimit fieldname
        ++fieldval;             // advance past the null

        // Set dims for next time through this loop
        dims = scope_strchr(fieldval, ',');
        if (dims) *dims = '\0'; // overwrite the , char to delimit fieldval

        // Add this name value pair
        H_ATTRIB(fields[ix], fieldname, fieldval, 0);
        HTTP_NEXT_FLD(ix);
    }

    httpFieldEnd(fields, ix);

    return fields;
}

void
reportCapturedMetric(const captured_metric_t *metric)
{
    if (!metric) return;

    const char *value = (const char *)metric->value; // casting away unsigned
    const char *name = (const char *)metric->name; // casting away unsigned

    event_field_t builtInFields[] = {
        PROC_FIELD(g_proc.procname),
        PID_FIELD(g_proc.pid),
        HOST_FIELD(g_proc.hostname),
        FIELDEND
    };
    event_field_t *capturedFields = createFieldsForCapturedMetrics(metric->dims);

    // Look for a decimal point.  Comma allows for localization, even
    // though the regex that creates value currently using does not.
    event_t out_mtc;
    char *endptr = NULL;
    if (scope_strpbrk(value, ".,")) {
        // Value looks like a floating point value...
        scope_errno = 0;
        double doubleval = scope_strtod(value, &endptr);
        if ((endptr == value) || (scope_errno != 0)) {
            if (scope_errno == ERANGE) {
                char *underover = (doubleval == 0.0) ? "underflow" : "overflow";
                DBG("Couldn't be converted to float: %s (%s)", value, underover);
            } else {
                DBG("Couldn't be converted to float: %s", value);
            }
            goto out;
        }
        event_t flt_met = FLT_EVENT(name, doubleval, typeFromStr(metric->type), builtInFields);
        scope_memmove(&out_mtc, &flt_met, sizeof(event_t));
    } else {
        // Value looks like an integer value...
        scope_errno = 0;
        long long int intval = scope_strtoll(value, &endptr, 10);
        if ((endptr == value) || (scope_errno != 0)) {
            DBG("Couldn't be converted to long long: %s", value);
            goto out;
        }
        event_t int_met = INT_EVENT(name, intval, typeFromStr(metric->type), builtInFields);
        scope_memmove(&out_mtc, &int_met, sizeof(event_t));
    }

    out_mtc.capturedFields = capturedFields;
    if (cmdSendMetric(g_mtc, &out_mtc)) {
        scopeLog(CFG_LOG_ERROR, "ERROR: reportCapturedMetric:cmdSendMetric");
    }

out:
    if (capturedFields) scope_free(capturedFields);
}

bool
doConnection(void)
{
    bool ready = FALSE;

    // if no connection, don't pull data from the queue
    if (ctlNeedsConnection(g_ctl, CFG_CTL)) {
        if (ctlConnect(g_ctl, CFG_CTL)) {
            reportProcessStart(g_ctl, FALSE, CFG_CTL);
            ready = TRUE;
        }
    } else {
        ready = TRUE;
    }

    if ((cfgLogStreamEnable(g_cfg.staticfg) == FALSE) && (ready == FALSE)) {
        if (mtcNeedsConnection(g_mtc)) {
            if (mtcConnect(g_mtc)) {
                ready = TRUE;
            }
        } else {
            ready = TRUE;
        }
    }

    return ready;
}

void
doHttpAgg()
{
    if (cfgMtcHttpEnable(g_cfg.staticfg)) {
        httpAggSendReport(g_http_agg, g_mtc);
    }
    httpAggReset(g_http_agg);
}

void
doEvent()
{
    uint64_t data;

    if (doConnection() == FALSE) return;

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
                doDNSMetricName(net->data_type, net);
            } else if (event->evtype == EVT_PROTO) {
                proto = (protocol_info *)data;
                doProtocolMetric(proto);
            } else {
                DBG(NULL);
                return;
            }

            scope_free(event);
        }
    }
    reportAllCapturedMetrics();
    ctlFlushLog(g_ctl);
    ctlFlush(g_ctl);
}

void
doPayload()
{
    uint64_t data;

    // if LS enabled, then check for a connection
    if (cfgLogStreamEnable(g_cfg.staticfg) && ctlNeedsConnection(g_ctl, CFG_LS)) {
        if (ctlConnect(g_ctl, CFG_LS)) {
            reportProcessStart(g_ctl, FALSE, CFG_LS);
        } else {
            return;
        }
    }

    while ((data = msgPayloadGet(g_ctl)) != -1) {
        if (data) {
            payload_info *pinfo = (payload_info *)data;
            net_info *net = &pinfo->net;
            size_t hlen = 1024;
            char pay[hlen];
            char *srcstr = NULL,
                netrx[]="netrx", nettx[]="nettx", none[]="none",
                tlsrx[]="tlsrx", tlstx[]="tlstx";

            switch (pinfo->src) {
            case NETTX:
                srcstr = nettx;
                break;

            case TLSTX:
                srcstr = tlstx;
                 break;

            case NETRX:
                srcstr = netrx;
                break;

            case TLSRX:
                srcstr = tlsrx;
                break;

            default:
                srcstr = none;
                break;
            }

            char lport[20], rport[20];
            char lip[INET6_ADDRSTRLEN];
            char rip[INET6_ADDRSTRLEN];

            if (net && net->active) {
                if (getConn(&net->localConn, lip, sizeof(lip), lport, sizeof(lport)) == FALSE) {
                    if (net->localConn.ss_family == AF_UNIX) {
                        scope_strncpy(lip, "af_unix", sizeof(lip));
                        scope_snprintf(lport, sizeof(lport), "%ld", net->lnode);
                    } else {
                        scope_strncpy(lip, srcstr, sizeof(lip));
                        scope_strncpy(lport, "0", sizeof(lport));
                    }
                }

                if (getConn(&net->remoteConn, rip, sizeof(rip), rport, sizeof(rport)) == FALSE) {
                    if (net->remoteConn.ss_family == AF_UNIX) {
                        scope_strncpy(rip, "af_unix", sizeof(rip));
                        scope_snprintf(rport, sizeof(rport), "%ld", net->rnode);
                    } else {
                        scope_strncpy(rip, srcstr, sizeof(rip));
                        scope_strncpy(rport, "0", sizeof(rport));
                    }
                }
            } else {
                scope_strncpy(lip, srcstr, sizeof(lip));
                scope_strncpy(lport, "0", sizeof(lport));
                scope_strncpy(rip, srcstr, sizeof(rip));
                scope_strncpy(rport, "0", sizeof(rport));
            }

            uint64_t netid = (net != NULL) ? net->uid : 0;
            char * protoName = pinfo->net.protoProtoDef
                ? pinfo->net.protoProtoDef->protname
                : (pinfo->net.tlsProtoDef
                   ? pinfo->net.tlsProtoDef->protname 
                   : "");
            struct timeval tv;
            scope_gettimeofday(&tv, NULL);
            double timestamp = tv.tv_sec + tv.tv_usec/1e6;
            int rc = scope_snprintf(pay, hlen,
                              "{\"type\":\"payload\",\"id\":\"%s\",\"pid\":%d,\"ppid\":%d,\"fd\":%d,\"src\":\"%s\",\"_channel\":%ld,\"len\":%ld,\"localip\":\"%s\",\"localp\":%s,\"remoteip\":\"%s\",\"remotep\":%s,\"protocol\":\"%s\",\"_time\":%.3f}",
                              g_proc.id, g_proc.pid, g_proc.ppid, pinfo->sockfd, srcstr, netid, pinfo->len, lip, lport, rip, rport, protoName, timestamp);
            if (rc < 0) {
                // unlikely
                if (pinfo->data) scope_free(pinfo->data);
                if (pinfo) scope_free(pinfo);
                DBG(NULL);
                return;
            }

            if (rc < hlen) {
                hlen = rc + 1;
            } else {
                hlen--;
                scopeLogWarn("fd:%d WARN: payload header was truncated", pinfo->sockfd);
            }

            char *bdata = NULL;

            if (cfgLogStreamEnable(g_cfg.staticfg)) {
                bdata = scope_calloc(1, hlen + pinfo->len);
                if (bdata) {
                    scope_memmove(bdata, pay, hlen);
                    scope_strncat(bdata, "\n", hlen);
                    scope_memmove(&bdata[hlen], pinfo->data, pinfo->len);
                    cmdSendPayload(g_ctl, bdata, hlen + pinfo->len);
                }
            } else if (ctlPayDir(g_ctl)) {
                int fd;
                char path[PATH_MAX];

                ///tmp/<splunk-pid>/<src_host:src_port:dst_port>.in
                switch (pinfo->src) {
                case NETTX:
                case TLSTX:
                    scope_snprintf(path, PATH_MAX, "%s/%d_%s:%s_%s:%s.out",
                             ctlPayDir(g_ctl), g_proc.pid, rip, rport, lip, lport);
                    break;

                case NETRX:
                case TLSRX:
                    scope_snprintf(path, PATH_MAX, "%s/%d_%s:%s_%s:%s.in",
                             ctlPayDir(g_ctl), g_proc.pid, rip, rport, lip, lport);
                    break;

                default:
                    scope_snprintf(path, PATH_MAX, "%s/%d.na",
                             ctlPayDir(g_ctl), g_proc.pid);
                    break;
                }

                if ((fd = scope_open(path, O_WRONLY | O_CREAT | O_APPEND, 0666)) != -1) {
                    if (checkEnv("SCOPE_PAYLOAD_HEADER", "true")) {
                         scope_write(fd, pay, rc);
                    }

                    size_t to_write = pinfo->len;
                    size_t written = 0;
                    int rc;

                    while (to_write > 0) {
                        rc = scope_write(fd, &pinfo->data[written], to_write);
                        if (rc <= 0) {
                            DBG(NULL);
                            break;
                        }

                        written += rc;
                        to_write -= rc;
                    }

                    scope_close(fd);
                }
            }

            if (bdata) scope_free(bdata);
            if (pinfo->data) scope_free(pinfo->data);
            if (pinfo) scope_free(pinfo);
        }
    }
}

void
doProcStartMetric(void)
{
    if (!cfgMtcProcEnable(g_cfg.staticfg)) {
        return;
    }

    char *urlEncodedCmd = NULL;
    char *command = g_proc.cmd; // default is no encoding

    // If we're reporting in statsd format, url encode the cmd
    // to avoid characters that could cause parsing problems
    // for statsd aggregators  :|@#,
    if (cfgMtcFormat(g_cfg.staticfg) == CFG_FMT_STATSD) {
        urlEncodedCmd = fmtUrlEncode(g_proc.cmd);
        command = urlEncodedCmd;
    }

    event_field_t fields[] = {
        STRFIELD("proc", (g_proc.procname), 4, TRUE),
        NUMFIELD("pid", (g_proc.pid), 4, TRUE),
        NUMFIELD("gid", (g_proc.gid), 4, TRUE),
        STRFIELD("groupname", (g_proc.groupname), 4, TRUE),
        NUMFIELD("uid", (g_proc.uid), 4, TRUE),
        STRFIELD("username", (g_proc.username), 4, TRUE),
        STRFIELD("host", (g_proc.hostname), 4, TRUE),
        STRFIELD("args", (command), 7, TRUE),
        STRFIELD("unit", ("process"), 1, TRUE),
        FIELDEND
    };
    event_t evt = INT_EVENT("proc.start", 1, DELTA, fields);
    cmdSendMetric(g_mtc, &evt);
    if (urlEncodedCmd) scope_free(urlEncodedCmd);
}
