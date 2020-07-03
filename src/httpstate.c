#include <errno.h>
#include <string.h>
#include "com.h"
#include "dbg.h"
#include "httpstate.h"
#include "plattime.h"
#include "search.h"


#define MIN_HDR_ALLOC (4  * 1024)
#define MAX_HDR_ALLOC (16 * 1024)

#define HTTP_START "HTTP/"
#define HTTP_END "\r\n"
#define CONTENT_LENGTH "Content-Length:"
static needle_t* g_http_start = NULL;
static needle_t* g_http_end = NULL;
static needle_t* g_http_clen = NULL;

static void setHttpState(http_state_t *httpstate, http_enum_t toState);
static void appendHeader(http_state_t *httpstate, char* buf, size_t len);
static size_t getContentLength(char *header, size_t len);
static size_t bytesToSkipForContentLength(http_state_t *httpstate, size_t len);
static bool setHttpId(httpId_t *httpId, net_info *net, int sockfd, uint64_t id, metric_t src);
static int reportHttp(http_state_t *httpstate);
static bool scanForHttpHeader(http_state_t *httpstate, char *buf, size_t len, httpId_t *httpId);


static void
setHttpState(http_state_t *httpstate, http_enum_t toState)
{
    if (!httpstate) return;
    switch (toState) {
        case HTTP_NONE:
            if (httpstate->hdr) free(httpstate->hdr);
            memset(httpstate, 0, sizeof(*httpstate));
            break;
        case HTTP_HDR:
        case HTTP_HDREND:
        case HTTP_DATA:
            break;
        default:
            DBG(NULL);
            return;
    }
    httpstate->state = toState;
}

static void
appendHeader(http_state_t *httpstate, char* buf, size_t len)
{
    if (!httpstate || !buf) return;

    // make sure we have enough allocated space
    size_t content_size = httpstate->hdrlen+len;
    size_t alloc_size = (!httpstate->hdralloc) ? MIN_HDR_ALLOC : httpstate->hdralloc;
    while (alloc_size < content_size) {
        alloc_size = alloc_size << 2; // same as multiplying by 4
        if (alloc_size > MAX_HDR_ALLOC) {
             DBG(NULL);
             // More than we're willing to allocate for one header.
             // We might have missed the end of the header???
             setHttpState(httpstate, HTTP_NONE);
             return;
        }
    }

    // If we need more space, realloc
    if (alloc_size != httpstate->hdralloc) {
        char* temp = realloc(httpstate->hdr, alloc_size);
        if (!temp) {
            DBG(NULL);
            // Don't return partial headers...  All or nothing.
            setHttpState(httpstate, HTTP_NONE);
            return;
        }
        httpstate->hdr = temp;
        httpstate->hdralloc = alloc_size;
    }

    // Append the data
    memcpy(&httpstate->hdr[httpstate->hdrlen], buf, len);
    httpstate->hdrlen += len;
}

static size_t
getContentLength(char *header, size_t len)
{
    size_t ix;
    size_t rc;
    char *val;

    // ex: Content-Length: 559\r\n
    if ((ix = needleFind(g_http_clen, header, len)) == -1) return -1;

    if ((ix <= 0) || (ix > len) || ((ix + needleLen(g_http_clen)) > len)) return -1;

    val = &header[ix + needleLen(g_http_clen)];

    errno = 0;
    rc = strtoull(val, NULL, 0);
    if ((errno != 0) || (rc == 0)) {
        return -1;
    }
    return rc;
}

static size_t
bytesToSkipForContentLength(http_state_t *httpstate, size_t len)
{
    // don't skip anything because we don't know anything about it
    // or because content length skipping is off
    if (!httpstate || (httpstate->clen == 0)) return 0;

    // skip the rest of the remaining content length
    // and turn off content length skipping
    if (len > httpstate->clen) {
        size_t rv = httpstate->clen;
        httpstate->clen = 0;
        return rv;
    }

    // skip this whole buf and maintain content length
    httpstate->clen -= len;
    return len;
}

static bool
setHttpId(httpId_t *httpId, net_info *net, int sockfd, uint64_t id, metric_t src)
{
    if (!httpId) return FALSE;

    /*
     * If we have an fd, use the uid/channel value as it's unique
     * else we are likley using TLS, so default to the session ID
     */
    in_port_t localPort, remotePort;
    if (net) {
        httpId->uid = net->uid;
        localPort = get_port_net(net, net->localConn.ss_family, LOCAL);
        remotePort = get_port_net(net, net->remoteConn.ss_family, REMOTE);
    } else if (id != -1) {
        httpId->uid = id;
        localPort = remotePort = 0;
    } else {
        DBG(NULL);
        return FALSE;
    }

    httpId->isSsl = ((src == TLSTX) || (src == TLSRX) ||
                    (localPort == 443) || (remotePort == 443));

    httpId->sockfd = sockfd;

    return TRUE;
}

// For now, only doing HTTP/1.X headers
static int
reportHttp(http_state_t *httpstate)
{
    if (!httpstate || !httpstate->hdr || !httpstate->hdrlen) return -1;

    protocol_info *proto = calloc(1, sizeof(struct protocol_info_t));
    http_post *post = calloc(1, sizeof(struct http_post_t));
    if (!proto || !post) {
        // Bummer!  We're losing info.  At least make sure we clean up.
        DBG(NULL);
        if (post) free(post);
        if (proto) free(proto);
        return -1;
    }

    // If the first 5 chars are HTTP/, it's a response header
    int isResponse =
      (needleFind(g_http_start, httpstate->hdr, needleLen(g_http_start)) != -1);

    // Set proto info
    proto->evtype = EVT_PROTO;
    proto->ptype = (isResponse) ? EVT_HRES : EVT_HREQ;
    proto->len = httpstate->hdrlen;
    proto->fd = httpstate->id.sockfd;
    proto->uid = httpstate->id.uid;
    proto->data = (char *)post;

    // Set post info
    post->ssl = httpstate->id.isSsl;
    post->start_duration = getTime();
    post->id = httpstate->id.uid;

    // "transfer ownership" of dynamically allocated header from
    // httpstate object to post object
    post->hdr = httpstate->hdr;
    httpstate->hdr = NULL;
    httpstate->hdrlen = 0;

    cmdPostEvent(g_ctl, (char *)proto);

    return 0;
}


/*
 * If we have an fd check for TCP
 * If we don't have a socket it can mean we are
 * called from certain TLS sessions; not an error
 *
 * If we are working down a content length, no
 * need to scan for a header
 *
 * Note that, at this point, we are not able to
 * use a content length optimization with gnutls
 * because it does not return a file descriptor
 * that is usable
*/
static bool
scanForHttpHeader(http_state_t *httpstate, char *buf, size_t len, httpId_t *httpId)
{
    if (!buf) return FALSE;

    // We need to handle "double interception" when ssl is involved, e.g.
    // intercepting an SSL_write(), then intercepting the write() it calls.
    // We use the isSSL flag to prevent interleaving ssl and non-ssl data.
    int headerCaptureInProgress = httpstate->state != HTTP_NONE;
    int isSslIsConsistent = httpstate->id.isSsl == httpId->isSsl;
    if (headerCaptureInProgress && !isSslIsConsistent) return FALSE;

    // Skip data if instructed to do so by previous content length
    if (httpstate->state == HTTP_DATA) {
        size_t bts = bytesToSkipForContentLength(httpstate, len);
        if (bts == len) return FALSE;
        buf = &buf[bts];
        len = len - bts;
        setHttpState(httpstate, HTTP_NONE);
    }

    // Look for start of http header
    if (httpstate->state == HTTP_NONE) {

        // find the start of http header data
        if (needleFind(g_http_start, buf, len) == -1) return FALSE;

        setHttpState(httpstate, HTTP_HDR);
        httpstate->id = *httpId;
    }

    // Look for header data
    size_t header_start = 0;
    size_t header_end = -1;
    int found_end_of_all_headers = FALSE;
    while ((httpstate->state == HTTP_HDR || httpstate->state == HTTP_HDREND) &&
           (header_start < len)) {

        header_end =
            needleFind(g_http_end, &buf[header_start], len-header_start);

        if (header_end == -1) {
            // We didn't find an end in this buffer, append the rest of the
            // buffer to what we've found before.
            setHttpState(httpstate, HTTP_HDR);
            appendHeader(httpstate, &buf[header_start], len-header_start);
            break;
        } else {
            found_end_of_all_headers =
                ((httpstate->state == HTTP_HDREND) && (header_end == 0));
            if (found_end_of_all_headers) break;

            // We found a complete header!
            setHttpState(httpstate, HTTP_HDREND);
            header_end += header_start;  // was measured from header_start
            header_end += needleLen(g_http_end);
            appendHeader(httpstate, &buf[header_start], header_end-header_start);
            header_start = header_end;
        }
    }

    // Found the end of all headers!  Time to report something!
    if (found_end_of_all_headers) {

        // append a null terminator to allow us to treat it as a string
        appendHeader(httpstate, "\0", 1);

        // check to see if there is a Content-Length in the header
        size_t clen = getContentLength(httpstate->hdr, httpstate->hdrlen);
        size_t content_in_this_buf = len - header_end;

        // post and event containing the header we found
        reportHttp(httpstate);

        // change httpstate to HTTP_DATA per Content-Length or HTTP_NONE
        if ((clen != -1) && (clen >= content_in_this_buf)) {
            httpstate->clen = clen - content_in_this_buf;
            setHttpState(httpstate, HTTP_DATA);
        } else {
            setHttpState(httpstate, HTTP_NONE);
        }
    }

    return (found_end_of_all_headers);
}

void
initHttpState(void)
{
    g_http_start = needleCreate(HTTP_START);
    g_http_end = needleCreate(HTTP_END);
    g_http_clen = needleCreate(CONTENT_LENGTH);
}

// allow all ports if they appear to have an HTTP header
bool
doHttp(uint64_t id, int sockfd, net_info *net, char *buf, size_t len, metric_t src, src_data_t dtype)
{
    if (!buf || !len) return FALSE;

    // If we know it's we're not looking at a stream, bail.
    if (net && net->type != SOCK_STREAM) return FALSE;

    httpId_t httpId = {0};
    if (!setHttpId(&httpId, net, sockfd, id, src)) return FALSE;

    int http_header_found = FALSE;

    // We won't always have net (looking at you, gnutls).  If net exists,
    // then we can use the http state it provides to look across multiple
    // buffers.  If net doesn't exist,  we can at least keep temp state
    // while within the current doHttp().
    http_state_t tempstate = {0};
    http_state_t *httpstate = (net) ? &net->http : &tempstate;


    // Handle the data in it's various format
    switch (dtype) {
        case BUF:
        {
            http_header_found = scanForHttpHeader(httpstate, buf, len, &httpId);
            break;
        }

        case MSG:
        {
            int i;
            struct msghdr *msg = (struct msghdr *)buf;
            struct iovec *iov;

            for (i = 0; i < msg->msg_iovlen; i++) {
                iov = &msg->msg_iov[i];
                if (iov && iov->iov_base) {
                    if (scanForHttpHeader(httpstate, (char*)&iov->iov_base, iov->iov_len, &httpId)) {
                        http_header_found = TRUE;
                        // stay in loop to count down content length
                    }
                }
            }
            break;
        }

        case IOV:
        {
            int i;
            // len is expected to be an iovcnt for an IOV data type
            int iovcnt = len;
            struct iovec *iov = (struct iovec *)buf;

            for (i = 0; i < iovcnt; i++) {
                if (iov[i].iov_base) {
                    if (scanForHttpHeader(httpstate, (char*)&iov[i].iov_base, iov[i].iov_len, &httpId)) {
                        http_header_found = TRUE;
                        // stay in loop to count down content length
                    }
                }
            }
            break;
        }

        default:
            DBG("%d", dtype);
            break;
    }

    // If our state is temporary, clean up after each doHttp call
    if (httpstate == &tempstate) {
        setHttpState(httpstate, HTTP_NONE);
    }

    return http_header_found;
}

void
resetHttp(http_state_t *httpstate)
{
    setHttpState(httpstate, HTTP_NONE);
}

