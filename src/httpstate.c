#include <errno.h>
#include <string.h>
#include "com.h"
#include "httpstate.h"
#include "plattime.h"
#include "search.h"

#define HTTP_START "HTTP/"
#define HTTP_END "\r\n\r\n"
#define CONTENT_LENGTH "Content-Length:"
static needle_t* g_http_start = NULL;
static needle_t* g_http_end = NULL;
static needle_t* g_http_clen = NULL;


void
initHttpState(void)
{
    g_http_start = needleCreate(HTTP_START);
    g_http_end = needleCreate(HTTP_END);
    g_http_clen = needleCreate(CONTENT_LENGTH);
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
scanFoundHttpHeader(net_info *net, char *buf, size_t len)
{
    if (!buf) return FALSE;

    // By default, turn off Content-Length skipping.
    // Only scan streams for http headers.
    if (net) {
        net->clen = 0;
        if (net->type != SOCK_STREAM) return FALSE;
    }

    size_t startLen = needleLen(g_http_start);
    size_t endLen = needleLen(g_http_end);

    // find the start of http header data
    size_t header_start =
        needleFind(g_http_start, buf, len);
    if (header_start == -1) return FALSE;
    header_start += startLen;

    // find the end of http header data
    size_t header_end =
        needleFind(g_http_end, &buf[header_start], len-header_start);
    if (header_end == -1) return FALSE;
    header_end += header_start;  // was measured from header_start
    header_end += endLen;

    // Only look for Content-Length header if we're able to save it for later
    if (net) {
        // if there is not a contentLength header, bail.
        size_t clen = getContentLength(&buf[header_start], header_end - header_start);
        if (clen == -1) return TRUE; // We found a complete header in buf

        // There is a Content-Length header!  Remember how much we can skip.
        size_t content_in_this_buf = len - header_end;
        if (clen >= content_in_this_buf) {
            net->clen = clen - content_in_this_buf;
        }
    }
    return TRUE; // We found a complete header in buf
}

static size_t
bytesToSkip(net_info *net, size_t len)
{
    // don't skip anything because we don't know anything about it
    // or because content length skipping is off
    if (!net || (net->clen == 0)) return 0;

    // skip the rest of the remaining content length
    // and turn off content length skipping
    if (len > net->clen) {
        size_t rv = net->clen;
        net->clen = 0;
        return rv;
    }

    // skip this whole buf and maintain content length
    net->clen -= len;
    return len;
}


// allow all ports if they appear to have an HTTP header
bool
isHttp(int sockfd, net_info *net, void **buf, size_t *len, metric_t src, src_data_t dtype)
{
    if (!buf || !*buf || !len) return FALSE;

    // Handle the data in it's various format
    switch (dtype) {
        case BUF:
        {
            /*
             * TODO: if the entire header is not contained in this buffer we will drop
             * the header. Need to keep state such that we can cross buffer boundaries
             * at some point.
             */
            size_t bts = bytesToSkip(net, *len);
            if (bts == *len) return FALSE;
            return scanFoundHttpHeader(net, &(((char*)*buf)[bts]), *len - bts);
        }

        case MSG:
        {
            int i;
            struct msghdr *msg = (struct msghdr *)*buf;
            struct iovec *iov;
            int http_header_found = FALSE;

            for (i = 0; i < msg->msg_iovlen; i++) {
                iov = &msg->msg_iov[i];
                if (iov && iov->iov_base) {
                    size_t bts = bytesToSkip(net, iov->iov_len);
                    if (bts == iov->iov_len) continue;
                    // TODO: we only return the first match
                    if (http_header_found) continue;
                    if (scanFoundHttpHeader(net, &(((char*)iov->iov_base)[bts]), iov->iov_len - bts)) {
                        *buf = iov->iov_base;
                        *len = iov->iov_len;
                        http_header_found = TRUE;
                    }
                }
            }
            return http_header_found;
        }

        case IOV:
        {
            int i;
            // len is expected to be an iovcnt for an IOV data type
            int iovcnt = *len;
            struct iovec *iov = (struct iovec *)*buf;
            int http_header_found = FALSE;

            for (i = 0; i < iovcnt; i++) {
                if (iov[i].iov_base) {
                    size_t bts = bytesToSkip(net, iov[i].iov_len);
                    if (bts == iov[i].iov_len) continue;
                    // TODO: we only return the first match
                    if (http_header_found) continue;
                    if (scanFoundHttpHeader(net, &(((char*)iov[i].iov_base)[bts]), iov[i].iov_len - bts)) {
                        *buf = iov[i].iov_base;
                        *len = iov[i].iov_len;
                        http_header_found = TRUE;
                    }
                }
            }
            return http_header_found;
        }

        case NONE:
        default:
            break;
    }

    return FALSE;
}


// For now, only doing HTTP/1.X headers
int
doHttp(uint64_t id, int sockfd, net_info *net, void *buf, size_t len, metric_t src)
{
    if ((buf == NULL) || (len <= 0)) return -1;

    unsigned int endix;
    in_port_t localPort, remotePort;
    size_t headsize;
    uint64_t uid;
    char *headend, *header, *hcopy;
    protocol_info *proto;
    http_post *post;
    size_t startLen = needleLen(g_http_start);

    /*
     * If we have an fd, use the uid/channel value as it's unique
     * else we are likley using TLS, so default to the session ID
     */
    if (net) {
        uid = net->uid;
        localPort = get_port_net(net, net->localConn.ss_family, LOCAL);
        remotePort = get_port_net(net, net->remoteConn.ss_family, REMOTE);
    } else if (id != -1) {
        uid = id;
        localPort = remotePort = 0;
    } else {
        return -1;
    }

    if (((endix = needleFind(g_http_end, buf, len)) != -1) &&
        (endix < len) && (endix > 0)) {
        header = buf;
        headend = &header[endix];
        headsize = (headend - header);

        // if the header size is < what we need to check for then just bail
        if (headsize < startLen) return -1;

        if ((post = calloc(1, sizeof(struct http_post_t))) == NULL) return -1;
        if ((hcopy = calloc(1, headsize + 4)) == NULL) {
            free(post);
            return -1;
        }

        if ((proto = calloc(1, sizeof(struct protocol_info_t))) == NULL) {
            free(post);
            free(hcopy);
            return -1;
        }

        post->start_duration = getTime();
        post->id = uid;
        strncpy(hcopy, header, headsize);
        post->hdr = hcopy;

        if ((src == TLSTX) || (src == TLSRX) ||
            (localPort == 443) || (remotePort == 443)) {
            post->ssl = 1;
        } else {
            post->ssl = 0;
        }

        // If the first 5 chars are HTTP/, it's a response header
        if (needleFind(g_http_start, hcopy, startLen) != -1) {
            proto->ptype = EVT_HRES;
        } else {
            proto->ptype = EVT_HREQ;
        }

        proto->evtype = EVT_PROTO;
        proto->len = headsize;
        proto->fd = sockfd;
        proto->data = (char *)post;
        cmdPostEvent(g_ctl, (char *)proto);
    }

    return 0;
}

