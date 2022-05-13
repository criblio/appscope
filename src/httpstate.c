#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
//#include <lshpack.h>

#include "com.h"
#include "dbg.h"
#include "httpstate.h"
#include "plattime.h"
#include "search.h"
#include "atomic.h"
#include "scopestdlib.h"

#define MIN_HDR_ALLOC (4  * 1024)
#define MAX_HDR_ALLOC (16 * 1024)

#define HTTP2_MAGIC "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
#define HTTP2_MAGIC_LEN 24

#define HTTP_START "HTTP/"
#define HTTP_END "\r\n"
static search_t* g_http_start = NULL;
static search_t* g_http_end = NULL;

#define HTTP_CLENGTH "(?i)\\r\\ncontent-length: (\\d+)"
#define HTTP_UPGRADE "(?i)\\r\\nupgrade: h2"
#define HTTP_CONNECT "(?i)\\r\\nconnection: upgrade"
static pcre2_code *g_http_clength = NULL;
static pcre2_code *g_http_upgrade = NULL;
static pcre2_code *g_http_connect = NULL;

static void setHttpState(http_state_t *httpstate, http_enum_t toState);
static void appendHeader(http_state_t *httpstate, char* buf, size_t len);
static size_t getContentLength(char *header, size_t len);
static size_t bytesToSkipForContentLength(http_state_t *httpstate, size_t len);
static bool setHttpId(httpId_t *httpId, net_info *net, int sockfd, uint64_t id, metric_t src);
static int reportHttp1(http_state_t *httpstate);
static bool parseHttp1(http_state_t *httpstate, char *buf, size_t len, httpId_t *httpId);

extern int      g_http_guard_enabled;
extern uint64_t g_http_guard[];

static void
setHttpState(http_state_t *httpstate, http_enum_t toState)
{
    if (!httpstate) return;
    switch (toState) {
        case HTTP_NONE:
            // cleanup the stash for the RX and TX sides
            if (httpstate->http2Buf.buf) {
                scope_free(httpstate->http2Buf.buf);
                httpstate->http2Buf.buf = NULL;
            }
            httpstate->http2Buf.len = 0;
            httpstate->http2Buf.size = 0;
            if (httpstate->hdr) {
                scope_free(httpstate->hdr);
                httpstate->hdr = NULL;
            }

            httpstate->hdrlen = 0;
            httpstate->hdralloc = 0;
            httpstate->clen = 0;
            scope_memset(&(httpstate->id), 0, sizeof(httpId_t));
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

    // If we need more space, scope_realloc
    if (alloc_size != httpstate->hdralloc) {
        char* temp = scope_realloc(httpstate->hdr, alloc_size);
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
    scope_memcpy(&httpstate->hdr[httpstate->hdrlen], buf, len);
    httpstate->hdrlen += len;
}

// Returns the integer value if header block contains a "content-length" header
// or -1 if not
static size_t
getContentLength(char *header, size_t len)
{
    if (!g_http_clength) return -1;

    pcre2_match_data *matches = pcre2_match_data_create_from_pattern(g_http_clength, NULL);
    if (!matches) return -1;

    int rc = pcre2_match_wrapper(g_http_clength,
            (PCRE2_SPTR)header, (PCRE2_SIZE)len,
            0, 0, matches, NULL);
    if (rc != 2) {
        pcre2_match_data_free(matches);
        return -1;
    }

    PCRE2_UCHAR *cLen; PCRE2_SIZE cLenLen;
    pcre2_substring_get_bynumber(matches, 1, &cLen, &cLenLen);

    scope_errno = 0;
    size_t ret = scope_strtoull((const char *)cLen, NULL, 0);
    if ((scope_errno != 0) || (ret == 0)) {
        ret = -1;
    }

    pcre2_match_data_free(matches);
    pcre2_substring_free(cLen);

    return ret;
}

// Returns TRUE if header block contains an "upgrade: h2" header
static bool
hasUpgrade(const char *header, size_t len)
{
    if (!g_http_upgrade) return FALSE;

    pcre2_match_data *matches = pcre2_match_data_create_from_pattern(g_http_upgrade, NULL);
    if (!matches) return FALSE;

    int rc = pcre2_match_wrapper(g_http_upgrade,
            (PCRE2_SPTR)header, (PCRE2_SIZE)len,
            0, 0, matches, NULL);

    pcre2_match_data_free(matches);

    return rc == 1;
}

// Returns TRUE if header block contains an "connection: upgrade" header
static bool
hasConnectionUpgrade(const char *header, size_t len)
{
    if (!g_http_connect) return FALSE;

    pcre2_match_data *matches = pcre2_match_data_create_from_pattern(g_http_connect, NULL);
    if (!matches) return FALSE;

    int rc = pcre2_match_wrapper(g_http_connect,
            (PCRE2_SPTR)header, (PCRE2_SIZE)len,
            0, 0, matches, NULL);

    pcre2_match_data_free(matches);

    return rc == 1;
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
    if (net) {
        httpId->uid = net->uid;
    } else if (id != -1) {
        httpId->uid = id;
    } else {
        DBG(NULL);
        return FALSE;
    }

    httpId->isSsl = (src == TLSTX) || (src == TLSRX);

    httpId->src = src;

    httpId->sockfd = sockfd;

    return TRUE;
}

// For now, only doing HTTP/1.X headers
static int
reportHttp1(http_state_t *httpstate)
{
    if (!httpstate || !httpstate->hdr || !httpstate->hdrlen) return -1;

    protocol_info *proto = scope_calloc(1, sizeof(struct protocol_info_t));
    http_post *post = scope_calloc(1, sizeof(struct http_post_t));
    if (!proto || !post) {
        // Bummer!  We're losing info.  At least make sure we clean up.
        DBG(NULL);
        if (post) scope_free(post);
        if (proto) scope_free(proto);
        return -1;
    }

    // If the first 5 chars are HTTP/, it's a response header
    int isSend = (httpstate->id.src == NETTX) || (httpstate->id.src == TLSTX);

    // Set proto info
    proto->evtype = EVT_PROTO;
    proto->ptype = (httpstate->isResponse) ? EVT_HRES : EVT_HREQ;
    // We're a server if we 1) sent a response or 2) received a request
    proto->isServer = (isSend && httpstate->isResponse) || (!isSend && !httpstate->isResponse);
    proto->len = httpstate->hdrlen;
    proto->fd = httpstate->id.sockfd;
    proto->uid = httpstate->id.uid;

    net_info *net;
    if ((net = getNetEntry(proto->fd))) {
        proto->sock_type = net->type;
        if (net->addrSetLocal) {
            scope_memcpy(&proto->localConn, &net->localConn, sizeof(struct sockaddr_storage));
        } else {
            proto->localConn.ss_family = -1;
        }
        if (net->addrSetRemote) {
            scope_memcpy(&proto->remoteConn, &net->remoteConn, sizeof(struct sockaddr_storage));
        } else {
            proto->remoteConn.ss_family = -1;
        }
    } else {
        proto->sock_type = -1;
        proto->localConn.ss_family = -1;
        proto->remoteConn.ss_family = -1;
    }

    // Set post info
    proto->data = (char *)post;
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

static bool
reportHttp2(http_state_t *state, net_info *net, http_buf_t *stash,
        const uint8_t *buf, uint32_t frameLen, httpId_t *httpId)
{
    if (!state || !stash || !buf || !frameLen || !httpId) {
        scopeLogError("ERROR: NULL reportHttp2() parameter");
        DBG(NULL);
        return FALSE;
    }

    http_post *post = scope_calloc(1, sizeof(struct http_post_t));
    if (!post) {
        scopeLogError("ERROR: failed to allocate post object");
        DBG(NULL);
        return FALSE;
    }
    post->ssl            = state->id.isSsl;
    post->start_duration = getTime();
    post->id             = state->id.uid;
    post->hdr            = scope_malloc(frameLen);
    if (!post->hdr) {
        scope_free(post);
        scopeLogError("ERROR: failed to allocate post data");
        DBG(NULL);
        return FALSE;
    }
    if (stash->len) {
        scope_memcpy(post->hdr, stash->buf, stash->len);
        scope_memcpy(post->hdr + stash->len, buf, frameLen - stash->len);
    } else {
        scope_memcpy(post->hdr, buf, frameLen);
    }

    protocol_info *proto = scope_calloc(1, sizeof(struct protocol_info_t));
    if (!proto) {
        scope_free(post->hdr);
        scope_free(post);
        scopeLogError("ERROR: failed to allocate protocol object");
        DBG(NULL);
        return FALSE;
    }

    proto->evtype   = EVT_PROTO;
    proto->ptype    = EVT_H2FRAME;
    // Unlike in the HTTP/1 case, we're sending TRUE here if the frame was
    // sent, not if we're the server. We haven't parsed the frame to know if
    // it's a request or response yet so we're sending half of the isServer
    // answer here and will finish the logic on the reporting side.
    proto->isServer = (state->id.src == NETTX) || (state->id.src == TLSTX);
    proto->len      = frameLen;
    proto->fd       = httpId->sockfd;
    proto->uid      = httpId->uid;
    proto->data     = (char *)post;
    if (net) {
        proto->sock_type = net->type;
        if (net->addrSetLocal) {
            scope_memcpy(&proto->localConn, &net->localConn, sizeof(struct sockaddr_storage));
        } else {
            proto->localConn.ss_family = -1;
        }
        if (net->addrSetRemote) {
            scope_memcpy(&proto->remoteConn, &net->remoteConn, sizeof(struct sockaddr_storage));
        } else {
            proto->remoteConn.ss_family = -1;
        }
    } else {
        proto->sock_type            = -1;
        proto->localConn.ss_family  = -1;
        proto->remoteConn.ss_family = -1;
    }

    bool ret = (uint8_t)(post->hdr[4]) & 0x04; // TRUE if END_HEADERS flag set

    //scopeLogDebug("DEBUG: posting HTTP/2 frame");
    cmdPostEvent(g_ctl, (char *)proto);

    return ret;
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
parseHttp1(http_state_t *httpstate, char *buf, size_t len, httpId_t *httpId)
{
    if (!buf) return FALSE;

    // We need to handle "double interception" when ssl is involved, e.g.
    // intercepting an SSL_write(), then intercepting the write() it calls.
    // We use the isSSL flag to prevent interleaving ssl and non-ssl data.
    {
        int headerCaptureInProgress = httpstate->state != HTTP_NONE;
        int isSslIsConsistent = httpstate->id.isSsl == httpId->isSsl;
        if (headerCaptureInProgress && !isSslIsConsistent) return FALSE;
    }

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

        if (searchExec(g_http_start, buf, len) == -1) return FALSE;

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
            searchExec(g_http_end, &buf[header_start], len-header_start);

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
            header_end += searchLen(g_http_end);
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
        size_t content_in_this_buf = len - (header_start + header_end + searchLen(g_http_end));

        httpstate->isResponse =
            (searchExec(g_http_start, httpstate->hdr, searchLen(g_http_start)) != -1);
        httpstate->hasUpgrade = hasUpgrade(httpstate->hdr, httpstate->hdrlen);
        httpstate->hasConnectionUpgrade = hasConnectionUpgrade(httpstate->hdr, httpstate->hdrlen);

        // post and event containing the header we found
        reportHttp1(httpstate);

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
    g_http_start = searchComp(HTTP_START);
    g_http_end = searchComp(HTTP_END);

    int        errNum;
    PCRE2_SIZE errPos;

    if (!(g_http_clength = pcre2_compile((PCRE2_SPTR)HTTP_CLENGTH,
            PCRE2_ZERO_TERMINATED, 0, &errNum, &errPos, NULL))) {
        scopeLogError("ERROR: HTTP/1 content-length regex failed; err=%d, pos=%ld",
                errNum, errPos);
    }
    if (!(g_http_upgrade = pcre2_compile((PCRE2_SPTR)HTTP_UPGRADE,
            PCRE2_ZERO_TERMINATED, 0, &errNum, &errPos, NULL))) {
        scopeLogError("ERROR: HTTP/1 upgrade regex failed; err=%d, pos=%ld",
                errNum, errPos);
    }
    if (!(g_http_connect = pcre2_compile((PCRE2_SPTR)HTTP_CONNECT,
            PCRE2_ZERO_TERMINATED, 0, &errNum, &errPos, NULL))) {
        scopeLogError("ERROR: HTTP/1 connection regex failed; err=%d, pos=%ld",
                errNum, errPos);
    }
}

static void
http2StashFrame(http_buf_t *stash, const uint8_t *buf, size_t len)
{
    if (!stash) {
        scopeLogError("ERROR: null stash");
        DBG(NULL);
        return;
    }

    // need to store the `len` we're given plus whatever's already stashed
    size_t need = len + stash->len;
    if (need > stash->size) {
        // round up to the next 1k boundary and scope_realloc
        need = ((need + 1023) / 1024) * 1024;
        uint8_t *newBuf = scope_realloc(stash->buf, need);
        if (!newBuf) {
            scopeLogError("ERROR: failed to (re)allocate frame buffer");
            DBG(NULL);
            return;
        }
        stash->buf = newBuf;
        stash->size = need;
    }

    // append what we're given to the stash
    scope_memcpy(stash->buf + stash->len, buf, len);
    stash->len += len;
}

static uint32_t
http2GetFrameLength(http_buf_t *stash, const uint8_t *buf, size_t len)
{
    if (!stash) {
        scopeLogError("ERROR: null stash");
        DBG(NULL);
        return -1;
    }

    uint32_t ret = 0;

    // The first three bytes (MSB first) of the frame header are the length but
    // some of the frame may already be stashed so this looks weird to deal
    // with the potential split.

    if (stash->len > 0) {
        ret += stash->buf[0] << 16;
    } else {
        ret += buf[0 - stash->len] << 16;
    }

    if (stash->len > 1) {
        ret += stash->buf[1] << 8;
    } else {
        ret += buf[1 - stash->len] << 8;
    }

    if (stash->len > 2) {
        ret += stash->buf[2];
    } else {
        ret += buf[2 - stash->len];
    }

    return ret;
}

static uint8_t
http2GetFrameType(http_buf_t *stash, const uint8_t *buf, size_t len)
{
    if (!stash) {
        scopeLogError("ERROR: null stash");
        DBG(NULL);
        return -1;
    }

    uint8_t ret = 0;

    // The fourth byte of the frame header is the type but some of the frame
    // may already be stashed so this looks weird to deal with the potential
    // split.

    if (stash->len > 3) {
        ret += stash->buf[3];
    } else {
        ret += buf[3 - stash->len];
    }

    return ret;
}


static uint8_t
http2GetFrameFlags(http_buf_t *stash, const uint8_t *buf, size_t len)
{
    if (!stash) {
        scopeLogError("ERROR: null stash");
        DBG(NULL);
        return -1;
    }

    uint8_t ret = 0;

    // The fifth byte of the frame header is the flags but some of the frame
    // may already be stashed so this looks weird to deal with the potential
    // split.

    if (stash->len > 4) {
        ret += stash->buf[4];
    } else {
        ret += buf[4 - stash->len];
    }

    return ret;
}



static uint32_t
http2GetFrameStream(http_buf_t *stash, const uint8_t *buf, size_t len)
{
    if (!stash) {
        scopeLogError("ERROR: null stash");
        DBG(NULL);
        return -1;
    }

    uint32_t ret = 0;

    // The sixth thru ninth bytes (MSB first) of the frame header are the
    // stream ID but some of the frame may already be stashed so this looks
    // weird to deal with the potential split.

    if (stash->len > 5) {
        ret += (stash->buf[5]&0x7F) << 24;
    } else {
        ret += (buf[5 - stash->len]&0x7F) << 24;
    }

    if (stash->len > 6) {
        ret += stash->buf[6] << 16;
    } else {
        ret += buf[6 - stash->len] << 16;
    }

    if (stash->len > 7) {
        ret += stash->buf[7] << 8;
    } else {
        ret += buf[7 - stash->len] << 8;
    }

    if (stash->len > 8) {
        ret += stash->buf[8];
    } else {
        ret += buf[8 - stash->len];
    }

    return ret;
}


static bool
parseHttp2(http_state_t* state, net_info *net, int isTx,
        const uint8_t *buf, size_t len, httpId_t *httpId)
{
    if (!buf || !len) {
        scopeLogError("ERROR: empty HTTP/2 buffer");
        DBG(NULL);
        return FALSE;
    }

    bool ret = FALSE; // TRUE if we saw the end of a header like parseHttp1()

    const uint8_t *bufPos = buf;                    // current position in buf
    size_t         bufLen = len;                    // number of buf bytes left
    http_buf_t    *stash  = &state->http2Buf;       // stash for partial frames
    while (bufLen > 0) {
        // skip over MAGIC
        if (bufLen >= HTTP2_MAGIC_LEN && !scope_strncmp((char*)buf, HTTP2_MAGIC, HTTP2_MAGIC_LEN)) {
            bufPos += HTTP2_MAGIC_LEN;
            bufLen -= HTTP2_MAGIC_LEN;
            if (!bufLen) return FALSE;
        }

        // stash the buffer if we don't have enough for a frame header
        if (stash->len + bufLen < 9) {
            http2StashFrame(stash, bufPos, bufLen);
            return FALSE;
        }

        // get the header values
        uint32_t fLen    = http2GetFrameLength(stash, bufPos, bufLen);
        uint8_t  fType   = http2GetFrameType(stash, bufPos, bufLen);
        uint8_t  fFlags  = http2GetFrameFlags(stash, bufPos, bufLen);
        uint32_t fStream = http2GetFrameStream(stash, bufPos, bufLen);

        // stash the buffer if we don't have enough for the whole frame
        if (stash->len + bufLen < (9 + fLen)) {
            http2StashFrame(stash, bufPos, bufLen);
            return FALSE;
        }

        scopeLogDebug("DEBUG: HTTP/2 %s frame found; type=0x%02x, flags=0x%02x, stream=%d",
                isTx ? "TX" : "RX", fType, fFlags, fStream);

        // process interesting frames
        switch (fType) {
            case 0x01:
                // process HEADERS frames
                ret |= reportHttp2(state, net, stash, bufPos, fLen+9, httpId);
                break;
            case 0x05:
                // process PUSH_PROMISE frames (unsolicited requests)
                ret |= reportHttp2(state, net, stash, bufPos, fLen+9, httpId);
                break;
            case 0x09:
                // process CONTINUATION frames (additional HEADERS)
                ret |= reportHttp2(state, net, stash, bufPos, fLen+9, httpId);
                break;
            default:
                // not interested in other frames
                break;
        }

        // skip over what we parsed in the buffer and clear the stash
        size_t bytesParsed = fLen + 9 - stash->len;
        bufPos += bytesParsed;
        bufLen -= bytesParsed;
        stash->len = 0; // the stash was parsed too
    }

    return ret;
}

static bool
doHttpBuffer(http_state_t states[HTTP_NUM], net_info *net, char *buf, size_t len,
        metric_t src, httpId_t *httpId)
{
    int isTx  = (src == NETTX || src == TLSTX) ? 1 : 0;
    http_state_t *state = &states[isTx];

    //scopeLogHexDebug(buf, len>64 ? 64 : len, "DEBUG: HTTP %s payload; ver=%d, len=%ld",
    //        isTx ? "TX" : "RX", (int)state->version[isTx], len);

    // detect HTTP version
    if (state->version == 0) {
        // Detect HTTP/2 by looking for the "magic" string at the start
        if (len >= HTTP2_MAGIC_LEN && !scope_strncmp(buf, HTTP2_MAGIC, HTTP2_MAGIC_LEN)) {
            states[HTTP_RX].version = 2;
            states[HTTP_TX].version = 2;

            // is there anything after the MAGIC?
            if (len > HTTP2_MAGIC_LEN) {
                // continue processing the remainder of the payload
                buf += HTTP2_MAGIC_LEN;
                len -= HTTP2_MAGIC_LEN;
            } else {
                // otherwise, done
                return FALSE;
            }
        }

        // Detect HTTP/1.x by looking for "HTTP/" which appears as the start
        // of a response and after method and URI in a request.
        else if (searchExec(g_http_start, buf, len) != -1) {
            states[HTTP_RX].version = 1;
            states[HTTP_TX].version = 1;
            // fall through to continue processing
        }

        else {
            scopeLogError("ERROR: HTTP version detection failed");
            DBG(NULL);
            return FALSE;
        }
    }

    if (state->version == 1) {
        // process the HTTP/1.x payload
        int ret = parseHttp1(state, buf, len, httpId);

        // if we saw a successful upgrade response...
        if (state->isResponse && state->hasUpgrade && state->hasConnectionUpgrade) {
            // upgrade the versions on both sides, TX and RX
            states[HTTP_RX].version = 2;
            states[HTTP_TX].version = 2;

            // clear state
            state->isResponse = FALSE;
            state->hasUpgrade = FALSE;
            state->hasConnectionUpgrade = FALSE;

            // See Issue #601.
            //   We're ignoring the possibility that there is an HTTP/2 frame
            //   tacked to the end of the HTTP/1.x response here. It happens!
        }

        return ret;
    }

    if (state->version == 2) {
        // process the HTTP/2 payload
        return parseHttp2(state, net, isTx, (uint8_t*)buf, len, httpId);
    }

    // invalid HTTP version
    scopeLogError("ERROR: HTTP/? unexpected version on %s; version=%d",
            isTx ? "TX" : "RX", state->version);
    DBG(NULL);
    return FALSE;
}

bool
doHttp(uint64_t id, int sockfd, net_info *net, char *buf, size_t len, metric_t src, src_data_t dtype)
{
    if (!buf || !len) {
        scopeLogWarn("WARN: doHttp() got no buffer");
        return FALSE;
    }

    // If we know we're not looking at a stream, bail.
    if (net && net->type != SOCK_STREAM) {
        scopeLogWarn("WARN: doHttp() not on SOCK_STREAM");
        return FALSE;
    }

    httpId_t httpId = {0};
    if (!setHttpId(&httpId, net, sockfd, id, src)) return FALSE;

    int guard_enabled = g_http_guard_enabled && net;
    if (guard_enabled) while (!atomicCasU64(&g_http_guard[sockfd], 0ULL, 1ULL));

    int http_header_found = FALSE;

    // We won't always have net (looking at you, gnutls).  If net exists,
    // then we can use the http state it provides to look across multiple
    // buffers.  If net doesn't exist,  we can at least keep temp state
    // while within the current doHttp().
    http_state_t tempstate[HTTP_NUM] = {0};
    http_state_t (*httpstate)[HTTP_NUM] = (net) ? &net->http : &tempstate;

    // Handle the data in it's various formats
    switch (dtype) {
        case BUF:
        {
            http_header_found = doHttpBuffer(*httpstate, net, buf, len, src, &httpId);
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
                    if (doHttpBuffer(*httpstate, net, iov->iov_base, iov->iov_len, src, &httpId)) {
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
                    if (doHttpBuffer(*httpstate, net, iov[i].iov_base, iov[i].iov_len, src, &httpId)) {
                        http_header_found = TRUE;
                        // stay in loop to count down content length
                    }
                }
            }
            break;
        }

        default:
            scopeLogWarn("WARN: doHttp() got unknown data type; %d", dtype);
            DBG("%d", dtype);
            break;
    }

    // If our state is temporary, clean up after each doHttp call
    if (httpstate == &tempstate) {
        resetHttp(*httpstate);
    }

    if (guard_enabled) while (!atomicCasU64(&g_http_guard[sockfd], 1ULL, 0ULL));

    return http_header_found;
}

void
resetHttp(http_state_t httpstate[HTTP_NUM])
{
    setHttpState(&httpstate[HTTP_RX], HTTP_NONE);
    setHttpState(&httpstate[HTTP_TX], HTTP_NONE);
}

