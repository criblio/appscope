#define _GNU_SOURCE

#include <errno.h>

#include "circbuf.h"
#include "com.h"
#include "dbg.h"
#include "metriccapture.h"
#include "scopestdlib.h"

// Consistent with src/sluice/js/input/MetricsIn.ts
#define STATSD         "^([^:]+):([\\d.]+)\\|(c|g|ms|s|h)$"
#define STATSD_CAPTURE_GROUPS 4
#define STATSD_EXT     "^([^:]+):([\\d.]+)\\|(c|g|ms|s|h)\\|(#[^$]+)$"
#define STATSD_EXT_CAPTURE_GROUPS 5
#define STATSD_EXT_DIM "((#|,)([^:]+):([^,]+))"

static pcre2_code *g_statsd_regex = NULL;
static pcre2_code *g_statsd_ext_regex = NULL;
static cbuf_handle_t g_metric_buf = NULL;

void
initMetricCapture(void)
{
    int        errNum;
    PCRE2_SIZE errPos;

    if (!g_statsd_regex) {
        if (!(g_statsd_regex = pcre2_compile((PCRE2_SPTR)STATSD,
                PCRE2_ZERO_TERMINATED, 0, &errNum, &errPos, NULL))) {
            scopeLogError("ERROR: statsd regex failed; err=%d, pos=%ld",
                    errNum, errPos);
        }
    }
    if (!g_statsd_ext_regex) {
        if (!(g_statsd_ext_regex = pcre2_compile((PCRE2_SPTR)STATSD_EXT,
                PCRE2_ZERO_TERMINATED, 0, &errNum, &errPos, NULL))) {
            scopeLogError("ERROR: statsd extended regex failed; err=%d, pos=%ld",
                    errNum, errPos);
        }
    }

    size_t buf_size = DEFAULT_CBUF_SIZE;
    char *qlen_str;
    if ((qlen_str = getenv("SCOPE_QUEUE_LENGTH")) != NULL) {
        unsigned long qlen;
        scope_errno = 0;
        qlen = scope_strtoul(qlen_str, NULL, 10);
        if (!scope_errno && qlen) {
            buf_size = qlen;
        }
    }

    if (!(g_metric_buf = cbufInit(buf_size))) {
        scopeLogError("ERROR: statsd buffer creation failed");
    }
}

static captured_metric_t *
createCapturedMetric(unsigned char *name, unsigned char *value,
                     unsigned char *type, unsigned char *dims)
{
    captured_metric_t *base = NULL;

    // These are required fields. (dims may be null)
    if (!name || !value || !type) goto err;

    // Alloc space
    base = scope_malloc(sizeof(captured_metric_t));
    if (!base) goto err;

    // Copy in field values
    base->name = name;
    base->value = value;
    base->type = type;
    base->dims = dims;

    return base;

err:
    if (name) pcre2_substring_free(name);
    if (value) pcre2_substring_free(value);
    if (type) pcre2_substring_free(type);
    if (dims) pcre2_substring_free(dims);
    if (base) scope_free(base);
    return NULL;
}

static void
destroyCapturedMetric(captured_metric_t **metric)
{
    if (!metric || !*metric) return;
    captured_metric_t *tmp = *metric;
    if (tmp->name) pcre2_substring_free(tmp->name);
    if (tmp->value) pcre2_substring_free(tmp->value);
    if (tmp->type) pcre2_substring_free(tmp->type);
    if (tmp->dims) pcre2_substring_free(tmp->dims);
    scope_free(tmp);
    *metric = NULL;
}

// defined in report.c
extern void reportCapturedMetric(const captured_metric_t *metric);

void
reportAllCapturedMetrics(void)
{
    if (!g_metric_buf) return;

    uint64_t data;
    while (cbufGet(g_metric_buf, &data) == 0) {
        if (!data) continue;
        captured_metric_t *metric = (captured_metric_t *)data;

        reportCapturedMetric(metric);

        destroyCapturedMetric(&metric);
    }
}

static bool
doMetricBuffer(uint64_t id, int sockfd, net_info *net, char *buf, size_t len, metric_t src)
{
    bool is_successful = FALSE;
    pcre2_match_data *matches = NULL;
    captured_metric_t *metric = NULL;

    if (!g_statsd_regex || !g_statsd_ext_regex || !g_metric_buf) goto out;

    // Try matching "extended statsd" first
    matches = pcre2_match_data_create_from_pattern(g_statsd_ext_regex, NULL);
    if (!matches) goto out;

    int rc = pcre2_match_wrapper(g_statsd_ext_regex,
            (PCRE2_SPTR)buf, (PCRE2_SIZE)len, 0, 0, matches, NULL);
    if (rc != STATSD_EXT_CAPTURE_GROUPS) {
        // Didn't get expected matches.  Try "standard statsd" next.
        pcre2_match_data_free(matches);
        matches = pcre2_match_data_create_from_pattern(g_statsd_regex, NULL);
        if (!matches) goto out;

        rc = pcre2_match_wrapper(g_statsd_regex,
                (PCRE2_SPTR)buf, (PCRE2_SIZE)len, 0, 0, matches, NULL);
        if (rc != STATSD_CAPTURE_GROUPS) goto out;
    }

    // Create a metric object to store the matching results
    {
        PCRE2_SIZE unused;
        PCRE2_UCHAR *name = NULL;
        PCRE2_UCHAR *val = NULL;
        PCRE2_UCHAR *type = NULL;
        PCRE2_UCHAR *dims = NULL;
        pcre2_substring_get_bynumber(matches, 1, &name, &unused);
        pcre2_substring_get_bynumber(matches, 2, &val, &unused);
        pcre2_substring_get_bynumber(matches, 3, &type, &unused);
        // Dimensions are only there for extended statsd
        if (rc == STATSD_EXT_CAPTURE_GROUPS) {
            pcre2_substring_get_bynumber(matches, 4, &dims, &unused);
        }

        if (!(metric = createCapturedMetric(name, val, type, dims))) goto out;
    }

    // Put the metric on a circular buffer for later reporting
    if (cbufPut(g_metric_buf, (uint64_t)metric) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        destroyCapturedMetric(&metric);
        goto out;
    }

    is_successful = TRUE;
out:
    if (matches) pcre2_match_data_free(matches);
    return is_successful;
}

bool
doMetricCapture(uint64_t id, int sockfd, net_info *net, char *buf, size_t len, metric_t src, src_data_t dtype)
{
    bool ret = FALSE;

    if (dtype == BUF) {
        // simple buffer, pass it through
        ret = doMetricBuffer(id, sockfd, net, buf, len, src);
    } else if (dtype == MSG) {
        // buffer is a msghdr for sendmsg/recvmsg
        int i;
        struct msghdr *msg = (struct msghdr *)buf;
        struct iovec *iov;
        for (i = 0; i < msg->msg_iovlen; i++) {
            iov = &msg->msg_iov[i];
            if (iov && iov->iov_base && (iov->iov_len > 0)) {
                ret = ret || doMetricBuffer(id, sockfd, net, iov->iov_base, iov->iov_len, src);
            }
        }
    } else if (dtype == IOV) {
        // buffer is an iovec, len is the iovcnt
        int i;
        int iovcnt = len;
        struct iovec *iov = (struct iovec *)buf;
        for (i = 0; i < iovcnt; i++) {
            if (iov[i].iov_base && (iov[i].iov_len > 0)) {
                ret = ret || doMetricBuffer(id, sockfd, net, iov[i].iov_base, iov[i].iov_len, src);
            }
        }
    } else {
        scopeLogWarn("WARN: doMetric() got unknown data type; %d", dtype);
        DBG("%d", dtype);
    }

    return ret;
}
