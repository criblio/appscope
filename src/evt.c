#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "evt.h"

struct _evt_t
{
    transport_t* transport;
    format_t* format;
    regex_t* log_file_re;
    regex_t* metric_name_re;
    regex_t* metric_value_re;
    regex_t* metric_field_re;
    cbuf_handle_t evbuf;
    unsigned src[CFG_SRC_MAX];
};


evt_t*
evtCreate()
{
    evt_t* evt = calloc(1, sizeof(evt_t));
    if (!evt) {
        DBG(NULL);
        return NULL;
    }

    if (!(evt->log_file_re = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (regcomp(evt->log_file_re, DEFAULT_LOG_FILE_FILTER, REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (!(evt->metric_name_re = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (regcomp(evt->metric_name_re, DEFAULT_METRIC_NAME_FILTER, REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (!(evt->metric_value_re = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (regcomp(evt->metric_value_re, DEFAULT_METRIC_VALUE_FILTER, REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (!(evt->metric_field_re = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    if (regcomp(evt->metric_field_re, DEFAULT_METRIC_FIELD_FILTER, REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        evtDestroy(&evt);
        return evt;
    }

    // format wants this
    fmtMetricFieldFilterSet(evt->format, evt->metric_field_re);

    evt->evbuf = cbufInit(DEFAULT_CBUF_SIZE);

    return evt;
}

void
evtDestroy(evt_t** evt)
{
    if (!evt || !*evt) return;
    evt_t *edestroy  = *evt;

    evtEvents(edestroy); // Try to send events.  We're doing this in an
                         // attempt to empty the evbuf before the evbuf
                         // is destroyed.  Any events added after evtEvents
                         // and before cbufFree wil be leaked.
    cbufFree(edestroy->evbuf);

    transportDestroy(&edestroy->transport);

    if (edestroy->log_file_re) {
        regfree(edestroy->log_file_re);
        free(edestroy->log_file_re);
    }

    if (edestroy->metric_name_re) {
        regfree(edestroy->metric_name_re);
        free(edestroy->metric_name_re);
    }

    if (edestroy->metric_value_re) {
        regfree(edestroy->metric_value_re);
        free(edestroy->metric_value_re);
    }

    if (edestroy->metric_field_re) {
        regfree(edestroy->metric_field_re);
        free(edestroy->metric_field_re);
    }

    fmtDestroy(&edestroy->format);
    free(edestroy);
    *evt = NULL;
}

int
evtSend(evt_t* evt, const char* msg)
{
    if (!evt || !msg) return -1;

    return transportSend(evt->transport, msg);
    return 0;
}

int
evtSendEvent(evt_t* evt, event_t* e)
{
    if (!evt || !e) return -1;

    char* msg = fmtString(evt->format, e);
    int rv = evtSend(evt, msg);
    if (msg) free(msg);
    return rv;
}

void
evtFlush(evt_t* evt)
{
    if (!evt) return;
    transportFlush(evt->transport);
}

regex_t*
evtLogFileFilter(evt_t* evt)
{

    if (!evt) {
        static regex_t* default_log_file_re = NULL;

        if (default_log_file_re) return default_log_file_re;

        if (!(default_log_file_re = calloc(1, sizeof(regex_t)))) {
            DBG(NULL);
            return NULL;
        }

        if (regcomp(default_log_file_re, DEFAULT_LOG_FILE_FILTER, REG_EXTENDED)) {
            // regcomp failed.
            DBG(NULL);
            free(default_log_file_re);
            default_log_file_re = NULL;
        }
        return default_log_file_re;
    }

    return evt->log_file_re;
}

regex_t *
evtMetricNameFilter(evt_t *evt)
{

    if (!evt) {
        static regex_t *default_metric_name_re = NULL;

        if (default_metric_name_re) return default_metric_name_re;

        if (!(default_metric_name_re = calloc(1, sizeof(regex_t)))) {
            DBG(NULL);
            return NULL;
        }

        if (regcomp(default_metric_name_re, DEFAULT_METRIC_NAME_FILTER, REG_EXTENDED)) {
            // regcomp failed.
            DBG(NULL);
            free(default_metric_name_re);
            default_metric_name_re = NULL;
        }
        return default_metric_name_re;
    }

    return evt->metric_name_re;
}

regex_t *
evtMetricValueFilter(evt_t *evt)
{

    if (!evt) {
        static regex_t *default_metric_value_re = NULL;

        if (default_metric_value_re) return default_metric_value_re;

        if (!(default_metric_value_re = calloc(1, sizeof(regex_t)))) {
            DBG(NULL);
            return NULL;
        }

        if (regcomp(default_metric_value_re, DEFAULT_METRIC_VALUE_FILTER, REG_EXTENDED)) {
            // regcomp failed.
            DBG(NULL);
            free(default_metric_value_re);
            default_metric_value_re = NULL;
        }

        return default_metric_value_re;
    }

    return evt->metric_value_re;
}

regex_t *
evtMetricFieldFilter(evt_t *evt)
{

    if (!evt) {
        static regex_t *default_metric_field_re = NULL;

        if (default_metric_field_re) return default_metric_field_re;

        if (!(default_metric_field_re = calloc(1, sizeof(regex_t)))) {
            DBG(NULL);
            return NULL;
        }

        if (regcomp(default_metric_field_re, DEFAULT_METRIC_FIELD_FILTER, REG_EXTENDED)) {
            // regcomp failed.
            DBG(NULL);
            free(default_metric_field_re);
            default_metric_field_re = NULL;
            fmtMetricFieldFilterSet(evt->format, NULL);
        }

        // format wants this
        fmtMetricFieldFilterSet(evt->format, default_metric_field_re);
        return default_metric_field_re;
    }

    return evt->metric_field_re;
}

unsigned
evtSource(evt_t* evt, cfg_evt_t src)
{
    if (evt && src < CFG_SRC_MAX) {
        return evt->src[src];
    }

    switch (src) {
        case CFG_SRC_LOGFILE:
            return DEFAULT_SRC_LOGFILE;
        case CFG_SRC_CONSOLE:
            return DEFAULT_SRC_CONSOLE;
        case CFG_SRC_SYSLOG:
            return DEFAULT_SRC_SYSLOG;
        case CFG_SRC_METRIC:
            return DEFAULT_SRC_METRIC;
        default:
            DBG(NULL);
            return DEFAULT_SRC_LOGFILE;
    }
}

int
evtConnected(evt_t *evt)
{
    if (!evt) return 0;
    return transportConnected(evt->transport);
}

int
evtConnect(evt_t *evt, config_t *cfg)
{
    if (cfgTransportType(cfg, CFG_EVT) == CFG_TCP) {
        transport_t *trans;

        trans = transportCreateTCP(cfgTransportHost(cfg, CFG_EVT),
                                   cfgTransportPort(cfg, CFG_EVT));
        if (trans) {
            evtTransportSet(evt, trans);
            return 1;
        } else {
            return 0;
        }
    }

    return 0;
}


void
evtTransportSet(evt_t* evt, transport_t* transport)
{
    if (!evt) return;

    // Don't leak if evtTransportSet is called repeatedly
    transportDestroy(&evt->transport);
    evt->transport = transport;
}

void
evtFormatSet(evt_t* evt, format_t* format)
{
    if (!evt) return;

    // Don't leak if evtFormatSet is called repeatedly
    fmtDestroy(&evt->format);
    evt->format = format;
    fmtMetricFieldFilterSet(evt->format, evt->metric_field_re);
}

void
evtLogFileFilterSet(evt_t* evt, const char* str)
{
    if (!evt) return;

    regex_t* temp_re = calloc(1, sizeof(regex_t));
    if (!temp_re) {
        DBG(NULL);
        return;
    }

    const char* filter_string;
    if (!str || str[0]=='\0') {
        // no str.  Revert to default filter
        filter_string = DEFAULT_LOG_FILE_FILTER;
    } else {
        filter_string = str;
    }

    int temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    if (!temp_re_valid) {
        // regcomp failed on filter_string.  Try the default
        filter_string = DEFAULT_LOG_FILE_FILTER;
        temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    }

    if (temp_re_valid) {
        // Out with the old
        regfree(evt->log_file_re);
        free(evt->log_file_re);
        // In with the new
        evt->log_file_re = temp_re;
    } else {
        DBG("%s", str);
        free(temp_re);
    }
}

void
evtMetricNameFilterSet(evt_t *evt, const char *str)
{
    if (!evt) return;

    regex_t *temp_re = calloc(1, sizeof(regex_t));
    if (!temp_re) {
        DBG(NULL);
        return;
    }

    const char *filter_string;
    if (!str || str[0]=='\0') {
        // no str.  Revert to default filter
        filter_string = DEFAULT_METRIC_NAME_FILTER;
    } else {
        filter_string = str;
    }

    int temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    if (!temp_re_valid) {
        // regcomp failed on filter_string.  Try the default
        filter_string = DEFAULT_METRIC_NAME_FILTER;
        temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    }

    if (temp_re_valid) {
        // Out with the old
        regfree(evt->metric_name_re);
        free(evt->metric_name_re);
        // In with the new
        evt->metric_name_re = temp_re;
    } else {
        DBG("%s", str);
        free(temp_re);
    }
}

void
evtMetricValueFilterSet(evt_t *evt, const char *str)
{
    if (!evt) return;

    regex_t *temp_re = calloc(1, sizeof(regex_t));
    if (!temp_re) {
        DBG(NULL);
        return;
    }

    const char *filter_string;
    if (!str || str[0]=='\0') {
        // no str.  Revert to default filter
        filter_string = DEFAULT_METRIC_VALUE_FILTER;
    } else {
        filter_string = str;
    }

    int temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    if (!temp_re_valid) {
        // regcomp failed on filter_string.  Try the default
        filter_string = DEFAULT_METRIC_VALUE_FILTER;
        temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    }

    if (temp_re_valid) {
        // Out with the old
        regfree(evt->metric_value_re);
        free(evt->metric_value_re);
        // In with the new
        evt->metric_value_re = temp_re;
    } else {
        DBG("%s", str);
        free(temp_re);
    }
}

void
evtMetricFieldFilterSet(evt_t *evt, const char *str)
{
    if (!evt) return;

    regex_t *temp_re = calloc(1, sizeof(regex_t));
    if (!temp_re) {
        DBG(NULL);
        return;
    }

    const char *filter_string;
    if (!str || str[0]=='\0') {
        // no str.  Revert to default filter
        filter_string = DEFAULT_METRIC_FIELD_FILTER;
    } else {
        filter_string = str;
    }

    int temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    if (!temp_re_valid) {
        // regcomp failed on filter_string.  Try the default
        filter_string = DEFAULT_METRIC_FIELD_FILTER;
        temp_re_valid = !regcomp(temp_re, filter_string, REG_EXTENDED);
    }

    if (temp_re_valid) {
        // Out with the old
        regfree(evt->metric_field_re);
        free(evt->metric_field_re);
        // In with the new
        evt->metric_field_re = temp_re;
        // format wants this
        fmtMetricFieldFilterSet(evt->format, evt->metric_field_re);
    } else {
        DBG("%s", str);
        free(temp_re);
    }
}

void
evtSourceSet(evt_t* evt, cfg_evt_t src, unsigned val)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    evt->src[src] = val;
}

int
evtMetric(evt_t *evt, const char *host, uint64_t uid, event_t *metric)
{
    int go = 0;
    char *msg;
    event_format_t event;
    struct timeb tb;
    char ts[128];
    regmatch_t match = {0};
    event_field_t *fld;
    regex_t *value_filter;

    if (!evt || !metric || !host ||
        (evtSource(evt, CFG_SRC_METRIC) == DEFAULT_SRC_METRIC) ||
        (regexec(evtMetricNameFilter(evt), metric->name, 1, &match, 0) != 0))
        return -1;

    /*
     * Loop through all metrics to test a value filter
     * At least one match must occur in order to emit this metric
     */
    value_filter = evtMetricValueFilter(evt);
    if (value_filter != NULL) {
        for (fld = metric->fields; fld->value_type != FMT_END; fld++) {
            if ((fld->value_type == FMT_STR) &&
                (regexec(value_filter, fld->value.str, 1, &match, 0) == 0)) {
                go = 1;
                break;
            }

            if (fld->value_type == FMT_NUM) {
                if (snprintf(ts, sizeof(ts), "%lld" , fld->value.num) < 0) {
                    continue;
                }

                if (regexec(value_filter, ts, 1, &match, 0) == 0) {
                    go = 1;
                    break;
                }
            }
        }
    }

    // At least one value filter needs to match
    if (go == 0) {
        return -1;
    }

    ftime(&tb);
    snprintf(ts, sizeof(ts), "%ld.%d", tb.time, tb.millitm);
    event.timestamp = ts;
    event.timesize = strlen(ts);

    event.src = "metric";
    event.hostname = host;

    // Format the metric string using the configured metric format type
    if ((event.data = fmtString(evt->format, metric)) == NULL) return -1;
    event.datasize = strlen(event.data);

    event.uid = uid;

    msg = fmtEventMessageString(evt->format, &event);
    if (JSON_DEBUG != 0) strcat(msg, "\n");

    if (cbufPut(evt->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free(msg);
    }

    free(event.data);
    return 0;
}

int
evtLog(evt_t *evt, const char *host, const char *path,
       const void *buf, size_t count, uint64_t uid)
{
    char *msg;
    event_format_t event;
    struct timeb tb;
    char ts[128];

    if (!evt || !buf || !path || !host) return -1;

    ftime(&tb);
    snprintf(ts, sizeof(ts), "%ld.%d", tb.time, tb.millitm);
    event.timestamp = ts;
    event.timesize = strlen(ts);
    event.src = path;
    event.hostname = host;
    event.data = (char *)buf;
    event.datasize = count;
    event.uid = uid;

    msg = fmtEventMessageString(evt->format, &event);

    if (cbufPut(evt->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free(msg);
    }

    return 0;
}

int
evtEvents(evt_t *evt)
{
    uint64_t data;

    if (!evt) return -1;

    while (cbufGet(evt->evbuf, &data) == 0) {
        if (data) {
            char *event = (char *)data;
            evtSend(evt, event);
            free(event);
        }
    }
    return 0;
}
