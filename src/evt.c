#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "evt.h"

typedef struct {
    int valid;
    regex_t re;
} local_re_t;

struct _evt_t
{
    transport_t* transport;
    format_t* format;
    local_re_t value_re[CFG_SRC_MAX];
    local_re_t field_re[CFG_SRC_MAX];
    local_re_t name_re[CFG_SRC_MAX];
    unsigned enabled[CFG_SRC_MAX];
    cbuf_handle_t evbuf;
};

static const char* valueFilterDefault[] = {
    DEFAULT_SRC_FILE_VALUE,
    DEFAULT_SRC_CONSOLE_VALUE,
    DEFAULT_SRC_SYSLOG_VALUE,
    DEFAULT_SRC_METRIC_VALUE,
};

static const char* fieldFilterDefault[] = {
    DEFAULT_SRC_FILE_FIELD,
    DEFAULT_SRC_CONSOLE_FIELD,
    DEFAULT_SRC_SYSLOG_FIELD,
    DEFAULT_SRC_METRIC_FIELD,
};

static const char* nameFilterDefault[] = {
    DEFAULT_SRC_FILE_NAME,
    DEFAULT_SRC_CONSOLE_NAME,
    DEFAULT_SRC_SYSLOG_NAME,
    DEFAULT_SRC_METRIC_NAME,
};

static unsigned srcEnabledDefault[] = {
    DEFAULT_SRC_FILE,
    DEFAULT_SRC_CONSOLE,
    DEFAULT_SRC_SYSLOG,
    DEFAULT_SRC_METRIC,
};


static void
filterSet(local_re_t* re, const char *str, const char* default_val)
{
    if (!re || !default_val) return;

    local_re_t temp;
    temp.valid = str && !regcomp(&temp.re, str, REG_EXTENDED | REG_NOSUB);
    if (!temp.valid) {
        // regcomp failed on str.  Try the default.
        temp.valid = !regcomp(&temp.re, default_val, REG_EXTENDED | REG_NOSUB);
    }

    if (temp.valid) {
        // Out with the old
        if (re->valid) regfree(&re->re);
        // In with the new
        *re = temp;
    } else {
        DBG("%s", str);
    }
}

evt_t*
evtCreate()
{
    evt_t* evt = calloc(1, sizeof(evt_t));
    if (!evt) {
        DBG(NULL);
        return NULL;
    }

    cfg_evt_t src;
    for (src=CFG_SRC_FILE;  src<CFG_SRC_MAX; src++) {
        filterSet(&evt->value_re[src], NULL, valueFilterDefault[src]);
        filterSet(&evt->field_re[src], NULL, fieldFilterDefault[src]);
        filterSet(&evt->name_re[src], NULL, nameFilterDefault[src]);
        evt->enabled[src] = srcEnabledDefault[src];
    }

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

    cfg_evt_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (edestroy->value_re[src].valid) regfree(&edestroy->value_re[src].re);
        if (edestroy->field_re[src].valid) regfree(&edestroy->field_re[src].re);
        if (edestroy->name_re[src].valid) regfree(&edestroy->name_re[src].re);
    }

    fmtDestroy(&edestroy->format);

    transportDestroy(&edestroy->transport);

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

    char* msg = fmtString(evt->format, e, NULL);
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

regex_t *
evtValueFilter(evt_t *evt, cfg_evt_t src)
{
    if (src < CFG_SRC_MAX) {
        if (evt && evt->value_re[src].valid) return &evt->value_re[src].re;
        static local_re_t default_re[CFG_SRC_MAX];
        if (!default_re[src].valid) {
            filterSet(&default_re[src], NULL, valueFilterDefault[src]);
        }
        if (default_re[src].valid) return &default_re[src].re;
    }
    DBG("%d", src);
    return NULL;
}

regex_t *
evtFieldFilter(evt_t *evt, cfg_evt_t src)
{
    if (src < CFG_SRC_MAX) {
        if (evt && evt->field_re[src].valid) return &evt->field_re[src].re;
        static local_re_t default_re[CFG_SRC_MAX];
        if (!default_re[src].valid) {
            filterSet(&default_re[src], NULL, fieldFilterDefault[src]);
        }
        if (default_re[src].valid) return &default_re[src].re;
    }
    DBG("%d", src);
    return NULL;
}

regex_t *
evtNameFilter(evt_t *evt, cfg_evt_t src)
{
    if (src < CFG_SRC_MAX) {
        if (evt && evt->name_re[src].valid) return &evt->name_re[src].re;
        static local_re_t default_re[CFG_SRC_MAX];
        if (!default_re[src].valid) {
            filterSet(&default_re[src], NULL, nameFilterDefault[src]);
        }
        if (default_re[src].valid) return &default_re[src].re;
    }
    DBG("%d", src);
    return NULL;
}

unsigned
evtSourceEnabled(evt_t* evt, cfg_evt_t src)
{
    if (src < CFG_SRC_MAX) {
        if (evt) return evt->enabled[src];
        return srcEnabledDefault[src];
    }

    DBG("%d", src);
    return srcEnabledDefault[CFG_SRC_FILE];
}

int
evtNeedsConnection(evt_t *evt)
{
    if (!evt) return 0;
    return transportNeedsConnection(evt->transport);
}

int
evtConnect(evt_t *evt)
{
    if (!evt) return 0;
    return transportConnect(evt->transport);
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
}

void
evtValueFilterSet(evt_t *evt, cfg_evt_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->value_re[src], str, valueFilterDefault[src]);
}

void
evtFieldFilterSet(evt_t *evt, cfg_evt_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->field_re[src], str, fieldFilterDefault[src]);
}

void
evtNameFilterSet(evt_t *evt, cfg_evt_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->name_re[src], str, nameFilterDefault[src]);
}

void
evtSourceEnabledSet(evt_t* evt, cfg_evt_t src, unsigned val)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    evt->enabled[src] = val;
}

#define MATCH_FOUND 1
#define NO_MATCH_FOUND 0

static int
anyValueFieldMatches(regex_t* filter, event_t* metric)
{
    if (!filter || !metric) return MATCH_FOUND;

    // Test the value of metric
    char valbuf[64];
    if (snprintf(valbuf, sizeof(valbuf), "%lld", metric->value) > 0) {
        if (!regexec(filter, valbuf, 0, NULL, 0)) return MATCH_FOUND;
    }

    // Test every field value until a match is found
    event_field_t *fld;
    for (fld = metric->fields; fld->value_type != FMT_END; fld++) {
        const char *str = NULL;
        if (fld->value_type == FMT_STR) {
            str = fld->value.str;
        } else if (fld->value_type == FMT_NUM) {
            if (snprintf(valbuf, sizeof(valbuf), "%lld", fld->value.num) > 0) {
                str = valbuf;
            }
        }

        if (str && !regexec(filter, str, 0, NULL, 0)) return MATCH_FOUND;
    }

    return NO_MATCH_FOUND;
}

int
evtMetric(evt_t *evt, const char *host, uint64_t uid, event_t *metric)
{
    char *msg;
    event_format_t event;
    struct timeb tb;
    char ts[128];
    regex_t *filter;

    if (!evt || !metric || !host || !metric) return -1;

    // Test for a name field match.  No match, no metric output
    if (!evtSourceEnabled(evt, CFG_SRC_METRIC) ||
        !(filter = evtNameFilter(evt, CFG_SRC_METRIC)) ||
        (regexec(filter, metric->name, 0, NULL, 0))) {
        return -1;
    }

    /*
     * Loop through all metric fields for at least one matching field value
     * No match, no metric output
     */
    if (!anyValueFieldMatches(evtValueFilter(evt, CFG_SRC_METRIC), metric)) {
        return -1;
    }

    ftime(&tb);
    if (snprintf(ts, sizeof(ts), "%ld.%d", tb.time, tb.millitm) < 0) return -1;
    event.timestamp = ts;
    event.timesize = strlen(ts);

    event.src = "metric";
    event.hostname = host;

    // Format the metric string using the configured metric format type
    event.data = fmtString(evt->format, metric, evtFieldFilter(evt, CFG_SRC_METRIC));
    if (!event.data) return -1;
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
       const void *buf, size_t count, uint64_t uid, cfg_evt_t logType)
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

    regex_t* filter = evtValueFilter(evt, logType);
    if (msg && filter && regexec(filter, msg, 0, NULL, 0)) {
        // This event doesn't match.  Drop it on the floor.
        free(msg);
        return 0;
    }

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
