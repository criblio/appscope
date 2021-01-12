#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>

#include "dbg.h"
#include "evtformat.h"
#include "com.h"


// This is ugly, but...
// In glibc 2.27 (ubuntu 18.04) and earlier clock_gettime
// resolved to 'clock_gettime@@GLIBC_2.2.5'.
// In glibc 2.31 (ubuntu 20.04), it's now 'clock_gettime@@GLIBC_2.17'.
// This voodoo avoids creating binaries which would require a glibc
// that's 2.31 or newer, even if built on new ubuntu 20.04.
// This is tested by test/glibcvertest.c.
//#ifdef __LINUX__
//__asm__(".symver clock_gettime,clock_gettime@GLIBC_2.2.5");
//#endif


// key names for event JSON
#define HOST "host"
#define TIME "_time"
#define DATA "data"
#define SOURCE "source"
#define CHANNEL "_channel"
#define SOURCETYPE "sourcetype"
#define EVENT "ev"
#define ID "id"
#define PROCNAME "proc"
#define CMDNAME "cmd"
#define PID "pid"

#define FMTERR(e, j)                                             \
    {                                                            \
    DBG("time=%s src=%s data=%p json=%p",                        \
        e.timestamp, e.src, e.data, json);                       \
    if (e.data) cJSON_Delete(e.data);                            \
    if (j) cJSON_Delete(j);                                      \
    return NULL;                                                 \
    }

typedef struct {
    const char* str;
    unsigned val;
} enum_map_t;

static enum_map_t watchTypeMap[] = {
    {"file",         CFG_SRC_FILE},
    {"console",      CFG_SRC_CONSOLE},
    {"syslog",       CFG_SRC_SYSLOG},
    {"metric",       CFG_SRC_METRIC},
    {"http",         CFG_SRC_HTTP},
    {"net",          CFG_SRC_NET},
    {"fs",           CFG_SRC_FS},
    {"dns",          CFG_SRC_DNS},
    {NULL,                    -1}
};

static const char*
valToStr(enum_map_t map[], unsigned val)
{
    enum_map_t* m;
    for (m=map; m->str; m++) {
        if (val == m->val) return m->str;
    }
    return NULL;
}

typedef struct {
    int valid;
    regex_t re;
} local_re_t;

struct _evt_fmt_t
{
    local_re_t value_re[CFG_SRC_MAX];
    local_re_t field_re[CFG_SRC_MAX];
    local_re_t name_re[CFG_SRC_MAX];
    unsigned enabled[CFG_SRC_MAX];

    struct {
        time_t time;
        unsigned long evtCount;
        int notified;
    } ratelimit;
};

static const char* valueFilterDefault[] = {
    DEFAULT_SRC_FILE_VALUE,
    DEFAULT_SRC_CONSOLE_VALUE,
    DEFAULT_SRC_SYSLOG_VALUE,
    DEFAULT_SRC_METRIC_VALUE,
    DEFAULT_SRC_HTTP_VALUE,
    DEFAULT_SRC_NET_VALUE,
    DEFAULT_SRC_FS_VALUE,
    DEFAULT_SRC_DNS_VALUE,
};

static const char* fieldFilterDefault[] = {
    DEFAULT_SRC_FILE_FIELD,
    DEFAULT_SRC_CONSOLE_FIELD,
    DEFAULT_SRC_SYSLOG_FIELD,
    DEFAULT_SRC_METRIC_FIELD,
    DEFAULT_SRC_HTTP_FIELD,
    DEFAULT_SRC_NET_FIELD,
    DEFAULT_SRC_FS_FIELD,
    DEFAULT_SRC_DNS_FIELD,
};

static const char* nameFilterDefault[] = {
    DEFAULT_SRC_FILE_NAME,
    DEFAULT_SRC_CONSOLE_NAME,
    DEFAULT_SRC_SYSLOG_NAME,
    DEFAULT_SRC_METRIC_NAME,
    DEFAULT_SRC_HTTP_NAME,
    DEFAULT_SRC_NET_NAME,
    DEFAULT_SRC_FS_NAME,
    DEFAULT_SRC_DNS_NAME,
};

static unsigned srcEnabledDefault[] = {
    DEFAULT_SRC_FILE,
    DEFAULT_SRC_CONSOLE,
    DEFAULT_SRC_SYSLOG,
    DEFAULT_SRC_METRIC,
    DEFAULT_SRC_HTTP,
    DEFAULT_SRC_NET,
    DEFAULT_SRC_FS,
    DEFAULT_SRC_DNS,
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

evt_fmt_t*
evtFormatCreate()
{
    evt_fmt_t* evt = calloc(1, sizeof(evt_fmt_t));
    if (!evt) {
        DBG(NULL);
        return NULL;
    }

    watch_t src;
    for (src=CFG_SRC_FILE;  src<CFG_SRC_MAX; src++) {
        filterSet(&evt->value_re[src], NULL, valueFilterDefault[src]);
        filterSet(&evt->field_re[src], NULL, fieldFilterDefault[src]);
        filterSet(&evt->name_re[src], NULL, nameFilterDefault[src]);
        evt->enabled[src] = srcEnabledDefault[src];
    }

    return evt;
}

void
evtFormatDestroy(evt_fmt_t** evt)
{
    if (!evt || !*evt) return;
    evt_fmt_t *edestroy  = *evt;

    watch_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (edestroy->value_re[src].valid) regfree(&edestroy->value_re[src].re);
        if (edestroy->field_re[src].valid) regfree(&edestroy->field_re[src].re);
        if (edestroy->name_re[src].valid) regfree(&edestroy->name_re[src].re);
    }

    free(edestroy);
    *evt = NULL;
}

regex_t *
evtFormatValueFilter(evt_fmt_t *evt, watch_t src)
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
evtFormatFieldFilter(evt_fmt_t *evt, watch_t src)
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
evtFormatNameFilter(evt_fmt_t *evt, watch_t src)
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
evtFormatSourceEnabled(evt_fmt_t *evt, watch_t src)
{
    if (src < CFG_SRC_MAX) {

        if (evt) return evt->enabled[src];
        return srcEnabledDefault[src];
    }

    DBG("%d", src);
    return srcEnabledDefault[CFG_SRC_FILE];
}

void
evtFormatValueFilterSet(evt_fmt_t *evt, watch_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->value_re[src], str, valueFilterDefault[src]);
}

void
evtFormatFieldFilterSet(evt_fmt_t *evt, watch_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->field_re[src], str, fieldFilterDefault[src]);
}

void
evtFormatNameFilterSet(evt_fmt_t *evt, watch_t src, const char *str)
{
    if (!evt || src >= CFG_SRC_MAX) return;
    filterSet(&evt->name_re[src], str, nameFilterDefault[src]);
}

void
evtFormatSourceEnabledSet(evt_fmt_t* evt, watch_t src, unsigned val)
{
    if (!evt || src >= CFG_SRC_MAX || val > 1) return;
    evt->enabled[src] = val;
}

#define MATCH_FOUND 1
#define NO_MATCH_FOUND 0

static int
anyValueFieldMatches(regex_t* filter, event_t* metric)
{
    if (!filter || !metric) return MATCH_FOUND;

    // Test the value of metric
    char valbuf[320]; // Seems crazy but -MAX_DBL.00 is 313 chars!
    valbuf[0]='\0';
    switch ( metric->value.type ) {
        case FMT_INT:
            snprintf(valbuf, sizeof(valbuf), "%lld", metric->value.integer);
            break;
        case FMT_FLT:
            snprintf(valbuf, sizeof(valbuf), "%.2f", metric->value.floating);
            break;
        default:
            DBG(NULL);
    }
    if (valbuf[0]) {
        if (!regexec_wrapper(filter, valbuf, 0, NULL, 0)) return MATCH_FOUND;
    }

    // Handle the case where there are no fields...
    if (!metric->fields) return NO_MATCH_FOUND;

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

        if (str && !regexec_wrapper(filter, str, 0, NULL, 0)) return MATCH_FOUND;
    }

    return NO_MATCH_FOUND;
}

cJSON *
fmtEventJson(event_format_t *sev)
{
    char numbuf[32];

    if (!sev || !sev->proc) return NULL;

    cJSON* json = cJSON_CreateObject();
    if (!json) goto err;

    if (!cJSON_AddStringToObjLN(json, SOURCETYPE, valToStr(watchTypeMap, sev->sourcetype))) goto err;

    if (!cJSON_AddStringToObjLN(json, ID, sev->proc->id)) goto err;

    if (!cJSON_AddNumberToObjLN(json, TIME, sev->timestamp)) goto err;
    if (!cJSON_AddStringToObjLN(json, SOURCE, sev->src)) goto err;
    if (!cJSON_AddStringToObjLN(json, HOST, sev->proc->hostname)) goto err;
    if (!cJSON_AddStringToObjLN(json, PROCNAME, sev->proc->procname)) goto err;
    if (!cJSON_AddStringToObjLN(json, CMDNAME, sev->proc->cmd)) goto err;
    if (!cJSON_AddNumberToObjLN(json, PID, sev->proc->pid)) goto err;
    if (snprintf(numbuf, sizeof(numbuf), "%llu", sev->uid) < 0) goto err;
    if (!cJSON_AddStringToObjLN(json, CHANNEL, numbuf)) goto err;
    cJSON_AddItemToObjectCS(json, DATA, sev->data);

    return json;
err:
    DBG("time=%s src=%s data=%p host=%s channel=%s json=%p",
            sev->timestamp, sev->src, sev->data, sev->proc->hostname, numbuf, json);
    if (json) cJSON_Delete(json);

    return NULL;
}

cJSON *
rateLimitMessage(proc_id_t *proc, watch_t src)
{
    event_format_t event;

    struct timeb tb;
    ftime(&tb);
    event.timestamp = tb.time + (double)tb.millitm/1000;
    event.src = "notice";
    event.proc = proc;
    event.uid = 0ULL;

    char string[128];
    if (snprintf(string, sizeof(string), "Truncated metrics. Your rate exceeded %d metrics per second", MAXEVENTSPERSEC) == -1) {
        return NULL;
    }
    event.data = cJSON_CreateString(string);
    event.sourcetype = src;

    cJSON* json = fmtEventJson(&event);
    return json;
}

static const char*
metricTypeStr(data_type_t type)
{
    switch (type) {
        case DELTA:
            return "counter";
        case CURRENT:
            return "gauge";
        case DELTA_MS:
            return "timer";
        case HISTOGRAM:
            return "histogram";
        case SET:
            return "set";
        default:
            return "unknown";
    }
}

static int
addJsonFields(event_field_t* fields, regex_t* fieldFilter, cJSON* json)
{
    if (!fields) return TRUE;

    event_field_t *fld;

    // Start adding key:value entries
    for (fld = fields; fld->value_type != FMT_END; fld++) {

        // skip outputting anything that doesn't match fieldFilter
        if (fieldFilter && regexec_wrapper(fieldFilter, fld->name, 0, NULL, 0)) continue;

        // skip if this field is not used in events
        if (fld->event_usage == FALSE) continue;

        if (fld->value_type == FMT_STR) {
            if (!cJSON_AddStringToObjLN(json, fld->name, fld->value.str)) return FALSE;
        } else if (fld->value_type == FMT_NUM) {
            if (!cJSON_AddNumberToObjLN(json, fld->name, fld->value.num)) return FALSE;
        } else {
            DBG("bad field type");
        }
    }

    return TRUE;
}

cJSON *
fmtMetricJson(event_t *metric, regex_t *fieldFilter, watch_t src)
{
    const char* metric_type = NULL;

    if (!metric) return NULL;

    cJSON *json = cJSON_CreateObject();
    if (!json) goto err;

    if (src == CFG_SRC_METRIC) {
        if (!cJSON_AddStringToObjLN(json, "_metric", metric->name)) goto err;
        metric_type = metricTypeStr(metric->type);
        if (!cJSON_AddStringToObjLN(json, "_metric_type", metric_type)) goto err;
        switch ( metric->value.type ) {
            case FMT_INT:
                if (!cJSON_AddNumberToObjLN(json, "_value", metric->value.integer)) goto err;
                break;
            case FMT_FLT:
                if (!cJSON_AddNumberToObjLN(json, "_value", metric->value.floating)) goto err;
                break;
            default:
                DBG(NULL);
        }
    }

    // Add fields
    if (!addJsonFields(metric->fields, fieldFilter, json)) goto err;
    return json;

err:
    DBG("_metric=%s _metric_type=%s _value=%lld, fields=%p, json=%p",
        metric->name, metric_type, metric->value, metric->fields, json);
    if (json) cJSON_Delete(json);

    return NULL;
}

static cJSON *
evtFSEvent(evt_fmt_t *evt, event_t *metric, proc_id_t *proc)
{
    event_format_t event;
    struct timeb tb;
    regex_t *filter;

    if (!evt || !metric || !proc) return NULL;

    // Test for a name field match.  No match, no metric output
    if (!evtFormatSourceEnabled(evt, metric->src) ||
        !(filter = evtFormatNameFilter(evt, metric->src))) { //||
        //(regexec_wrapper(filter, metric->name, 0, NULL, 0))) {
        return NULL;
    }

    /*
     * Loop through all metric fields for at least one matching field value
     * No match, no metric output
     */
    if (!anyValueFieldMatches(evtFormatValueFilter(evt, metric->src), metric)) {
        return NULL;
    }

    ftime(&tb);
    event.timestamp = tb.time + (double)tb.millitm/1000;
    event.src = metric->name;
    event.sourcetype = metric->src;

    // Format the metric string using the configured metric format type
    event.data = fmtMetricJson(metric, evtFormatFieldFilter(evt, metric->src), metric->src);
    if (!event.data) return NULL;

    cJSON *json = cJSON_CreateObject();
    if (!json) FMTERR(event, json);

    if (!cJSON_AddStringToObjLN(json, SOURCETYPE,
                                valToStr(watchTypeMap, event.sourcetype))) FMTERR(event, json);
    if (!cJSON_AddStringToObjLN(json, SOURCE, event.src)) FMTERR(event, json);
    if (!cJSON_AddStringToObjLN(json, CMDNAME, proc->cmd)) FMTERR(event, json);
    if (!cJSON_AddNumberToObjLN(json, PID, proc->pid)) FMTERR(event, json);
    if (!cJSON_AddStringToObjLN(json, HOST, proc->hostname)) FMTERR(event, json);
    if (!cJSON_AddNumberToObjLN(json, TIME, event.timestamp)) FMTERR(event, json);;
    cJSON_AddItemToObjectCS(json, DATA, event.data);

    return json;
}

static cJSON *
evtFormatHelper(evt_fmt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc, watch_t src)
{
    event_format_t event;
    struct timeb tb;
    time_t now;
    
    regex_t *filter;

    if (!evt || !metric || !proc) return NULL;

    // Test for a name field match.  No match, no metric output
    if (!evtFormatSourceEnabled(evt, src) ||
        !(filter = evtFormatNameFilter(evt, src)) ||
        (regexec_wrapper(filter, metric->name, 0, NULL, 0))) {
        return NULL;
    }

    // rate limited to MAXEVENTSPERSEC
    if (time(&now) != evt->ratelimit.time) {
        evt->ratelimit.time = now;
        evt->ratelimit.evtCount = evt->ratelimit.notified = 0;
    } else if (++evt->ratelimit.evtCount >= MAXEVENTSPERSEC) {
        // one notice per truncate
        if (evt->ratelimit.notified == 0) {
            cJSON* notice = rateLimitMessage(proc, src);
            evt->ratelimit.notified = (notice)?1:0;
            return notice;
        }
    }

    /*
     * Loop through all metric fields for at least one matching field value
     * No match, no metric output
     */
    if (!anyValueFieldMatches(evtFormatValueFilter(evt, src), metric)) {
        return NULL;
    }

    ftime(&tb);
    event.timestamp = tb.time + (double)tb.millitm/1000;
    event.src = metric->name;
    event.proc = proc;
    event.uid = uid;
    event.sourcetype = src;

    // Format the metric string using the configured metric format type
    event.data = fmtMetricJson(metric, evtFormatFieldFilter(evt, src), src);
    if (!event.data) return NULL;

    return fmtEventJson(&event);
}

cJSON *
evtFormatMetric(evt_fmt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc)
{
    switch (metric->src) {
        case CFG_SRC_FS:
        case CFG_SRC_NET:
            return evtFSEvent(evt, metric, proc);
            break;
        default:
            return evtFormatHelper(evt, metric, uid, proc, metric->src);
    }
}

cJSON *
evtFormatHttp(evt_fmt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc)
{
    return evtFormatHelper(evt, metric, uid, proc, CFG_SRC_HTTP);
}

cJSON *
evtFormatLog(evt_fmt_t *evt, const char *path, const void *buf, size_t count,
       uint64_t uid, proc_id_t* proc)
{
    event_format_t event;
    struct timeb tb;
    watch_t logType;

    if (!evt || !path || !buf || !proc) return NULL;

    regex_t* filter;
    if (evtFormatSourceEnabled(evt, CFG_SRC_CONSOLE) &&
       (filter = evtFormatNameFilter(evt, CFG_SRC_CONSOLE)) &&
       (!regexec_wrapper(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_CONSOLE;
    } else if (evtFormatSourceEnabled(evt, CFG_SRC_FILE) &&
       (filter = evtFormatNameFilter(evt, CFG_SRC_FILE)) &&
       (!regexec_wrapper(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_FILE;
    } else {
        return NULL;
    }

    ftime(&tb);
    event.timestamp = tb.time + (double)tb.millitm/1000;
    event.src = path;
    event.proc = proc;
    event.uid = uid;
    event.data = cJSON_CreateStringFromBuffer(buf, count);
    if (!event.data) return NULL;
    event.sourcetype = logType;

    cJSON * json = fmtEventJson(&event);
    if (!json) return NULL;


    cJSON* dataField = cJSON_GetObjectItem(json, "data");
    if (dataField && dataField->valuestring) {
        filter = evtFormatValueFilter(evt, logType);
        if (filter && regexec_wrapper(filter, dataField->valuestring, 0, NULL, 0)) {
            // This event doesn't match.  Drop it on the floor.
            cJSON_Delete(json);
            return NULL;
        }
    }

    return json;
}
