#define _GNU_SOURCE
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "dbg.h"
#include "evtformat.h"
#include "strset.h"
#include "com.h"
#include "scopestdlib.h"


// This is ugly, but...
// In glibc 2.27 (ubuntu 18.04) and earlier clock_gettime
// resolved to 'clock_gettime@@GLIBC_2.2.5'.
// In glibc 2.31 (ubuntu 20.04), it's now 'clock_gettime@@GLIBC_2.17'.
// This voodoo avoids creating binaries which would require a glibc
// that's 2.31 or newer, even if built on new ubuntu 20.04.
// This is tested by test/glibcvertest.c.
//#ifdef __linux__
//__asm__(".symver clock_gettime,clock_gettime@GLIBC_2.2.5");
//#endif


// key names for event JSON
#define HOST "host"
#define TIME "_time"
#define DATA "data"
#define SOURCE "source"
#define SOURCETYPE "sourcetype"
#define EVENT "ev"
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
    const char *str;
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

static const char *
valToStr(enum_map_t map[], unsigned val)
{
    enum_map_t *m;
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
        // runtime params
        time_t time;
        unsigned long evtCount;
        int notified;
        // configured param
        unsigned long maxEvtPerSec;
    } ratelimit;

    custom_tag_t** tags;
};

static const char *valueFilterDefault[] = {
    DEFAULT_SRC_FILE_VALUE,
    DEFAULT_SRC_CONSOLE_VALUE,
    DEFAULT_SRC_SYSLOG_VALUE,
    DEFAULT_SRC_METRIC_VALUE,
    DEFAULT_SRC_HTTP_VALUE,
    DEFAULT_SRC_NET_VALUE,
    DEFAULT_SRC_FS_VALUE,
    DEFAULT_SRC_DNS_VALUE,
};

static const char *fieldFilterDefault[] = {
    DEFAULT_SRC_FILE_FIELD,
    DEFAULT_SRC_CONSOLE_FIELD,
    DEFAULT_SRC_SYSLOG_FIELD,
    DEFAULT_SRC_METRIC_FIELD,
    DEFAULT_SRC_HTTP_FIELD,
    DEFAULT_SRC_NET_FIELD,
    DEFAULT_SRC_FS_FIELD,
    DEFAULT_SRC_DNS_FIELD,
};

static const char *nameFilterDefault[] = {
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
filterSet(local_re_t *re, const char *str, const char *default_val)
{
    if (!re) return;

    local_re_t temp;
    temp.valid = str && !regcomp(&temp.re, str, REG_EXTENDED | REG_NOSUB);
    if (!temp.valid && default_val) {
        // regcomp failed on str.  Try the default.
        temp.valid = !regcomp(&temp.re, default_val, REG_EXTENDED | REG_NOSUB);
    }

    if (temp.valid) {
        // Out with the old
        if (re->valid) regfree(&re->re);
        // In with the new
        *re = temp;
    } else {
        if (default_val) DBG("%s", str);
    }
}

evt_fmt_t *
evtFormatCreate()
{
    evt_fmt_t *evt = scope_calloc(1, sizeof(evt_fmt_t));
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

    evt->ratelimit.maxEvtPerSec = DEFAULT_MAXEVENTSPERSEC;

    evt->tags = DEFAULT_CUSTOM_TAGS;

    return evt;
}

static
void
evtFormatDestroyTags(custom_tag_t*** tags)
{
    if (!tags || !*tags) return;
    custom_tag_t** t = *tags;
    int i = 0;
    while (t[i]) {
        scope_free(t[i]->name);
        scope_free(t[i]->value);
        scope_free(t[i]);
        i++;
    }
    scope_free(t);
    *tags = NULL;
}

void
evtFormatDestroy(evt_fmt_t **evt)
{
    if (!evt || !*evt) return;
    evt_fmt_t *edestroy  = *evt;

    watch_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (edestroy->value_re[src].valid) regfree(&edestroy->value_re[src].re);
        if (edestroy->field_re[src].valid) regfree(&edestroy->field_re[src].re);
        if (edestroy->name_re[src].valid) regfree(&edestroy->name_re[src].re);
    }

    evtFormatDestroyTags(&edestroy->tags);

    scope_free(edestroy);
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

unsigned
evtFormatRateLimit(evt_fmt_t *evt)
{
    return (evt) ? evt->ratelimit.maxEvtPerSec : DEFAULT_MAXEVENTSPERSEC;
}

custom_tag_t**
evtFormatCustomTags(evt_fmt_t* fmt)
{
    return (fmt) ? fmt->tags : DEFAULT_CUSTOM_TAGS;
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
evtFormatSourceEnabledSet(evt_fmt_t *evt, watch_t src, unsigned val)
{
    if (!evt || src >= CFG_SRC_MAX || val > 1) return;
    evt->enabled[src] = val;
}

void
evtFormatRateLimitSet(evt_fmt_t *evt, unsigned val)
{
    if (!evt) return;
    evt->ratelimit.maxEvtPerSec = val;
}

void
evtFormatCustomTagsSet(evt_fmt_t* fmt, custom_tag_t** tags)
{
    if (!fmt) return;

    // Don't leak with multiple set operations
    evtFormatDestroyTags(&fmt->tags);

    if (!tags || !*tags) return;

    // get a count of how big to scope_calloc
    int num = 0;
    while(tags[num]) num++;

    fmt->tags = scope_calloc(1, sizeof(custom_tag_t*) * (num+1));
    if (!fmt->tags) {
        DBG(NULL);
        return;
    }

    int i, j = 0;
    for (i = 0; i<num; i++) {
        custom_tag_t* t = scope_calloc(1, sizeof(custom_tag_t));
        char* n = scope_strdup(tags[i]->name);
        char* v = scope_strdup(tags[i]->value);
        if (!t || !n || !v) {
            DBG("t=%p n=%p v=%p", t, n, v);
            if (t) scope_free (t);
            if (n) scope_free (n);
            if (v) scope_free (v);
            continue;
        }
        t->name = n;
        t->value = v;
        fmt->tags[j++]=t;
    }
}

#define MATCH_FOUND 1
#define NO_MATCH_FOUND 0

static int
anyValueFieldMatches(regex_t *filter, event_t *metric)
{
    if (!filter || !metric) return MATCH_FOUND;

    // Test the value of metric
    char valbuf[320]; // Seems crazy but -MAX_DBL.00 is 313 chars!
    valbuf[0]='\0';
    switch ( metric->value.type ) {
        case FMT_INT:
            scope_snprintf(valbuf, sizeof(valbuf), "%lld", metric->value.integer);
            break;
        case FMT_FLT:
            scope_snprintf(valbuf, sizeof(valbuf), "%.2f", metric->value.floating);
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
            if (scope_snprintf(valbuf, sizeof(valbuf), "%lld", fld->value.num) > 0) {
                str = valbuf;
            }
        }

        if (str && !regexec_wrapper(filter, str, 0, NULL, 0)) return MATCH_FOUND;
    }

    return NO_MATCH_FOUND;
}

static void
addCustomJsonFields(custom_tag_t **tags, cJSON *json, strset_t *addedFields)
{
    if (!json || !tags) return;

    custom_tag_t *tag;
    int i = 0;
    while ((tag = tags[i++])) {

        // Don't allow duplicate field names if addedFields is non-null
        if (addedFields && !strSetAdd(addedFields, tag->name)) continue;
        cJSON_AddStringToObject(json, tag->name, tag->value);
    }
}

cJSON *
fmtEventJson(evt_fmt_t *efmt, event_format_t *sev)
{
    if (!sev || !sev->proc) return NULL;

    cJSON *json = cJSON_CreateObject();
    if (!json) goto err;

    if (!cJSON_AddStringToObjLN(json, SOURCETYPE, valToStr(watchTypeMap, sev->sourcetype))) goto err;

    if (!cJSON_AddNumberToObjLN(json, TIME, sev->timestamp)) goto err;
    if (!cJSON_AddStringToObjLN(json, SOURCE, sev->src)) goto err;
    if (!cJSON_AddStringToObjLN(json, HOST, sev->proc->hostname)) goto err;
    if (!cJSON_AddStringToObjLN(json, PROCNAME, sev->proc->procname)) goto err;
    if (!cJSON_AddStringToObjLN(json, CMDNAME, sev->proc->cmd)) goto err;
    if (!cJSON_AddNumberToObjLN(json, PID, sev->proc->pid)) goto err;
    
    if (efmt) {
        addCustomJsonFields(evtFormatCustomTags(efmt), json, NULL);
    }
    cJSON_AddItemToObjectCS(json, DATA, sev->data);

    return json;
err:
    DBG("time=%s src=%s data=%p host=%s json=%p",
            sev->timestamp, sev->src, sev->data, sev->proc->hostname, json);
    if (json) cJSON_Delete(json);
    if (sev->data) cJSON_Delete(sev->data);

    return NULL;
}

cJSON *
rateLimitMessage(proc_id_t *proc, watch_t src, unsigned maxEvtPerSec)
{
    event_format_t event;

    struct timeval tv;
    scope_gettimeofday(&tv, NULL);

    event.timestamp = tv.tv_sec + tv.tv_usec/1e6;
    event.src = "notice";
    event.proc = proc;
    event.uid = 0ULL;

    char string[128];
    if (scope_snprintf(string, sizeof(string), "Truncated metrics. Your rate exceeded %u metrics per second", maxEvtPerSec) == -1) {
        return NULL;
    }
    event.data = cJSON_CreateString(string);
    event.sourcetype = src;

    cJSON *json = fmtEventJson(NULL, &event);
    return json;
}

static const char *
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
addJsonFields(event_field_t *fields, regex_t *fieldFilter, cJSON *json, strset_t *addedFields)
{
    if (!fields) return TRUE;

    event_field_t *fld;

    // Start adding key:value entries
    for (fld = fields; fld->value_type != FMT_END; fld++) {

        // skip outputting anything that doesn't match fieldFilter
        if (fieldFilter && regexec_wrapper(fieldFilter, fld->name, 0, NULL, 0)) continue;

        // skip if this field is not used in events
        if (fld->event_usage == FALSE) continue;

        // Don't allow duplicate field names
        if (!strSetAdd(addedFields, fld->name)) continue;

        if (fld->value_type == FMT_STR) {
            if (!cJSON_AddStringToObjLN(json, fld->name, fld->value.str)) continue;
        } else if (fld->value_type == FMT_NUM) {
            if (!cJSON_AddNumberToObjLN(json, fld->name, fld->value.num)) continue;
        } else {
            DBG("bad field type");
        }
    }

    return TRUE;
}

cJSON *
fmtMetricJson(event_t *metric, regex_t *fieldFilter, watch_t src, custom_tag_t **tags)
{
    const char *metric_type = NULL;
    strset_t *addedFields = NULL;

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

    // addedFields lets us avoid duplicate field names.  If we go to
    // add one that's already in the set, skip it.  In this way precidence
    // is given to capturedFields then custom fields then remaining fields.
    addedFields = strSetCreate(DEFAULT_SET_SIZE);
    if (!addJsonFields(metric->capturedFields, fieldFilter, json, addedFields)) goto err;
    addCustomJsonFields(tags, json, addedFields);
    if (!addJsonFields(metric->fields, fieldFilter, json, addedFields)) goto err;
    strSetDestroy(&addedFields);

    return json;

err:
    DBG("_metric=%s _metric_type=%s _value=%lld, fields=%p, json=%p",
        metric->name, metric_type, metric->value, metric->fields, json);
    if (addedFields) strSetDestroy(&addedFields);
    if (json) cJSON_Delete(json);

    return NULL;
}

static cJSON *
evtFormatHelper(evt_fmt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc, watch_t src)
{
    event_format_t event;

    struct timeval tv;
    scope_gettimeofday(&tv, NULL);
    
    regex_t *filter;

    if (!evt || !metric || !proc) return NULL;

    // Test for a name field match.  No match, no metric output
    if (!evtFormatSourceEnabled(evt, src) ||
        !(filter = evtFormatNameFilter(evt, src)) ||
        (regexec_wrapper(filter, metric->name, 0, NULL, 0))) {
        return NULL;
    }

    // rate limited to maxEvtPerSec
    if (evt->ratelimit.maxEvtPerSec == 0) {
        ; // no rate limiting.
    } else if (tv.tv_sec != evt->ratelimit.time) {
        evt->ratelimit.time = tv.tv_sec;
        evt->ratelimit.evtCount = evt->ratelimit.notified = 0;
    } else if (++evt->ratelimit.evtCount >= evt->ratelimit.maxEvtPerSec) {
        // one notice per truncate
        if (evt->ratelimit.notified == 0) {
            cJSON *notice = rateLimitMessage(proc, src, evt->ratelimit.maxEvtPerSec);
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

    event.timestamp = tv.tv_sec + tv.tv_usec/1e6;
    event.src = metric->name;
    event.proc = proc;
    event.uid = uid;
    event.sourcetype = src;

    // Format the metric string using the configured metric format type
    if (!metric->data) {
        event.data = fmtMetricJson(metric, evtFormatFieldFilter(evt, src), src, NULL);
    } else {
        event.data = metric->data;
    }

    if (!event.data) return NULL;

    return fmtEventJson(evt, &event);
}

cJSON *
evtFormatMetric(evt_fmt_t *efmt, event_t *metric, uint64_t uid, proc_id_t *proc)
{
    return evtFormatHelper(efmt, metric, uid, proc, metric->src);
}

cJSON *
evtFormatHttp(evt_fmt_t *evt, event_t *metric, uint64_t uid, proc_id_t *proc)
{
    return evtFormatHelper(evt, metric, uid, proc, CFG_SRC_HTTP);
}
