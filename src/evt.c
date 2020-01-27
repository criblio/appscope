#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/timeb.h>
#include <time.h>

#include "dbg.h"
#include "evt.h"

typedef struct {
    int valid;
    regex_t re;
} local_re_t;

struct _evt_t
{
    format_t* format;
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

    // default format for events, which can be overriden
    evt->format = fmtCreate(CFG_EVENT_ND_JSON);

    return evt;
}

void
evtDestroy(evt_t** evt)
{
    if (!evt || !*evt) return;
    evt_t *edestroy  = *evt;

    cfg_evt_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (edestroy->value_re[src].valid) regfree(&edestroy->value_re[src].re);
        if (edestroy->field_re[src].valid) regfree(&edestroy->field_re[src].re);
        if (edestroy->name_re[src].valid) regfree(&edestroy->name_re[src].re);
    }

    fmtDestroy(&edestroy->format);

    free(edestroy);
    *evt = NULL;
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
        if (!regexec(filter, valbuf, 0, NULL, 0)) return MATCH_FOUND;
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

        if (str && !regexec(filter, str, 0, NULL, 0)) return MATCH_FOUND;
    }

    return NO_MATCH_FOUND;
}

cJSON *
rateLimitMessage(proc_id_t *proc)
{
    event_format_t event;

    struct timeb tb;
    ftime(&tb);
    event.timestamp = tb.time + tb.millitm/1000;
    event.src = "notice";
    event.proc = proc;
    event.uid = 0ULL;

    char string[128];
    if (snprintf(string, sizeof(string), "Truncated metrics. Your rate exceeded %d metrics per second", MAXEVENTSPERSEC) == -1) {
        return NULL;
    }
    event.data = cJSON_CreateString(string);
    event.sourcetype = CFG_SRC_METRIC;

    cJSON* json = fmtEventJson(&event);
    return json;
}

cJSON *
evtMetric(evt_t *evt, event_t* metric, uint64_t uid, proc_id_t* proc)
{
    event_format_t event;
    struct timeb tb;
    time_t now;
    
    regex_t *filter;

    if (!evt || !metric || !proc) return NULL;

    // Test for a name field match.  No match, no metric output
    if (!evtSourceEnabled(evt, CFG_SRC_METRIC) ||
        !(filter = evtNameFilter(evt, CFG_SRC_METRIC)) ||
        (regexec(filter, metric->name, 0, NULL, 0))) {
        return NULL;
    }

    // rate limited to MAXEVENTSPERSEC
    if (time(&now) != evt->ratelimit.time) {
        evt->ratelimit.time = now;
        evt->ratelimit.evtCount = evt->ratelimit.notified = 0;
    } else if (++evt->ratelimit.evtCount >= MAXEVENTSPERSEC) {
        // one notice per truncate
        if (evt->ratelimit.notified == 0) {
            cJSON* notice = rateLimitMessage(proc);
            evt->ratelimit.notified = (notice)?1:0;
            return notice;
        }
    }

    /*
     * Loop through all metric fields for at least one matching field value
     * No match, no metric output
     */
    if (!anyValueFieldMatches(evtValueFilter(evt, CFG_SRC_METRIC), metric)) {
        return NULL;
    }

    ftime(&tb);
    event.timestamp = tb.time + tb.millitm/1000;
    event.src = metric->name;
    event.proc = proc;
    event.uid = uid;

    // Format the metric string using the configured metric format type
    event.data = fmtMetricJson(metric, evtFieldFilter(evt, CFG_SRC_METRIC));
    if (!event.data) return NULL;
    event.sourcetype = CFG_SRC_METRIC;

    return fmtEventJson(&event);
}

cJSON *
evtLog(evt_t *evt, const char *path, const void *buf, size_t count,
       uint64_t uid, proc_id_t* proc)
{
    event_format_t event;
    struct timeb tb;
    cfg_evt_t logType;

    if (!evt || !path || !buf || !proc) return NULL;

    regex_t* filter;
    if (evtSourceEnabled(evt, CFG_SRC_CONSOLE) &&
       (filter = evtNameFilter(evt, CFG_SRC_CONSOLE)) &&
       (!regexec(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_CONSOLE;
    } else if (evtSourceEnabled(evt, CFG_SRC_FILE) &&
       (filter = evtNameFilter(evt, CFG_SRC_FILE)) &&
       (!regexec(filter, path, 0, NULL, 0))) {
        logType = CFG_SRC_FILE;
    } else {
        return NULL;
    }

    ftime(&tb);
    event.timestamp = tb.time + tb.millitm/1000;
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
        filter = evtValueFilter(evt, logType);
        if (filter && regexec(filter, dataField->valuestring, 0, NULL, 0)) {
            // This event doesn't match.  Drop it on the floor.
            cJSON_Delete(json);
            return NULL;
        }
    }

    return json;
}
