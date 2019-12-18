#define _GNU_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "cJSON.h"
#include "dbg.h"
#include "format.h"
#include "scopetypes.h"

// key names for event JSON
#define HOST "host"
#define TIME "_time"
#define DATA "_raw"
#define SOURCE "source"
#define CHANNEL "_channel"
#define TYPE "ty"
#define EVENT "ev"
#define ID "id"

#define TRUE 1
#define FALSE 0


struct _format_t
{
    cfg_out_format_t format;
    struct {
        char* prefix;
        unsigned max_len;       // Max length in bytes of a statsd string
    } statsd;
    unsigned verbosity;
    custom_tag_t** tags;
};


// Constructors Destructors
format_t*
fmtCreate(cfg_out_format_t format)
{
    if (format >= CFG_FORMAT_MAX) return NULL;

    format_t* f = calloc(1, sizeof(format_t));
    if (!f) {
        DBG(NULL);
        return NULL;
    }
    f->format = format;
    f->statsd.prefix = (DEFAULT_STATSD_PREFIX) ? strdup(DEFAULT_STATSD_PREFIX) : NULL;
    f->statsd.max_len = DEFAULT_STATSD_MAX_LEN;
    f->verbosity = DEFAULT_OUT_VERBOSITY;
    f->tags = DEFAULT_CUSTOM_TAGS;

    return f;
}

static
void
fmtDestroyTags(custom_tag_t*** tags)
{
    if (!tags || !*tags) return;
    custom_tag_t** t = *tags;
    int i = 0;
    while (t[i]) {
        free(t[i]->name);
        free(t[i]->value);
        free(t[i]);
        i++;
    }
    free(t);
    *tags = NULL;
}

void
fmtDestroy(format_t** fmt)
{
    if (!fmt || !*fmt) return;
    format_t* f = *fmt;
    if (f->statsd.prefix) free(f->statsd.prefix);
    fmtDestroyTags(&f->tags);
    free(f);
    *fmt = NULL;
}

static char*
statsdType(data_type_t x)
{
    switch (x) {
        case DELTA:
            return "c";
        case CURRENT:
            return "g";
        case DELTA_MS:
            return "ms";
        case HISTOGRAM:
            return "h";
        case SET:
            return "s";
        default:
            DBG("%d", x);
            return "";
    }
}

static int
createStatsFieldString(format_t* fmt, event_field_t* f, char* tag, int sizeoftag)
{
    if (!fmt || !f || !tag || sizeoftag <= 0) return -1;

    int sz;

    switch (f->value_type) {
        case FMT_NUM:
            sz = snprintf(tag, sizeoftag, "%s:%lli", f->name, f->value.num);
            break;
        case FMT_STR:
            sz = snprintf(tag, sizeoftag, "%s:%s", f->name, f->value.str);
            break;
        default:
            DBG("%d %s", f->value_type, f->name);
            sz = -1;
    }
    return sz;
}

static void
appendStatsdFieldString(format_t* fmt, char* tag, int sz, char** end, int* bytes, int* firstTagAdded)
{
    if (!*firstTagAdded) {
        sz += 2; // add space for the |#
        if ((*bytes + sz) >= fmt->statsd.max_len) return;
        *end = stpcpy(*end, "|#");
        *end = stpcpy(*end, tag);
        strcpy(*end, "\n"); // add newline, but don't advance end
        *firstTagAdded = 1;
    } else {
        sz += 1; // add space for the comma
        if ((*bytes + sz) >= fmt->statsd.max_len) return;
        *end = stpcpy(*end, ",");
        *end = stpcpy(*end, tag);
        strcpy(*end, "\n"); // add newline, but don't advance end
    }
    *bytes += sz;
}


static void
addStatsdFields(format_t* fmt, event_field_t* fields, char** end, int* bytes, int* firstTagAdded, regex_t* fieldFilter)
{
    if (!fmt || !fields || ! end || !*end || !bytes) return;

    char tag[fmt->statsd.max_len+1];
    tag[fmt->statsd.max_len] = '\0'; // Ensures null termination
    int sz;

    event_field_t* f;
    for (f = fields; f->value_type != FMT_END; f++) {

        if (fieldFilter && regexec(fieldFilter, f->name, 0, NULL, 0)) continue;

        // Honor Verbosity
        if (f->cardinality > fmt->verbosity) continue;

        sz = createStatsFieldString(fmt, f, tag, sizeof(tag));
        if (sz < 0) break;

        appendStatsdFieldString(fmt, tag, sz, end, bytes, firstTagAdded);
    }
}

static void
addCustomFields(format_t* fmt, custom_tag_t** tags, char** end, int* bytes, int* firstTagAdded)
{
    if (!fmt || !tags || !*tags || !end || !*end || !bytes) return;

    char tag[fmt->statsd.max_len+1];
    tag[fmt->statsd.max_len] = '\0'; // Ensures null termination
    int sz;

    custom_tag_t* t;
    int i = 0;
    while ((t = tags[i++])) {

        // No verbosity setting exists for custom fields.

        sz = snprintf(tag, sizeof(tag), "%s:%s", t->name, t->value);
        if (sz < 0) break;

        appendStatsdFieldString(fmt, tag, sz, end, bytes, firstTagAdded);
    }
}

// Accessors
static char *
fmtEventNdJson(format_t *fmt, event_format_t *sev)
{
    char* buf = NULL;
    char numbuf[32];

    if (!fmt || !sev) return NULL;

    cJSON* json = cJSON_CreateObject();
    if (!json) goto cleanup;

    if (!cJSON_AddStringToObjLN(json, TYPE, EVENT)) goto cleanup;

    if (sev->hostname && sev->procname && sev->cmd) {
        char id[strlen(sev->hostname) + strlen(sev->procname) + strlen(sev->cmd) + 4];

        snprintf(id, sizeof(id), "%s-%s-%s", sev->hostname, sev->procname, sev->cmd);
        if (!cJSON_AddStringToObjLN(json, ID, id)) goto cleanup;
    } else {
        char id[8];
        snprintf(id, sizeof(id), "badid");
        if (!cJSON_AddStringToObjLN(json, ID, id)) goto cleanup;
    }

    if (!cJSON_AddNumberToObjLN(json, TIME, sev->timestamp)) goto cleanup;
    if (!cJSON_AddStringToObjLN(json, SOURCE, sev->src)) goto cleanup;
    cJSON* data = cJSON_CreateStringFromBuffer(sev->data, sev->datasize);
    if (!data) goto cleanup;
    cJSON_AddItemToObjectCS(json, DATA, data);
    if (!cJSON_AddStringToObjLN(json, HOST, sev->hostname)) goto cleanup;
    if (snprintf(numbuf, sizeof(numbuf), "%llu", sev->uid) < 0) goto cleanup;
    if (!cJSON_AddStringToObjLN(json, CHANNEL, numbuf)) goto cleanup;

    if (!(buf = cJSON_PrintUnformatted(json))) goto cleanup;

cleanup:
    if (!buf) {
        DBG("time=%s src=%s data=%p host=%s channel=%s json=%p",
            sev->timestamp, sev->src, sev->data, sev->hostname, numbuf, json);
    }
    if (json) cJSON_Delete(json);

    return buf;
}

static int
addJsonFields(format_t* fmt, event_field_t* fields, regex_t* fieldFilter, cJSON* json)
{
    if (!fmt || !fields) return TRUE;

    event_field_t *fld;

    // Start adding key:value entries
    for (fld = fields; fld->value_type != FMT_END; fld++) {

        // skip outputting anything that doesn't match fieldFilter
        if (fieldFilter && regexec(fieldFilter, fld->name, 0, NULL, 0)) continue;

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

static char *
fmtMetricJson(format_t *fmt, event_t *metric, regex_t* fieldFilter)
{
    char* buf = NULL;
    const char* metric_type = NULL;

    if (!fmt || !metric) return NULL;

    cJSON *json = cJSON_CreateObject();
    if (!json) goto cleanup;

    if (!cJSON_AddStringToObjLN(json, "_metric", metric->name)) goto cleanup;
    metric_type = metricTypeStr(metric->type);
    if (!cJSON_AddStringToObjLN(json, "_metric_type", metric_type)) goto cleanup;
    switch ( metric->value.type ) {
        case FMT_INT:
            if (!cJSON_AddNumberToObjLN(json, "_value", metric->value.integer)) goto cleanup;
            break;
        case FMT_FLT:
            if (!cJSON_AddNumberToObjLN(json, "_value", metric->value.floating)) goto cleanup;
            break;
        default:
            DBG(NULL);
    }

    // Add fields
    if (!addJsonFields(fmt, metric->fields, fieldFilter, json)) goto cleanup;

    buf = cJSON_PrintUnformatted(json);

cleanup:
    if (!buf) {
        DBG("_metric=%s _metric_type=%s _value=%lld, fields=%p, json=%p",
            metric->name, metric_type, metric->value, metric->fields, json);
    }
    if (json) cJSON_Delete(json);

    return buf;
}

static char*
fmtStatsDString(format_t* fmt, event_t* e, regex_t* fieldFilter)
{
    if (!fmt || !e) return NULL;

    char* end = calloc(1, fmt->statsd.max_len + 1);
    if (!end) {
         DBG("%s", e->name);
         return NULL;
    }

    char* end_start = end;

    // First, calculate size
    int bytes = 0;
    bytes += strlen(fmt->statsd.prefix);
    bytes += strlen(e->name);
    char valuebuf[320]; // :-MAX_DBL.00| => max of 315 chars for float
    int n = -1;
    switch ( e->value.type ) {
        case FMT_INT:
            n = sprintf(valuebuf, ":%lli|", e->value.integer);
            break;
        case FMT_FLT:
            n = sprintf(valuebuf, ":%.2f|", e->value.floating);
            break;
        default:
            DBG(NULL);
    }
    if (n < 0) {
        free(end_start);
        return NULL;
    }
    bytes += n; // size of value in valuebuf
    char* type = statsdType(e->type);
    bytes += strlen(type);

    // Test the calloc'd size is adequate
    if (bytes >= fmt->statsd.max_len) {
        free(end_start);
        return NULL;
    }

    // Then construct it
    end = stpcpy(end, fmt->statsd.prefix);
    end = stpcpy(end, e->name);
    end = stpcpy(end, valuebuf);
    end = stpcpy(end, type);

    // Add a newline that will get overwritten if there are fields.
    // (strcpy doesn't advance end)
    strcpy(end, "\n");

    int firstTagAdded = 0;
    addCustomFields(fmt, fmt->tags, &end, &bytes, &firstTagAdded);
    addStatsdFields(fmt, e->fields, &end, &bytes, &firstTagAdded, fieldFilter);

    // Now that we're done, we can count the trailing newline
    bytes += 1;

    return end_start;
}

char *
fmtEventMessageString(format_t *fmt, event_format_t *evmsg)
{
    if (!fmt || !evmsg) return NULL;

    switch (fmt->format) {
        case CFG_EVENT_ND_JSON:
            return fmtEventNdJson(fmt, evmsg);
        default:
            DBG("%d %s", fmt->format, evmsg->src);
            return NULL;
    }
}

char*
fmtString(format_t* fmt, event_t* e, regex_t* fieldFilter)
{
    if (!fmt) return NULL;

    switch (fmt->format) {
        case CFG_METRIC_STATSD:
            return fmtStatsDString(fmt, e, fieldFilter);
        case CFG_METRIC_JSON:
        case CFG_EVENT_ND_JSON:
            return fmtMetricJson(fmt, e, fieldFilter);
        default:
            DBG("%d %s %p", fmt->format, e->name, fieldFilter);
            return NULL;
    }
}

const char*
fmtStatsDPrefix(format_t* fmt)
{
    return (fmt && fmt->statsd.prefix) ? fmt->statsd.prefix : DEFAULT_STATSD_PREFIX;
}

unsigned
fmtStatsDMaxLen(format_t* fmt)
{
    return (fmt) ? fmt->statsd.max_len : DEFAULT_STATSD_MAX_LEN;
}

unsigned
fmtOutVerbosity(format_t* fmt)
{
    return (fmt) ? fmt->verbosity : DEFAULT_OUT_VERBOSITY;
}

custom_tag_t**
fmtCustomTags(format_t* fmt)
{
    return (fmt) ? fmt->tags : DEFAULT_CUSTOM_TAGS;
}

// Setters

void
fmtStatsDPrefixSet(format_t* fmt, const char* prefix)
{
    if (!fmt) return;

    // Don't leak on repeated sets
    if (fmt->statsd.prefix) free(fmt->statsd.prefix);
    fmt->statsd.prefix = (prefix) ? strdup(prefix) : NULL;
}

void
fmtStatsDMaxLenSet(format_t* fmt, unsigned v)
{
    if (!fmt) return;
    fmt->statsd.max_len = v;
}

void
fmtOutVerbositySet(format_t* fmt, unsigned v)
{
    if (!fmt) return;
    if (v > CFG_MAX_VERBOSITY) v = CFG_MAX_VERBOSITY;
    fmt->verbosity = v;
}

void
fmtCustomTagsSet(format_t* fmt, custom_tag_t** tags)
{
    if (!fmt) return;

    // Don't leak with multiple set operations
    fmtDestroyTags(&fmt->tags);

    if (!tags || !*tags) return;

    // get a count of how big to calloc
    int num = 0;
    while(tags[num]) num++;

    fmt->tags = calloc(1, sizeof(custom_tag_t*) * (num+1));
    if (!fmt->tags) {
        DBG(NULL);
        return;
    }

    int i, j = 0;
    for (i = 0; i<num; i++) {
        custom_tag_t* t = calloc(1, sizeof(custom_tag_t));
        char* n = strdup(tags[i]->name);
        char* v = strdup(tags[i]->value);
        if (!t || !n || !v) {
            if (t) free (t);
            if (n) free (n);
            if (v) free (v);
            DBG("t=%p n=%p v=%p", t, n, v);
            continue;
        }
        t->name = n;
        t->value = v;
        fmt->tags[j++]=t;
    }
}
