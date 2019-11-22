#define _GNU_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "dbg.h"
#include "format.h"
#include "scopetypes.h"
#include "yaml.h"

// key names for event JSON
#define HOST "host"
#define TIME "_time"
#define DATA "_raw"
#define SOURCE "source"
#define CHANNEL "_channel"

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
    regex_t* metric_field_re;
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
    f->metric_field_re = NULL;

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
        sz += 3; // add space for the |#, and trailing newline
        if ((*bytes + sz) > fmt->statsd.max_len) return;
        *end = stpcpy(*end, "|#");
        *end = stpcpy(*end, tag);
        strcpy(*end, "\n"); // add newline, but don't advance end
        *firstTagAdded = 1;
    } else {
        sz += 2; // add space for the comma, and trailing newline
        if ((*bytes + sz) > fmt->statsd.max_len) return;
        *end = stpcpy(*end, ",");
        *end = stpcpy(*end, tag);
        strcpy(*end, "\n"); // add newline, but don't advance end
    }
    *bytes += (sz - 1); // Don't count the newline, it's ok to overwrite
}


static void
addStatsdFields(format_t* fmt, event_field_t* fields, char** end, int* bytes, int* firstTagAdded)
{
    if (!fmt || !fields || ! end || !*end || !bytes) return;

    char tag[fmt->statsd.max_len+1];
    tag[fmt->statsd.max_len] = '\0'; // Ensures null termination
    int sz;

    event_field_t* f;
    for (f = fields; f->value_type != FMT_END; f++) {

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
static size_t
eventJsonSize(event_format_t *event)
{
    if (!event || !event->src || !event->datasize || !event->timesize) return 0;
    size_t size = sizeof("{'_time':,'source':'','_raw':'','host':'','_channel':''}");

    // time
    size += event->timesize;
    // source
    size += strlen(event->src);
    // _raw
    size += event->datasize;
    // host
    size += strlen(event->hostname);
    // _channel
    size += strlen("18446744073709551615");
    // fudge factor for things like escaped single quote characters.
    size += 256;
    return size;
}

static char *
fmtEventJson(format_t *fmt, event_format_t *sev)
{
    if (!sev) return NULL;

    yaml_emitter_t emitter;
    yaml_event_t event;
    char numbuf[32];
    int rv;

    int emitter_created = 0;
    int emitter_opened = 0;
    int everything_successful = 0;

    size_t bytes_written = 0;
    char* buf = NULL;

    // Get the size of a complete json string & allocate
    const size_t bufsize = eventJsonSize(sev);
    if (!bufsize) goto cleanup;
    buf = calloc(1, bufsize);
    if (!buf) goto cleanup;

    // Yaml init stuff
    emitter_created = yaml_emitter_initialize(&emitter);
    if (!emitter_created) goto cleanup;

    yaml_emitter_set_unicode(&emitter, 1);

    yaml_emitter_set_output_string(&emitter, (yaml_char_t*)buf, bufsize,
                                   &bytes_written);
    emitter_opened = yaml_emitter_open(&emitter);
    if (!emitter_opened) goto cleanup;

    rv = yaml_document_start_event_initialize(&event, NULL, NULL, NULL, 1);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_mapping_start_event_initialize(&event, NULL,
                                             (yaml_char_t*)YAML_MAP_TAG, 1,
                                             YAML_FLOW_MAPPING_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // Start adding key:value entries
    // "TIME": "epochvalue.ms",
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)TIME, strlen(TIME), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_FLOAT_TAG,
                                      (yaml_char_t*)sev->timestamp, sev->timesize, 1, 0,
                                      YAML_PLAIN_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // SOURCE: "pathname" || "syslog" || "high cardinality metric" || ...
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)SOURCE, strlen(SOURCE), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)sev->src, strlen(sev->src), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // "data": "app data",
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)DATA, strlen(DATA), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)sev->data, sev->datasize, 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // HOST: "hostname"
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)HOST, strlen(HOST), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)sev->hostname, strlen(sev->hostname), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // "channel": "unique value",
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)CHANNEL, strlen(CHANNEL), 0, 1,
                                      YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = snprintf(numbuf, sizeof(numbuf), "%llu", sev->uid);
    if (rv <= 0) goto cleanup;

    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)numbuf, rv, 0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // Done with key:value entries
    // Tell yaml to wrap it up
    rv = yaml_mapping_end_event_initialize(&event);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_document_end_event_initialize(&event, 1);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    everything_successful = 1;

cleanup:
    if (!everything_successful) {
        DBG("bufsize=%zu bytes_written=%zu buf=%p emitter_created=%d "
            "emitter_opened=%d emitter_error=%d emitter_problem=%s",
            bufsize, bytes_written, buf, emitter_created,
            emitter_opened, emitter.error, emitter.problem);
        if (buf) free(buf);
        buf = NULL;
    }

    if (emitter_opened) yaml_emitter_close(&emitter);
    if (emitter_created) yaml_emitter_delete(&emitter);

    return buf;
}

static size_t
metricJsonSize(event_t *metric)
{
    if (!metric) return 0;

    size_t size = 256; // fudge factor for things like escaped quote chars.

    size += strlen(metric->name) + 4;            // include {, "", :
    size += 21;                                  // -9223372036854775808,

    if (!metric->fields) return size;

    event_field_t *fld;
    for (fld = metric->fields; fld->value_type != FMT_END; fld++) {
        size += strlen(fld->name) + 3;           // include "", :

        switch (fld->value_type) {
        case FMT_NUM:
            size += 21;                          // -9223372036854775808,
            break;
        case FMT_STR:
            size += strlen(fld->value.str) + 3;  // include quotes and comma
            break;
        default:
            break;
        }
    }

    return size;
}

static int
addJsonFields(format_t* fmt, event_field_t* fields, yaml_emitter_t* emitter)
{
    if (!fmt || !fields) return TRUE;

    int rv;
    yaml_event_t event;
    char numbuf[32];
    event_field_t *fld;
    regex_t *field_filter;
    regmatch_t match = {0};

    field_filter = fmtMetricFieldFilter(fmt);
    
    // Start adding key:value entries
    for (fld = fields; fld->value_type != FMT_END; fld++) {
        // if the field filter matches, we exclude that field
        if (field_filter &&
            (regexec(field_filter, fld->name, 1, &match, 0) == 0)) {
            continue;
        }

        // "Key"
        rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                          (yaml_char_t*)fld->name, strlen(fld->name), 0, 1,
                                          YAML_DOUBLE_QUOTED_SCALAR_STYLE);
        if (!rv || !yaml_emitter_emit(emitter, &event)) return FALSE;

        // "Value"
        if (fld->value_type == FMT_STR) {
            rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                              (yaml_char_t*)fld->value.str, strlen(fld->value.str),
                                              0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
            if (!rv || !yaml_emitter_emit(emitter, &event)) return FALSE;
        } else if (fld->value_type == FMT_NUM) {
                rv = snprintf(numbuf, sizeof(numbuf), "%lli" , fld->value.num);
                if (rv <= 0) return FALSE;

                rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_INT_TAG,
                                                  (yaml_char_t*)numbuf, rv, 1, 0,
                                                  YAML_PLAIN_SCALAR_STYLE);
                if (!rv || !yaml_emitter_emit(emitter, &event)) return FALSE;
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
fmtMetricJson(format_t *fmt, event_t *metric)
{
    if (!metric) return NULL;

    yaml_emitter_t emitter;
    yaml_event_t event;
    char numbuf[32];
    int rv;
    int emitter_created = 0;
    int emitter_opened = 0;
    int everything_successful = 0;
    size_t bytes_written = 0;
    char* buf = NULL;

    // Get the size of a complete json string & allocate
    const size_t bufsize = metricJsonSize(metric);
    if (!bufsize) goto cleanup;
    buf = calloc(1, bufsize);
    if (!buf) goto cleanup;

    // Yaml init stuff
    emitter_created = yaml_emitter_initialize(&emitter);
    if (!emitter_created) goto cleanup;

    yaml_emitter_set_unicode(&emitter, 1);

    yaml_emitter_set_output_string(&emitter, (yaml_char_t*)buf, bufsize,
                                   &bytes_written);
    emitter_opened = yaml_emitter_open(&emitter);
    if (!emitter_opened) goto cleanup;

    rv = yaml_document_start_event_initialize(&event, NULL, NULL, NULL, 1);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_mapping_start_event_initialize(&event, NULL,
                                             (yaml_char_t*)YAML_MAP_TAG, 1,
                                             YAML_FLOW_MAPPING_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // add the base metric definition
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)"_metric", strlen("_metric"),
                                      0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)metric->name, strlen(metric->name),
                                      0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // add the metric type
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)"_metric_type", strlen("_metric_type"),
                                      0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;
    const char* metric_type = metricTypeStr(metric->type);
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)metric_type, strlen(metric_type),
                                      0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // add the value
    rv = snprintf(numbuf, sizeof(numbuf), "%lli", metric->value);
    if (rv <= 0) goto cleanup;
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_STR_TAG,
                                      (yaml_char_t*)"_value", strlen("_value"),
                                      0, 1, YAML_DOUBLE_QUOTED_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;
    rv = yaml_scalar_event_initialize(&event, NULL, (yaml_char_t*)YAML_INT_TAG,
                                      (yaml_char_t*)numbuf, rv, 1, 0, YAML_PLAIN_SCALAR_STYLE);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    // Add key:value fields
    if (!addJsonFields(fmt, metric->fields, &emitter)) goto cleanup;

    // Done with key:value entries
    // Tell yaml to wrap it up
    rv = yaml_mapping_end_event_initialize(&event);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    rv = yaml_document_end_event_initialize(&event, 1);
    if (!rv || !yaml_emitter_emit(&emitter, &event)) goto cleanup;

    everything_successful = 1;

cleanup:
    if (!everything_successful) {
        DBG("bufsize=%zu bytes_written=%zu buf=%p emitter_created=%d "
            "emitter_opened=%d emitter_error=%d emitter_problem=%s",
            bufsize, bytes_written, buf, emitter_created,
            emitter_opened, emitter.error, emitter.problem);
        if (buf) free(buf);
        buf = NULL;
    }

    if (emitter_opened) yaml_emitter_close(&emitter);
    if (emitter_created) yaml_emitter_delete(&emitter);

    return buf;
}

static char*
fmtStatsDString(format_t* fmt, event_t* e)
{
    if (!fmt || !e) return NULL;

    char* end = calloc(1, fmt->statsd.max_len + 1);
    if (!end) {
         DBG("%s", e->name);
         return NULL;
    }

    char* end_start = end;

    // First, just estimate size, w/worst case
    int bytes = 0;
    bytes += strlen(fmt->statsd.prefix);
    bytes += strlen(e->name);
    bytes += 22; // :-9223372036854775808|
    bytes += 3;  // ms, newline
    if (bytes > fmt->statsd.max_len) {
        free(end_start);
        return NULL;
    }

    // Construct it
    end = stpcpy(end, fmt->statsd.prefix);
    end = stpcpy(end, e->name);
    int n = sprintf(end, ":%lli|", e->value);
    if (n>0) end += n;
    end = stpcpy(end, statsdType(e->type));

    // Update bytes to reflect actual (not worst case, probabaly)
    bytes = end - end_start;

    // Add a newline that will get overwritten if there are fields
    // strcpy doesn't advance end
    strcpy(end, "\n");

    int firstTagAdded = 0;
    addCustomFields(fmt, fmt->tags, &end, &bytes, &firstTagAdded);
    addStatsdFields(fmt, e->fields, &end, &bytes, &firstTagAdded);

    // Now that we're done, we can count the trailing newline
    bytes += 1;

    return end_start;
}

char *
fmtEventMessageString(format_t *fmt, event_format_t *evmsg)
{
    if (!fmt || !evmsg) return NULL;

    switch (fmt->format) {
        case CFG_EVENT_JSON_RAW_JSON:
            return fmtEventJson(fmt, evmsg);
        default:
            DBG("%d %s", fmt->format, evmsg->src);
            return NULL;
    }
}

char*
fmtString(format_t* fmt, event_t* e)
{
    if (!fmt) return NULL;

    switch (fmt->format) {
        case CFG_METRIC_STATSD:
        case CFG_EVENT_JSON_RAW_STATSD:
            return fmtStatsDString(fmt, e);
        case CFG_METRIC_JSON:
        case CFG_EVENT_JSON_RAW_JSON:
            return fmtMetricJson(fmt, e);
        default:
            DBG("%d %s", fmt->format, e->name);
            return NULL;
    }
}

regex_t *
fmtMetricFieldFilter(format_t *fmt)
{
    if (!fmt) return NULL;
    return fmt->metric_field_re;
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
fmtMetricFieldFilterSet(format_t *fmt, regex_t *filter)
{
    if (!fmt || !filter) return;

    fmt->metric_field_re = filter;
}

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
