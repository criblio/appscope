#define _GNU_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>
#include "dbg.h"
#include "mtcformat.h"
#include "strset.h"
#include "com.h"
#include "scopestdlib.h"


struct _mtc_fmt_t
{
    cfg_mtc_format_t format;
    struct {
        char* prefix;
        unsigned max_len;       // Max length in bytes of a statsd string
    } statsd;
    unsigned verbosity;
    custom_tag_t** tags;
};


// Constructors Destructors
mtc_fmt_t*
mtcFormatCreate(cfg_mtc_format_t format)
{
    if (format >= CFG_FORMAT_MAX) return NULL;

    mtc_fmt_t* f = scope_calloc(1, sizeof(mtc_fmt_t));
    if (!f) {
        DBG(NULL);
        return NULL;
    }
    f->format = format;
    f->statsd.prefix = (DEFAULT_STATSD_PREFIX) ? scope_strdup(DEFAULT_STATSD_PREFIX) : NULL;
    f->statsd.max_len = DEFAULT_STATSD_MAX_LEN;
    f->verbosity = DEFAULT_MTC_VERBOSITY;
    f->tags = DEFAULT_CUSTOM_TAGS;

    return f;
}

static
void
mtcFormatDestroyTags(custom_tag_t*** tags)
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
mtcFormatDestroy(mtc_fmt_t** fmt)
{
    if (!fmt || !*fmt) return;
    mtc_fmt_t* f = *fmt;
    if (f->statsd.prefix) scope_free(f->statsd.prefix);
    mtcFormatDestroyTags(&f->tags);
    scope_free(f);
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
createStatsFieldString(mtc_fmt_t* fmt, event_field_t* f, char* tag, int sizeoftag)
{
    if (!fmt || !f || !tag || sizeoftag <= 0) return -1;

    int sz;

    switch (f->value_type) {
        case FMT_NUM:
            sz = scope_snprintf(tag, sizeoftag, "%s:%lli", f->name, f->value.num);
            break;
        case FMT_STR:
            if (!f->value.str) return -1;
            sz = scope_snprintf(tag, sizeoftag, "%s:%s", f->name, f->value.str);
            break;
        default:
            DBG("%d %s", f->value_type, f->name);
            sz = -1;
    }
    return sz;
}

static void
appendStatsdFieldString(mtc_fmt_t* fmt, char* tag, int sz, char** end, int* bytes, strset_t* addedFields)
{
    if (strSetEntryCount(addedFields) == 1) {
        sz += 2; // add space for the |#
        if ((*bytes + sz) >= fmt->statsd.max_len) return;
        *end = scope_stpcpy(*end, "|#");
        *end = scope_stpcpy(*end, tag);
        scope_strcpy(*end, "\n"); // add newline, but don't advance end
    } else {
        sz += 1; // add space for the comma
        if ((*bytes + sz) >= fmt->statsd.max_len) return;
        *end = scope_stpcpy(*end, ",");
        *end = scope_stpcpy(*end, tag);
        scope_strcpy(*end, "\n"); // add newline, but don't advance end
    }
    *bytes += sz;
}


static void
addStatsdFields(mtc_fmt_t* fmt, event_field_t* fields, char** end, int* bytes, strset_t* addedFields, regex_t* fieldFilter)
{
    if (!fmt || !fields || ! end || !*end || !bytes) return;

    char tag[fmt->statsd.max_len+1];
    tag[fmt->statsd.max_len] = '\0'; // Ensures null termination
    int sz;

    event_field_t* f;
    for (f = fields; f->value_type != FMT_END; f++) {

        if (fieldFilter && regexec_wrapper(fieldFilter, f->name, 0, NULL, 0)) continue;

        // Honor Verbosity
        if (f->cardinality > fmt->verbosity) continue;

        // Don't allow duplicate field names
        if (!strSetAdd(addedFields, f->name)) continue;

        sz = createStatsFieldString(fmt, f, tag, sizeof(tag));
        if (sz < 0) continue;

        appendStatsdFieldString(fmt, tag, sz, end, bytes, addedFields);
    }
}

static void
addStatsdCustomFields(mtc_fmt_t *fmt, custom_tag_t **tags, char **end, int *bytes, strset_t *addedFields)
{
    if (!fmt || !tags || !*tags || !end || !*end || !bytes) return;

    char tag[fmt->statsd.max_len+1];
    tag[fmt->statsd.max_len] = '\0'; // Ensures null termination
    int sz;

    custom_tag_t* t;
    int i = 0;
    while ((t = tags[i++])) {

        // Don't allow duplicate field names
        if (!strSetAdd(addedFields, t->name)) continue;

        // No verbosity setting exists for custom fields.

        sz = scope_snprintf(tag, sizeof(tag), "%s:%s", t->name, t->value);
        if (sz < 0) break;

        appendStatsdFieldString(fmt, tag, sz, end, bytes, addedFields);
    }
}

static char*
mtcFormatStatsDString(mtc_fmt_t* fmt, event_t* e, regex_t* fieldFilter)
{
    if (!fmt || !e) return NULL;

    char* end = scope_calloc(1, fmt->statsd.max_len + 1);
    if (!end) {
         DBG("%s", e->name);
         return NULL;
    }

    char* end_start = end;

    // First, calculate size
    int bytes = 0;
    bytes += scope_strlen(fmt->statsd.prefix);
    bytes += scope_strlen(e->name);
    char valuebuf[320]; // :-MAX_DBL.00| => max of 315 chars for float
    int n = -1;
    switch ( e->value.type ) {
        case FMT_INT:
            n = scope_sprintf(valuebuf, ":%lli|", e->value.integer);
            break;
        case FMT_FLT:
            n = scope_sprintf(valuebuf, ":%.2f|", e->value.floating);
            break;
        default:
            DBG(NULL);
    }
    if (n < 0) {
        scope_free(end_start);
        return NULL;
    }
    bytes += n; // size of value in valuebuf
    char* type = statsdType(e->type);
    bytes += scope_strlen(type);

    // Test the calloc'd size is adequate
    if (bytes >= fmt->statsd.max_len) {
        scope_free(end_start);
        return NULL;
    }

    // Then construct it
    end = scope_stpcpy(end, fmt->statsd.prefix);
    end = scope_stpcpy(end, e->name);
    end = scope_stpcpy(end, valuebuf);
    end = scope_stpcpy(end, type);

    // Add a newline that will get overwritten if there are fields.
    // (strcpy doesn't advance end)
    scope_strcpy(end, "\n");

    // addedFields lets us avoid duplicate field names.  If we go to
    // add one that's already in the set, skip it.  In this way precedence
    // is given to capturedFields then custom fields then remaining fields.
    strset_t *addedFields = strSetCreate(DEFAULT_SET_SIZE);
    if (addedFields) {
        addStatsdFields(fmt, e->capturedFields, &end, &bytes, addedFields, NULL);
        addStatsdCustomFields(fmt, fmt->tags, &end, &bytes, addedFields);
        addStatsdFields(fmt, e->fields, &end, &bytes, addedFields, fieldFilter);
        strSetDestroy(&addedFields);
    }

    // Now that we're done, we can count the trailing newline
    bytes += 1;

    return end_start;
}

static bool
appendPromField(FILE *stream, mtc_fmt_t *fmt, event_field_t *field, strset_t *addedFields)
{
    if (!fmt || !field) return FALSE;

    char delim;
    if (strSetEntryCount(addedFields) == 1) {
        delim= '{';
    } else {
        delim= ',';
    }

    int retVal = -1;
    switch (field->value_type) {
        case FMT_NUM:
            retVal = scope_fprintf(stream, "%c%s=\"%lli\"", delim, field->name, field->value.num);
            break;
        case FMT_STR:
            retVal = scope_fprintf(stream, "%c%s=\"%s\"", delim, field->name, field->value.str);
            break;
        default:
            DBG("%d %s", field->value_type, field->name);
    }
    return retVal >= 0;
}

static bool
addPromFields(mtc_fmt_t *fmt, event_field_t *fields, FILE *stream, strset_t *addedFields, regex_t *fieldFilter)
{
    if (!fmt || !stream) return FALSE;
    if (!fields) return TRUE;            // ok, just no fields to add

    event_field_t *field;
    for (field = fields; field->value_type != FMT_END; field++) {

        if (fieldFilter && regexec_wrapper(fieldFilter, field->name, 0, NULL, 0)) continue;

        // Skip empty values
        if (field->value_type == FMT_STR && !field->value.str) continue;

        // Honor Verbosity
        if (field->cardinality > fmt->verbosity) continue;

        // Don't allow duplicate field names
        if (!strSetAdd(addedFields, field->name)) continue;

        if (!appendPromField(stream, fmt, field, addedFields)) return FALSE;
    }

    return TRUE;
}

static bool
addPromCustomFields(mtc_fmt_t *fmt, custom_tag_t **tags, FILE *stream, strset_t *addedFields)
{
    if (!fmt || !stream) return FALSE;
    if (!tags || !*tags) return TRUE;    // ok, just no fields to add

    custom_tag_t *tag;
    int i = 0;
    while ((tag = tags[i++])) {

        // Don't allow duplicate field names
        if (!strSetAdd(addedFields, tag->name)) continue;

        // No verbosity setting exists for custom fields.

        event_field_t f = STRFIELD(tag->name, tag->value, 0, TRUE);
        if (!appendPromField(stream, fmt, &f, addedFields)) return FALSE;
    }

    return TRUE;
}

static const char *
promTypeStr(data_type_t type)
{
    if (type == CURRENT) return "gauge";
    // For prometheus format, we're not currently using
    // timer, histogram, or set.  Return counter for all of these.
    return "counter";
}

static char *
mtcFormatPromString(mtc_fmt_t *fmt, event_t *evt, regex_t *fieldFilter)
{
    if (!fmt || !evt) return NULL;

    bool completely_successful = FALSE;
    char *name = NULL;
    FILE *stream = NULL;
    char *prom_str = NULL;
    size_t size = 0;
    strset_t *addedFields = NULL;

    // memstream, just to let it manage growth in prom_str as needed.
    stream = scope_open_memstream(&prom_str, &size);
    if (!stream || !prom_str) goto out;

    // Copy metric name into a buffer, then convert "." to "_".
    if (scope_asprintf(&name, "%s%s", fmt->statsd.prefix, evt->name) < 0) {
        name = NULL;
        goto out;
    }
    char *ptr;
    for (ptr=name; *ptr!='\0'; ptr++) {
        if (*ptr=='.') *ptr='_';
    }

    // Add the TYPE comment line
    if (scope_fprintf(stream, "# TYPE %s %s\n", name, promTypeStr(evt->type) ) < 0) goto out;
    // Add the metric, start with the name
    if (scope_fprintf(stream, "%s", name) < 0) goto out;


    // Add fields to the metric
    if (!(addedFields = strSetCreate(DEFAULT_SET_SIZE))) goto out;
    if (!addPromFields(fmt, evt->capturedFields, stream, addedFields, NULL)) goto out;
    if (!addPromCustomFields(fmt, fmt->tags, stream, addedFields)) goto out;
    if (!addPromFields(fmt, evt->fields, stream, addedFields, fieldFilter)) goto out;
    if (strSetEntryCount(addedFields) >= 1) {
        if (scope_fprintf(stream, "}") < 0) goto out;
    }

    // Add the value to the metric
    switch ( evt->value.type ) {
        case FMT_INT:
            if (scope_fprintf(stream, " %lli\n", evt->value.integer) < 0) goto out;
            break;
        case FMT_FLT:
            if (scope_fprintf(stream, " %.2f\n", evt->value.floating) < 0) goto out;
            break;
        default:
            DBG(NULL);
    }

    // Prometheus metric timestamps are optional.
    // If desired, here's the spot to add them.

    completely_successful = TRUE;

out:
    if (stream) scope_fclose(stream);
    if (name) scope_free(name);
    if (!completely_successful) {
        DBG("%d %s", evt->value.type, evt->name);
        if (prom_str) scope_free(prom_str);
        prom_str = NULL;
    }
    strSetDestroy(&addedFields);
    return prom_str;
}

char *
mtcFormatEventForOutput(mtc_fmt_t *fmt, event_t *evt, regex_t *fieldFilter)
{
    if (!fmt || !evt ) return NULL;

    char *msg = NULL;
    cJSON *json = NULL;
    cJSON *json_root = NULL;

    if (fmt->format == CFG_FMT_STATSD) {
        msg = mtcFormatStatsDString(fmt, evt, fieldFilter);
    } else if (fmt->format == CFG_FMT_NDJSON) {
        custom_tag_t **tags = mtcFormatCustomTags(fmt);
        if (!(json = fmtMetricJson(evt, NULL, CFG_SRC_METRIC, tags))) goto out;

        // Request is for this json, plus a _time field
        struct timeval tv;
        scope_gettimeofday(&tv, NULL);
        double timestamp = tv.tv_sec + tv.tv_usec/1e6;
        cJSON_AddNumberToObjLN(json, "_time", timestamp);

        // add envelope for metric events 
        // https://github.com/criblio/appscope/issues/198
        
        if (!(json_root = cJSON_CreateObject())) goto out;
        if (!cJSON_AddStringToObjLN(json_root, "type", "metric")) goto out;
        cJSON_AddItemToObjectCS(json_root, "body", json);

        if ((msg = cJSON_PrintUnformatted(json_root))) {
            int strsize = scope_strlen(msg);
            char *temp = scope_realloc(msg, strsize+2); // room for "\n\0"
            if (!temp) {
                DBG(NULL);
                scopeLogInfo("mtcFormat scope_realloc error");
                scope_free(msg);
                msg = NULL;
            } else {
                msg = temp;
                msg[strsize] = '\n';
                msg[strsize+1] = '\0';
            }
        }
    } else if (fmt->format == CFG_FMT_PROMETHEUS) {
        msg = mtcFormatPromString(fmt, evt, fieldFilter);
    }

out:
    if (json_root) {
        cJSON_Delete(json_root);
    } else if (json) {
        cJSON_Delete(json);
    }
    return msg;
}

const char*
mtcFormatStatsDPrefix(mtc_fmt_t* fmt)
{
    return (fmt && fmt->statsd.prefix) ? fmt->statsd.prefix : DEFAULT_STATSD_PREFIX;
}

unsigned
mtcFormatStatsDMaxLen(mtc_fmt_t* fmt)
{
    return (fmt) ? fmt->statsd.max_len : DEFAULT_STATSD_MAX_LEN;
}

unsigned
mtcFormatVerbosity(mtc_fmt_t* fmt)
{
    return (fmt) ? fmt->verbosity : DEFAULT_MTC_VERBOSITY;
}

custom_tag_t**
mtcFormatCustomTags(mtc_fmt_t* fmt)
{
    return (fmt) ? fmt->tags : DEFAULT_CUSTOM_TAGS;
}

// Setters

void
mtcFormatStatsDPrefixSet(mtc_fmt_t* fmt, const char* prefix)
{
    if (!fmt) return;

    // Don't leak on repeated sets
    if (fmt->statsd.prefix) scope_free(fmt->statsd.prefix);
    fmt->statsd.prefix = (prefix) ? scope_strdup(prefix) : NULL;
}

void
mtcFormatStatsDMaxLenSet(mtc_fmt_t* fmt, unsigned v)
{
    if (!fmt) return;
    fmt->statsd.max_len = v;
}

void
mtcFormatVerbositySet(mtc_fmt_t* fmt, unsigned v)
{
    if (!fmt) return;
    if (v > CFG_MAX_VERBOSITY) v = CFG_MAX_VERBOSITY;
    fmt->verbosity = v;
}

void
mtcFormatCustomTagsSet(mtc_fmt_t* fmt, custom_tag_t** tags)
{
    if (!fmt) return;

    // Don't leak with multiple set operations
    mtcFormatDestroyTags(&fmt->tags);

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

static int
isHex(const char x)
{
    // I'm not using isxdigit because I don't want locale to affect it.
    if (x >= '0' && x <= '9') return 1;
    if (x >= 'A' && x <= 'F') return 1;
    if (x >= 'a' && x <= 'f') return 1;
    return 0;
}
static char
fromHex(const char x) {
    if (x >= '0' && x <= '9') return (x - '0');
    if (x >= 'A' && x <= 'F') return (x - 'A' + 10);
    if (x >= 'a' && x <= 'f') return (x - 'a' + 10);

    DBG(NULL);
    return 0;
}

static char
toHex(const char code) {
    static char hex[] = "0123456789ABCDEF";
    return hex[code & 15];
}

static int
isUnreserved(const char x)
{
    // I'm not using isalnum because I don't want locale to affect it.
    if (x >= 'a' && x <= 'z') return 1;
    if (x >= 'A' && x <= 'Z') return 1;
    if (x >= '0' && x <= '9') return 1;
    if (x == '-' || x == '.' || x == '_' || x == '~') return 1;
    return 0;
}

char*
fmtUrlEncode(const char* in_str)
{
    // rfc3986 Percent-Encoding
    if (!in_str) return NULL;
    char *out = scope_malloc(scope_strlen(in_str) * 3 + 1);
    if (!out) return NULL;

    char *inptr = (char*) in_str;
    char *outptr = out;

    while (*inptr) {
        if (isUnreserved(*inptr)) {
            *outptr++ = *inptr;
        } else {
            *outptr++ = '%';
            *outptr++ = toHex(*inptr >> 4);
            *outptr++ = toHex(*inptr);
        }
        inptr++;
    }
    *outptr = '\0';
    return out;
}

char*
fmtUrlDecode(const char* in_str)
{
    // rfc3986 Percent-Encoding
    if (!in_str) return NULL;
    char *out = scope_malloc(scope_strlen(in_str) + 1);
    if (!out) return NULL;

    char *inptr = (char*) in_str;
    char *outptr = out;

    while (*inptr) {
        if (*inptr == '%') {
            if (isHex(inptr[1]) && isHex(inptr[2])) {
                *outptr++ = fromHex(inptr[1]) << 4 | fromHex(inptr[2]);
                inptr += 2;
            } else {
                DBG(NULL);
                break;
            }
        } else {
            *outptr++ = *inptr;
        }
        inptr++;
    }
    *outptr = '\0';
    return out;
}
