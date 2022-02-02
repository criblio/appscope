#define _GNU_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>
#include "cJSON.h"
#include "dbg.h"
#include "mtcformat.h"
#include "scopetypes.h"
#include "strset.h"
#include "com.h"


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

    mtc_fmt_t* f = calloc(1, sizeof(mtc_fmt_t));
    if (!f) {
        DBG(NULL);
        return NULL;
    }
    f->format = format;
    f->statsd.prefix = (DEFAULT_STATSD_PREFIX) ? strdup(DEFAULT_STATSD_PREFIX) : NULL;
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
        free(t[i]->name);
        free(t[i]->value);
        free(t[i]);
        i++;
    }
    free(t);
    *tags = NULL;
}

void
mtcFormatDestroy(mtc_fmt_t** fmt)
{
    if (!fmt || !*fmt) return;
    mtc_fmt_t* f = *fmt;
    if (f->statsd.prefix) free(f->statsd.prefix);
    mtcFormatDestroyTags(&f->tags);
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
createStatsFieldString(mtc_fmt_t* fmt, event_field_t* f, char* tag, int sizeoftag)
{
    if (!fmt || !f || !tag || sizeoftag <= 0) return -1;

    int sz;

    switch (f->value_type) {
        case FMT_NUM:
            sz = snprintf(tag, sizeoftag, "%s:%lli", f->name, f->value.num);
            break;
        case FMT_STR:
            if (!f->value.str) return -1;
            sz = snprintf(tag, sizeoftag, "%s:%s", f->name, f->value.str);
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
        *end = stpcpy(*end, "|#");
        *end = stpcpy(*end, tag);
        strcpy(*end, "\n"); // add newline, but don't advance end
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
addCustomFields(mtc_fmt_t* fmt, custom_tag_t** tags, char** end, int* bytes, strset_t *addedFields)
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

        sz = snprintf(tag, sizeof(tag), "%s:%s", t->name, t->value);
        if (sz < 0) break;

        appendStatsdFieldString(fmt, tag, sz, end, bytes, addedFields);
    }
}

static char*
mtcFormatStatsDString(mtc_fmt_t* fmt, event_t* e, regex_t* fieldFilter)
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

    // addedFields lets us avoid duplicate field names.  If we go to
    // add one that's already in the set, skip it.  In this way precidence
    // is given to capturedFields then custom fields then remaining fields.
    strset_t *addedFields = strSetCreate(DEFAULT_SET_SIZE);
    addStatsdFields(fmt, e->capturedFields, &end, &bytes, addedFields, NULL);
    addCustomFields(fmt, fmt->tags, &end, &bytes, addedFields);
    addStatsdFields(fmt, e->fields, &end, &bytes, addedFields, fieldFilter);
    strSetDestroy(&addedFields);

    // Now that we're done, we can count the trailing newline
    bytes += 1;

    return end_start;
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
        gettimeofday(&tv, NULL);
        double timestamp = tv.tv_sec + tv.tv_usec/1e6;
        cJSON_AddNumberToObjLN(json, "_time", timestamp);

        // add envelope for metric events 
        // https://github.com/criblio/appscope/issues/198
        
        if (!(json_root = cJSON_CreateObject())) goto out;
        if (!cJSON_AddStringToObjLN(json_root, "type", "metric")) goto out;
        cJSON_AddItemToObjectCS(json_root, "body", json);

        if ((msg = cJSON_PrintUnformatted(json_root))) {
            int strsize = strlen(msg);
            char *temp = realloc(msg, strsize+2); // room for "\n\0"
            if (!temp) {
                DBG(NULL);
                scopeLogInfo("mtcFormat realloc error");
                free(msg);
                msg = NULL;
            } else {
                msg = temp;
                msg[strsize] = '\n';
                msg[strsize+1] = '\0';
            }
        }
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
    if (fmt->statsd.prefix) free(fmt->statsd.prefix);
    fmt->statsd.prefix = (prefix) ? strdup(prefix) : NULL;
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
    char *out = malloc(strlen(in_str) * 3 + 1);
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
    char *out = malloc(strlen(in_str) + 1);
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
