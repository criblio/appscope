#define _GNU_SOURCE

#include <regex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dbg.h"
#include "format.h"
#include "scopetypes.h"


struct _format_t
{
    cfg_out_format_t format;
    struct {
        char* prefix;
        unsigned max_len;       // Max length in bytes of a statsd string
    } statsd;
    unsigned verbosity;
    custom_tag_t** tags;
    regex_t* regex;
};


// Constructors Destructors
format_t*
fmtCreate(cfg_out_format_t format)
{
    if (format >= CFG_FORMAT_MAX) return NULL;

    format_t* f = calloc(1, sizeof(format_t));
    if (!f) return NULL;
    f->format = format;
    f->statsd.prefix = (DEFAULT_STATSD_PREFIX) ? strdup(DEFAULT_STATSD_PREFIX) : NULL;
    f->statsd.max_len = DEFAULT_STATSD_MAX_LEN;
    f->verbosity = DEFAULT_OUT_VERBOSITY;
    f->tags = DEFAULT_CUSTOM_TAGS;
    f->regex = calloc(1, sizeof(regex_t));
    if ((f->regex) && regcomp(f->regex, "\\$[a-zA-Z0-9_]+", REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        free(f->regex);
        f->regex = NULL;
    }

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
    if (f->regex) regfree(f->regex);
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
static char*
fmtStatsDString(format_t* fmt, event_t* e)
{
    if (!fmt || !e) return NULL;

    char* end = calloc(1, fmt->statsd.max_len + 1);
    if (!end) return NULL;

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

char*
fmtString(format_t* fmt, event_t* e)
{
    if (!fmt) return NULL;

    switch (fmt->format) {
        case CFG_EXPANDED_STATSD:
            return fmtStatsDString(fmt, e);
        case CFG_NEWLINE_DELIMITED:
        default:
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

static char*
doEnvVariableSubstitution(format_t* fmt, char* value)
{
    if (!fmt || !value) return NULL;

    regmatch_t match = {0};

    int out_size = strlen(value) + 1;
    char* outval = calloc(1, out_size);
    if (!outval) return NULL;

    char* outptr = outval;  // "tail" pointer where text can be appended
    char* inptr = value;    // "current" pointer as we scan through value

    while (fmt->regex && !regexec(fmt->regex, inptr, 1, &match, 0)) {

        int match_size = match.rm_eo - match.rm_so;

        // if the character before the match is '\', don't do substitution
        char* escape_indicator = &inptr[match.rm_so - 1];
        int escaped = (escape_indicator >= value) && (*escape_indicator == '\\');

        if (escaped) {
            // copy the part before the match, except the escape char '\'
            outptr = stpncpy(outptr, inptr, match.rm_so - 1);
            // copy the matching env variable name
            outptr = stpncpy(outptr, &inptr[match.rm_so], match_size);
            // move to the next part of the input value
            inptr = &inptr[match.rm_eo];
            continue;
        }

        // lookup the part that matched to see if we can substitute it
        char env_name[match_size + 1];
        strncpy(env_name, &inptr[match.rm_so], match_size);
        env_name[match_size] = '\0';
        char* env_value = getenv(&env_name[1]); // offset of 1 skips the $

        // Grow outval buffer any time env_value is bigger than env_name
        int size_growth = (!env_value) ? 0 : strlen(env_value) - match_size;
        if (size_growth > 0) {
            char* new_outval = realloc (outval, out_size + size_growth);
            if (new_outval) {
                out_size += size_growth;
                outptr = new_outval + (outptr - outval);
                outval = new_outval;
            } else {
                free(outval);
                return NULL;
            }
        }

        // copy the part before the match
        outptr = stpncpy(outptr, inptr, match.rm_so);
        // either copy in the env value or the variable that wasn't found
        outptr = stpcpy(outptr, (env_value) ? env_value : env_name);
        // move to the next part of the input value
        inptr = &inptr[match.rm_eo];
    }

    // copy whatever is left
    strcpy(outptr, inptr);

    return outval;
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
    if (!fmt->tags) return;

    int i, j = 0;
    for (i = 0; i<num; i++) {
        custom_tag_t* t = calloc(1, sizeof(custom_tag_t));
        char* n = strdup(tags[i]->name);
        char* v = doEnvVariableSubstitution(fmt, tags[i]->value);
        if (!t || !n || !v) {
            if (t) free (t);
            if (n) free (n);
            if (v) free (v);
            continue;
        }
        fmt->tags[j++]=t;
        t->name = n;
        t->value = v;
    }
}
