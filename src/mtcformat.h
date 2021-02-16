#ifndef __MTC_FORMAT_H__
#define __MTC_FORMAT_H__

#include "pcre2posix.h"
#include "scopetypes.h"
#include "cfg.h"
#include "cJSON.h"


// This event structure is meant to meet our needs w.r.t. statsd,
// but abstracting it a bit.  In this terminology, a statsd metric
// with (dogstatsd) tags is an event with event fields.

typedef enum {FMT_END, FMT_STR, FMT_NUM} field_value_t;

typedef struct {
    const char *name;
    field_value_t value_type;
    bool event_usage;
    union {
        const char *str;
        long long num;
    } value;
    unsigned cardinality;   // verbosity uses this
} event_field_t;

#define STRFIELD(n,v,c,e) {n,    FMT_STR, e, { .str = v }, c}
#define NUMFIELD(n,v,c,e) {n,    FMT_NUM, e, { .num = v }, c}
#define FIELDEND        {NULL, FMT_END, FALSE, { .num = 0 }, 0}

// statsd:  COUNTER,   GAUGE,    TIMER, HISTOGRAM, SET
typedef enum {DELTA, CURRENT, DELTA_MS, HISTOGRAM, SET} data_type_t;

typedef enum {FMT_INT, FMT_FLT} value_t;
typedef struct {
    const char *const name;
    struct {
        const value_t type;
        union {
            const long long integer;
            double floating;
        };
    } value;
    const data_type_t type;
    event_field_t *fields;
    watch_t src;
    cJSON *data;
} event_t;

#define INT_EVENT(n, v, t, f) {n, { FMT_INT, .integer=v}, t, f, CFG_SRC_METRIC}
#define FLT_EVENT(n, v, t, f) {n, { FMT_FLT, .floating=v}, t, f, CFG_SRC_METRIC}

typedef struct event_format {
    double timestamp;
    const char *src;
    proc_id_t* proc;
    unsigned long long uid;
    cJSON *data;
    watch_t sourcetype;
} event_format_t;

typedef struct _mtc_fmt_t mtc_fmt_t;

// Constructors Destructors
mtc_fmt_t*          mtcFormatCreate(cfg_mtc_format_t);
void                mtcFormatDestroy(mtc_fmt_t**);

// Accessors
const char*         mtcFormatStatsDPrefix(mtc_fmt_t*);
unsigned            mtcFormatStatsDMaxLen(mtc_fmt_t*);
unsigned            mtcFormatVerbosity(mtc_fmt_t*);
custom_tag_t**      mtcFormatCustomTags(mtc_fmt_t*);

// This function returns a pointer to a malloc()'d buffer.
// The caller is responsible for deallocating with free().
char*               mtcFormatEventForOutput(mtc_fmt_t*, event_t*, regex_t*);

// Setters
void                mtcFormatStatsDPrefixSet(mtc_fmt_t*, const char*);
void                mtcFormatStatsDMaxLenSet(mtc_fmt_t*, unsigned);
void                mtcFormatVerbositySet(mtc_fmt_t*, unsigned);
void                mtcFormatCustomTagsSet(mtc_fmt_t*, custom_tag_t**);

// Helper functions - returns a pointer to a malloc'd buffer.
// The caller is reponsible for deallocating with free().
char* fmtUrlEncode(const char*);
char* fmtUrlDecode(const char*);

#endif // __MTC_FORMAT_H__
