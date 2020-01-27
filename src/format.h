#ifndef __FORMAT_H__
#define __FORMAT_H__

#include <regex.h>
#include "scopetypes.h"
#include "cfg.h"
#include "cJSON.h"


// This event structure is meant to meet our needs w.r.t. statsd,
// but abstracting it a bit.  In this terminology, a statsd metric
// with (dogstatsd) tags is an event with event fields.

typedef enum {FMT_END, FMT_STR, FMT_NUM} field_value_t;

typedef struct {
    const char* const name;
    const field_value_t value_type;
    union {
        const char* str;
        long long num;
    } value;
    const unsigned cardinality;   // verbosity uses this
} event_field_t;

#define STRFIELD(n,v,c) {n,    FMT_STR, { .str = v }, c}
#define NUMFIELD(n,v,c) {n,    FMT_NUM, { .num = v }, c}
#define FIELDEND        {NULL, FMT_END, { .num = 0 }, 0}

// statsd:  COUNTER,   GAUGE,    TIMER, HISTOGRAM, SET
typedef enum {DELTA, CURRENT, DELTA_MS, HISTOGRAM, SET} data_type_t;

typedef enum {FMT_INT, FMT_FLT} value_t;
typedef struct {
    const char* const name;
    struct {
        const value_t type;
        union {
            const long long integer;
            double floating;
        };
    } value;
    const data_type_t type;
    event_field_t* fields;
} event_t;

#define INT_EVENT(n, v, t, f) {n, { FMT_INT, .integer=v}, t, f}
#define FLT_EVENT(n, v, t, f) {n, { FMT_FLT, .floating=v}, t, f}

typedef struct event_format {
    double timestamp;
    const char *src;
    proc_id_t* proc;
    cJSON *data;
    unsigned long long uid;
} event_format_t;

typedef struct _format_t format_t;

// Constructors Destructors
format_t*           fmtCreate(cfg_out_format_t);
void                fmtDestroy(format_t**);

// Accessors
const char*         fmtStatsDPrefix(format_t*);
unsigned            fmtStatsDMaxLen(format_t*);
unsigned            fmtOutVerbosity(format_t*);
custom_tag_t**      fmtCustomTags(format_t*);

// This function returns a pointer to a malloc()'d buffer.
// The caller is responsible for deallocating with free().
char*               fmtStatsDString(format_t*, event_t*, regex_t*);

// These functions return a pointer to malloc()'d buffers.
// The caller is reposibile for deallocating with cJSON_Delete().
cJSON *             fmtMetricJson(event_t*, regex_t*);
cJSON *             fmtEventJson(event_format_t *);

// Setters
void                fmtStatsDPrefixSet(format_t*, const char*);
void                fmtStatsDMaxLenSet(format_t*, unsigned);
void                fmtOutVerbositySet(format_t*, unsigned);
void                fmtCustomTagsSet(format_t*, custom_tag_t**);

// Helper functions - returns a pointer to a malloc'd buffer.
// The caller is reponsible for deallocating with free().
char* fmtUrlEncode(const char*);
char* fmtUrlDecode(const char*);

#endif // __FORMAT_H__
