#define _GNU_SOURCE
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include "com.h"
#include "dbg.h"
#include "httpagg.h"
#include "utils.h"


#define DEFAULT_TARGET_LEN ( 128 )
#define MAX_CODE_ENTRIES ( 64 )

typedef enum {
    SERVER_DURATION,
    CLIENT_DURATION,
    REQUEST_BYTES,
    RESPONSE_BYTES,
    FIELD_MAX
} counter_field_enum;

enum_map_t fieldMapIn[] = {
    {"http_server_duration",          SERVER_DURATION},
    {"http_client_duration",          CLIENT_DURATION},
    {"http_request_content_length",   REQUEST_BYTES},
    {"http_response_content_length",  RESPONSE_BYTES},
    {NULL,                    -1}
};

enum_map_t fieldMapOut[] = {
    {"http.server.duration",          SERVER_DURATION},
    {"http.client.duration",          CLIENT_DURATION},
    {"http.request.content_length",   REQUEST_BYTES},
    {"http.response.content_length",  RESPONSE_BYTES},
    {NULL,                    -1}
};

typedef struct {
    int code;             // example status code values: 200, 404, 503, etc.
    uint64_t count;       // the number of this code seen
} status_code_t;

typedef struct {
    uint64_t total;       // cumulative total
    uint64_t num_entries; // number of entries, to support average calculation
} agg_counter_t;

typedef struct {
    char * uri;           // the key that comes from http_target
    status_code_t status[MAX_CODE_ENTRIES];
    agg_counter_t field[FIELD_MAX];
} target_agg_t;

struct _http_agg_t {
    target_agg_t** target;
    uint64_t count;
    uint64_t alloc;
};


http_agg_t *
httpAggCreate()
{
    http_agg_t* agg = calloc(1, sizeof(*agg));
    target_agg_t** target_lst = calloc(1, sizeof(*target_lst) * DEFAULT_TARGET_LEN);
    if (!agg || !target_lst) {
        if (agg) free(agg);
        if (target_lst) free(target_lst);
        DBG("agg = %p, target_lst = %p", agg, target_lst);
        return NULL;
    }

    agg->target = target_lst;
    agg->count = 0;
    agg->alloc = DEFAULT_TARGET_LEN;

    return agg;
}

void
httpAggDestroy(http_agg_t **http_agg_ptr)
{
    if (!http_agg_ptr || !*http_agg_ptr) return;

    // free all contents first
    http_agg_t* http_agg = *http_agg_ptr;
    httpAggReset(http_agg);

    free(http_agg->target);
    free(http_agg);

    *http_agg_ptr = NULL;
}

// accessor for event; return the string value for the name field
static const char *
str_value(event_t *evt, const char *name)
{
    if (!evt || !evt->fields) return NULL;

    event_field_t *field;
    for (field = evt->fields; field->value_type != FMT_END; field++) {
        if ((!strcmp(field->name, name)) &&
            (field->value_type == FMT_STR)) {
            return field->value.str;
        }
    }
    return NULL;
}

// accessor for event; return the numeric value for the name field
static long long
num_value(event_t *evt, const char *name)
{
    if (!evt || !evt->fields) return LLONG_MIN;

    event_field_t *field;
    for (field = evt->fields; field->value_type != FMT_END; field++) {
        if ((!strcmp(field->name, name)) &&
            (field->value_type == FMT_NUM)) {
            return field->value.num;
        }
    }
    return LLONG_MIN;
}

static target_agg_t *
get_target_entry(http_agg_t *http_agg, const char* target_val)
{
    if (!http_agg || !target_val) return NULL;

    // per rfc3986: query strings start with a '?'
    // https://example.com/over/there?name=ferret
    // if a target_val has a query string ignore that part of the uri.
    // This is done as just one small way to manage the cardiality.
    char *temp_uri = strdup(target_val);
    if (!temp_uri) {
        DBG(NULL);
        return NULL;
    }
    char *query_ptr = strchr(temp_uri, '?');
    if (query_ptr) *query_ptr = '\0';


    // look to see if target already exists in list
    // if so, return a pointer to it.
    int i;
    for (i=0; i<http_agg->count; i++) {
        if (!strcmp(http_agg->target[i]->uri, temp_uri)) {
            free(temp_uri);
            return http_agg->target[i];
        }
    }

    // if not, and we're out of room, realloc
    if (http_agg->count >= http_agg->alloc) {
        uint64_t new_size = http_agg->alloc << 2; // same as multiplying by 4
        target_agg_t **temp_target = realloc(http_agg->target, sizeof(*temp_target) * new_size);
        if (!temp_target) {
            free(temp_uri);
            DBG(NULL);
            return NULL;
        }
        memset(&temp_target[http_agg->count], 0,
               sizeof(*temp_target) * (new_size - http_agg->alloc));
        http_agg->target = temp_target;
        http_agg->alloc = new_size;
    }

    // Now create the new target entry
    target_agg_t *temp_target = calloc(1, sizeof(*temp_target));
    if (!temp_target) {
        free(temp_uri);
        DBG(NULL);
        return NULL;
    }

    // Add the new target entry
    temp_target->uri = temp_uri;
    http_agg->target[http_agg->count++] = temp_target;

    return temp_target;
}

static void
add_counter(agg_counter_t *counter, long long value)
{
    if (!counter) return;
    counter->total += value;
    counter->num_entries++;
}

static void
add_status(target_agg_t *entry, long long value)
{
    // rfc7231 defines 41 possible values from 100 to 505
    // This supports all of these without depending on the exact definition
    if (value < 0 || value > 1000) {
        DBG("%lld", value);
        return;
    }

    int i;
    for (i=0; i<MAX_CODE_ENTRIES; i++) {
        if (entry->status[i].code == value) {
            // the code already exists, increment the count for it
            entry->status[i].count++;
            break;
        }
        if (entry->status[i].code == 0) {
            // add code to the end of the list and increment count
            entry->status[i].code = value;
            entry->status[i].count++;
            break;
        }
    }
}

void
httpAggAddMetric(http_agg_t *http_agg,
                 event_t *duration,
                 size_t request_len, size_t response_len)
{
    if (!http_agg || !duration) return;

    // Aggregation is keyed by target (uri).  Get this from the duration event.
    const char *target_val = str_value(duration, "http_target");
    if (!target_val) return;

    target_agg_t *target_entry = get_target_entry(http_agg, target_val);
    if (!target_entry) return;

    // Record the status in the target_entry
    long long status_val = num_value(duration, "http_status_code");
    add_status(target_entry, status_val);

    // Record the field data in the target_entry
    counter_field_enum dur_field = strToVal(fieldMapIn, duration->name);
    switch (dur_field) {
        case SERVER_DURATION:
        case CLIENT_DURATION:
            if (duration->value.type == FMT_INT) {
                add_counter(&target_entry->field[dur_field], duration->value.integer);
            } else {
                DBG(NULL);
            }
            break;
        default:
            DBG(NULL);
    }
    if (request_len != -1) {
        add_counter(&target_entry->field[REQUEST_BYTES], request_len);
    }
    if (response_len != -1) {
        add_counter(&target_entry->field[RESPONSE_BYTES], response_len);
    }
}

static void
report_target(mtc_t *mtc, target_agg_t *target)
{
    {
        int i;
        for (i=0; i<MAX_CODE_ENTRIES; i++) {
            if (target->status[i].code == 0) break;

            event_field_t fields[] = {
                STRFIELD("http_target", target->uri, 4, TRUE),
                NUMFIELD("http_status_code", target->status[i].code, 1, TRUE),
                STRFIELD("proc",        g_proc.procname, 4, TRUE),
                NUMFIELD("pid",         g_proc.pid,      4, TRUE),
                STRFIELD("host",        g_proc.hostname, 4, TRUE),
                STRFIELD("unit",        "request", 4, TRUE),
                FIELDEND
            };
            event_t metric = INT_EVENT("http.requests",
                                       target->status[i].count, DELTA, fields);
            cmdSendMetric(mtc, &metric);
        }
    }

    {
        counter_field_enum i;
        for (i = SERVER_DURATION; i < FIELD_MAX; i++) {
            if (target->field[i].num_entries == 0) continue;
            char *unit;
            data_type_t metric_type;

            if ((i == SERVER_DURATION) || (i == CLIENT_DURATION)) {
                unit = "millisecond";
                metric_type = DELTA_MS;
            } else {
                unit = "byte";
                metric_type = DELTA;
            }

            event_field_t fields[] = {
                STRFIELD("http_target", target->uri,     4, TRUE),
                NUMFIELD("numops",      target->field[i].num_entries, 8, TRUE),
                STRFIELD("proc",        g_proc.procname, 4, TRUE),
                NUMFIELD("pid",         g_proc.pid,      4, TRUE),
                STRFIELD("host",        g_proc.hostname, 4, TRUE),
                STRFIELD("unit",        unit, 4, TRUE),
                FIELDEND
            };
            event_t metric = INT_EVENT(valToStr(fieldMapOut, i),
                                       target->field[i].total, metric_type, fields);
            cmdSendMetric(mtc, &metric);
        }
    }
}

void
httpAggSendReport(http_agg_t *http_agg, mtc_t *mtc)
{
    if (!http_agg || !mtc) return;

    int i;
    for (i=0; i<http_agg->count; i++) {
        target_agg_t *target = http_agg->target[i];
        report_target(mtc, target);
    }
}

void
httpAggReset(http_agg_t *http_agg)
{
    if (!http_agg) return;

    int i;
    for (i=0; i<http_agg->count; i++) {
        target_agg_t *target = http_agg->target[i];
        if (target) {
            if (target->uri) free(target->uri);
            free(target);
        }
        http_agg->target[i] = NULL;
    }
    http_agg->count = 0;
}


