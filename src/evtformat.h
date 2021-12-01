#ifndef __EVT_FORMAT_H__
#define __EVT_FORMAT_H__
#include "cJSON.h"
#include "mtcformat.h"
#include "pcre2posix.h"
#include <stdint.h>

typedef struct _evt_fmt_t evt_fmt_t;

// Constructors Destructors
evt_fmt_t *evtFormatCreate();
void evtFormatDestroy(evt_fmt_t **);

// Accessors
regex_t *evtFormatValueFilter(evt_fmt_t *, watch_t);
regex_t *evtFormatFieldFilter(evt_fmt_t *, watch_t);
regex_t *evtFormatNameFilter(evt_fmt_t *, watch_t);
unsigned evtFormatSourceEnabled(evt_fmt_t *, watch_t);
unsigned evtFormatRateLimit(evt_fmt_t *);
custom_tag_t **evtFormatCustomTags(evt_fmt_t *);

// These are the exposed functions that are expected to be used externally
cJSON *evtFormatMetric(evt_fmt_t *, event_t *, uint64_t, proc_id_t *);
cJSON *evtFormatHttp(evt_fmt_t *, event_t *, uint64_t, proc_id_t *);

// Could be static; these are lower level funcs only exposed for testing
cJSON *fmtMetricJson(event_t *, regex_t *, watch_t);
cJSON *fmtEventJson(evt_fmt_t *, event_format_t *);

// Setters (modifies evt_fmt_t, but does not persist modifications)
void evtFormatValueFilterSet(evt_fmt_t *, watch_t, const char *);
void evtFormatFieldFilterSet(evt_fmt_t *, watch_t, const char *);
void evtFormatNameFilterSet(evt_fmt_t *, watch_t, const char *);
void evtFormatSourceEnabledSet(evt_fmt_t *, watch_t, unsigned);
void evtFormatRateLimitSet(evt_fmt_t *, unsigned);
void evtFormatCustomTagsSet(evt_fmt_t *, custom_tag_t **);

#endif // __EVT_FORMAT_H__
