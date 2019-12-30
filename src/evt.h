#ifndef __EVT_H__
#define __EVT_H__
#include <regex.h>
#include <stdint.h>
#include "cJSON.h"
#include "format.h"

typedef struct _evt_t evt_t;

// Constructors Destructors
evt_t *             evtCreate();
void                evtDestroy(evt_t **);

// Accessors
regex_t *           evtValueFilter(evt_t *, cfg_evt_t);
regex_t *           evtFieldFilter(evt_t *, cfg_evt_t);
regex_t *           evtNameFilter(evt_t *, cfg_evt_t);
unsigned            evtSourceEnabled(evt_t *, cfg_evt_t);

cJSON *             evtMetric(evt_t *, event_t *, uint64_t, proc_id_t *);
cJSON *             evtLog(evt_t *, const char *, const void *, size_t,
                           uint64_t, proc_id_t *);

// Setters (modifies evt_t, but does not persist modifications)
void                evtFormatSet(evt_t *, format_t *);
void                evtValueFilterSet(evt_t *, cfg_evt_t, const char *);
void                evtFieldFilterSet(evt_t *, cfg_evt_t, const char *);
void                evtNameFilterSet(evt_t *, cfg_evt_t, const char *);
void                evtSourceEnabledSet(evt_t *, cfg_evt_t, unsigned);

#endif // __EVT_H__

