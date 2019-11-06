#ifndef __EVT_H__
#define __EVT_H__
#include <regex.h>
#include <sys/timeb.h>
#include "format.h"
#include "transport.h"
#include "dbg.h"
#include "circbuf.h"

#define JSON_DEBUG 0

typedef struct _evt_t evt_t;

// Constructors Destructors
evt_t *             evtCreate();
void                evtDestroy(evt_t **);

// Accessors
int                 evtSend(evt_t *, const char *);
int                 evtSendEvent(evt_t *, event_t *);
void                evtFlush(evt_t *);
regex_t *           evtLogFileFilter(evt_t *);
regex_t *           evtMetricNameFilter(evt_t *);
regex_t *           evtMetricValueFilter(evt_t *);
regex_t *           evtMetricFieldFilter(evt_t *);
unsigned            evtSource(evt_t *, cfg_evt_t);
int                 evtMetric(evt_t *, const char *, uint64_t, event_t *);
int                 evtLog(evt_t *, const char *, const char *, const void *, size_t, uint64_t);
int                 evtEvents(evt_t *);
int                 evtConnected(evt_t *);

// Setters (modifies evt_t, but does not persist modifications)
void                evtTransportSet(evt_t *, transport_t *);
void                evtFormatSet(evt_t *, format_t *);
void                evtLogFileFilterSet(evt_t *, const char *);
void                evtMetricNameFilterSet(evt_t *, const char *);
void                evtMetricValueFilterSet(evt_t *, const char *);
void                evtMetricFieldFilterSet(evt_t *, const char *);
void                evtSourceSet(evt_t *, cfg_evt_t, unsigned);
int                 evtConnect(evt_t *, config_t *);

#endif // __EVT_H__

