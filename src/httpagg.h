#ifndef __HTTPREPORT_H__
#define __HTTPREPORT_H__
#include "mtc.h"

// This was written to do aggregation of http for the metrics channel (statsd)
//
// A normal lifecycle:
//   Create
//   AddMetric
//   AddMetric
//   AddMetric
//   SendReport (sends a summary of all Metrics received before it)
//   Reset (returns to a state similar to Create)

typedef struct _http_agg_t http_agg_t;

http_agg_t *httpAggCreate();
void httpAggDestroy(http_agg_t **);
void httpAggAddMetric(http_agg_t *, event_t *, size_t, size_t);
void httpAggSendReport(http_agg_t *, mtc_t *);
void httpAggReset(http_agg_t *);

#endif // __HTTPREPORT_H__
