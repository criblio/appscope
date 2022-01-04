#ifndef __METRIC_H__
#define __METRIC_H__

#include "report.h"
#include "state.h"
#include "state_private.h"

void initMetricCapture(void);
bool doMetricCapture(uint64_t, int, net_info*, char*, size_t,
                     metric_t, src_data_t);
void reportAllCapturedMetrics(void);

typedef struct {
    unsigned char *name;
    unsigned char *value;
    unsigned char *type;
    unsigned char *dims;
} captured_metric_t;

#endif // __METRIC_H__
