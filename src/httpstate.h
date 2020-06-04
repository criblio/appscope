#ifndef __HTTPSTATE_H__
#define __HTTPSTATE_H__

#include "report.h"
#include "state.h"
#include "state_private.h"

void initHttpState(void);
bool doHttp(uint64_t, int, net_info*, char*, size_t, metric_t, src_data_t);

#endif // __HTTPSTATE_H__
