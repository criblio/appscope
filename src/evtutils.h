#ifndef __EVTUTILS_H__
#define __EVTUTILS_H__

#include "report.h"
#include "state.h"
#include "state_private.h"

protocol_info * evtProtoCreate(void);
bool evtProtoDelete(protocol_info *proto);

bool evtDelete(evt_type *event);



#endif // __EVTUTILS_H__
