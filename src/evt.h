#ifndef __EVT_H__
#define __EVT_H__
#include <regex.h>
#include "evt.h"
#include "format.h"
#include "transport.h"

typedef struct _evt_t evt_t;

// Constructors Destructors
evt_t*              evtCreate();
void                evtDestroy(evt_t**);

// Accessors
int                 evtSend(evt_t*, const char* msg);
int                 evtSendEvent(evt_t*, event_t*);
void                evtFlush(evt_t*);
regex_t*            evtLogFileFilter(evt_t*);
unsigned            evtSource(evt_t*, cfg_evt_t);


// Setters (modifies evt_t, but does not persist modifications)
void                evtTransportSet(evt_t*, transport_t*);
void                evtFormatSet(evt_t*, format_t*);
void                evtLogFileFilterSet(evt_t*, const char*);
void                evtSourceSet(evt_t*, cfg_evt_t, unsigned);


#endif // __EVT_H__

