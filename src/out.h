#ifndef __OUT_H__
#define __OUT_H__
#include "format.h"
#include "log.h"
#include "transport.h"
#include "format.h"

typedef struct _out_t out_t;

// Constructors Destructors
out_t*              outCreate();
void                outDestroy(out_t**);

// Accessors
int                 outSend(out_t*, const char* msg);
int                 outSendEvent(out_t*, event_t*);

// Setters (modifies out_t, but does not persist modifications)
void                outTransportSet(out_t*, transport_t*);
void                outFormatSet(out_t*, format_t*);
void                outLogReferenceSet(out_t*, log_t*);

// Getters
int outTransportDescriptor(out_t *);

#endif // __OUT_H__

