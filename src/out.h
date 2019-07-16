#ifndef __OUT_H__
#define __OUT_H__
#include "transport.h"

typedef struct _out_t out_t;

// Constructors Destructors
out_t*              outCreate();
void                outDestroy(out_t**);

// Accessors
int                 outSend(out_t*, const char* msg);
const char*         outStatsDPrefix(out_t*);

// Setters (modifies out_t, but does not persist modifications)
void                outTransportSet(out_t*, transport_t*);
void                outStatsDPrefixSet(out_t*, const char*);

#endif // __OUT_H__

