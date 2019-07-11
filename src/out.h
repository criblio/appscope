#ifndef __OUT_H__
#define __OUT_H__
#include "transport.h"

typedef struct _out_t out_t;

// Constructors Destructors
out_t*              outCreate();
void                outDestroy(out_t**);

// Accessors
int                 outSend(out_t*, char* msg);
char*               outStatsDPrefix(out_t*);

// Setters (modifies out_t, but does not persist modifications)
void                outSetTransport(out_t*, transport_t*);
void                outStatsDPrefixSet(out_t*, char*);

#endif // __OUT_H__

