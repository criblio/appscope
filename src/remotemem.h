#ifndef __REMOTEMEM_H__
#define __REMOTEMEM_H__

#include "scopestdlib.h"
#include "scopetypes.h"
#include "inttypes.h"

// Retrieves information from remote process

uint64_t remoteProcSymbolAddr(pid_t, const char *);

#endif // __REMOTEMEM_H__
