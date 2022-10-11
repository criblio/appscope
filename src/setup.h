#ifndef __SETUP_H__
#define __SETUP_H__

#include "scopetypes.h"

service_status_t setupService(const char *);
int setupConfigure(void *, size_t);
char *setupLoadFileIntoMem(size_t *, const char *);

#endif // __SETUP_H__
