#ifndef __SETUP_H__
#define __SETUP_H__

#include "scopetypes.h"

int setupService(const char *);
int setupConfigure(void *, size_t);
char* setupLoadFileIntoMem(size_t *, char*);

#endif // __SETUP_H__
