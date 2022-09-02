#ifndef __SETUP_H__
#define __SETUP_H__

#include "scopetypes.h"

int setupService(const char *);
int setupConfigure(void *, size_t, bool);
char* setupLoadFileIntoMem(size_t *, char*);

#endif // __SETUP_H__
