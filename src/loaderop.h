#ifndef __LOADEROP_H__
#define __LOADEROP_H__

#include "scopetypes.h"

int loaderOpPatchLibrary(const char*);
char * loaderOpGetLoader(const char *);
int loaderOpSetLibrary(const char *);
int loaderOpSetupLoader(char *);

#endif // __LOADEROP_H__
