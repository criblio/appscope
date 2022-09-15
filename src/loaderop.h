#ifndef __LOADEROP_H__
#define __LOADEROP_H__

#include "scopetypes.h"

typedef enum {
    PATCH_FAILED,   // patch operation was failed
    PATCH_SUCCESS,  // patch operation was success
    PATCH_NO_OP,    // patch operation was not performed
} patch_status_t;

patch_status_t loaderOpPatchLibrary(const char*);
int loaderOpSetupLoader(char *);

#endif // __LOADEROP_H__
