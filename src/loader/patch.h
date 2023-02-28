#ifndef __PATCH_H__
#define __PATCH_H__

#include <unistd.h>

typedef enum {
    PATCH_FAILED,   // patch operation was failed
    PATCH_SUCCESS,  // patch operation was success
    PATCH_NO_OP,    // patch operation was not performed
} patch_status_t;

patch_status_t patchLibrary(const char*);
patch_status_t patchLoader(unsigned char *, uid_t, gid_t);

#endif // __PATCH_H__
