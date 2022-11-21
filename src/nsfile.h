#ifndef __NSFILE_H__
#define __NSFILE_H__

#include "scopestdlib.h"
#include "scopetypes.h"


/*
* API in this module provides a set of wrappers which allows to perform
* requested file operation in a following way:
*
* - set effective GID to the container GID
* - set effective UID to the container UID
* - perform requested file operation
* - restore effective UID to the original UID
* - restore effective GID to the original GID
*
* The scheme above is required since extracting the files (creating files) to the container context
* are based on the file permissions from container namespace mount perspective. The UID and GID
* in container namespace can differs from host settings, (see uid_map gid_map for details).
* We need to adjust loader and library operations to above to created files have proper permissions.
*
* On other hand operations like attach (`ptrace` in particular) required to be performed
* while having original effective UID and GID - root from the host.
*/

int nsFileShmOpen(const char *, int, mode_t , uid_t, gid_t, uid_t, gid_t);
int nsFileOpen(const char *, int, uid_t, gid_t, uid_t, gid_t);
int nsFileOpenWithMode(const char *, int, mode_t, uid_t, gid_t, uid_t, gid_t);
int nsFileMkdir(const char *, mode_t, uid_t, gid_t, uid_t, gid_t);
int nsFileMksTemp(char *, uid_t, gid_t, uid_t, gid_t);
int nsFileRename(const char *, const char *, uid_t, gid_t, uid_t, gid_t);
FILE *nsFileFopen(const char *, const char *, uid_t, gid_t, uid_t, gid_t);
int nsFileSymlink(const char *, const char *, uid_t, gid_t, uid_t, gid_t, int *);

#endif // __NSFILE_H__
