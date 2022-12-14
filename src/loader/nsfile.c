#define _GNU_SOURCE
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>

#include "nsfile.h"
#include "scopestdlib.h"

int
nsFileShmOpen(const char *name, int oflag, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    setegid(nsGid);
    seteuid(nsUid);

    int fd = shm_open(name, oflag, mode);

    seteuid(restoreUid);
    setegid(restoreGid);
    return fd;
}

int
nsFileOpen(const char *pathname, int flags, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {

    setegid(nsGid);
    seteuid(nsUid);

    int fd = open(pathname, flags);

    seteuid(restoreUid);
    setegid(restoreGid);
    return fd;
}

int
nsFileOpenWithMode(const char *pathname, int flags, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {

    setegid(nsGid);
    seteuid(nsUid);

    int fd = open(pathname, flags, mode);

    seteuid(restoreUid);
    setegid(restoreGid);
    return fd;
}

int
nsFileMkdir(const char *pathname, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    setegid(nsGid);
    seteuid(nsUid);

    int res = mkdir(pathname, mode);

    seteuid(restoreUid);
    setegid(restoreGid);
    return res;
}

int
nsFileMksTemp(char *template, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    setegid(nsGid);
    seteuid(nsUid);

    int res = mkstemp(template);

    seteuid(restoreUid);
    setegid(restoreGid);
    return res;
}

int
nsFileRename(const char *oldpath, const char *newpath, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    setegid(nsGid);
    seteuid(nsUid);

    int res = rename(oldpath, newpath);

    seteuid(restoreUid);
    setegid(restoreGid);
    return res;
}

FILE *
nsFileFopen(const char *restrict pathname, const char *restrict mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    setegid(nsGid);
    seteuid(nsUid);

    FILE *fp = fopen(pathname, mode);

    seteuid(restoreUid);
    setegid(restoreGid);
    return fp;
}

int
nsFileSymlink(const char *target, const char *linkpath, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid, int *errnoVal) {
    setegid(nsGid);
    seteuid(nsUid);

    int res = symlink(target, linkpath);
    if (res == -1) {
        *errnoVal = errno;
    }

    seteuid(restoreUid);
    setegid(restoreGid);
    return res;
}
