#define _GNU_SOURCE

#include "nsfile.h"
#include "scopestdlib.h"

int
nsFileShmOpen(const char *name, int oflag, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int fd = scope_shm_open(name, oflag, mode);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return fd;
}

int
nsFileOpen(const char *pathname, int flags, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {

    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int fd = scope_open(pathname, flags);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return fd;
}

int
nsFileOpenWithMode(const char *pathname, int flags, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {

    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int fd = scope_open(pathname, flags, mode);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return fd;
}

int
nsFileMkdir(const char *pathname, mode_t mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int res = scope_mkdir(pathname, mode);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return res;
}

int
nsFileMksTemp(char *template, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int res = scope_mkstemp(template);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return res;
}

int
nsFileRename(const char *oldpath, const char *newpath, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int res = scope_rename(oldpath, newpath);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return res;
}

int nsFileRemove(const char *path, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid, int *errnoVal) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int res = scope_remove(path);
    if (res == -1) {
        *errnoVal = scope_errno;
    }

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return res;
}

FILE *
nsFileFopen(const char *restrict pathname, const char *restrict mode, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    FILE *fp = scope_fopen(pathname, mode);

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return fp;
}

int
nsFileSymlink(const char *target, const char *linkpath, uid_t nsUid, gid_t nsGid, uid_t restoreUid, gid_t restoreGid, int *errnoVal) {
    scope_setegid(nsGid);
    scope_seteuid(nsUid);

    int res = scope_symlink(target, linkpath);
    if (res == -1) {
        *errnoVal = scope_errno;
    }

    scope_seteuid(restoreUid);
    scope_setegid(restoreGid);
    return res;
}
