#ifndef __NSINFO_H__
#define __NSINFO_H__

#include <stdbool.h>
#include <unistd.h>

// Retrieve namespace information
bool nsInfoGetPidNs(pid_t, pid_t *);
bool nsInfoIsPidInSameMntNs(pid_t);
uid_t nsInfoTranslateUid(pid_t);
uid_t nsInfoTranslateUidRootDir(const char *, pid_t);
gid_t nsInfoTranslateGid(pid_t);
gid_t nsInfoTranslateGidRootDir(const char *, pid_t);

#endif // __NSINFO_H__
