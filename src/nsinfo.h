#ifndef __NSINFO_H__
#define __NSINFO_H__

#include "scopetypes.h"

// Retrieve namespace information

bool nsInfoGetPidNs(pid_t, pid_t *);
bool nsInfoIsPidInSameMntNs(pid_t);
uid_t nsInfoTranslateUid(pid_t);
gid_t nsInfoTranslateGid(pid_t);

#endif // __NSINFO_H__
