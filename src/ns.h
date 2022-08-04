#ifndef __NS_H__
#define __NS_H__

#include "scopetypes.h"

bool nsIsPidInChildNs(pid_t pid, pid_t *nsPid);
int nsForkAndExec(pid_t parentPid, pid_t nsPid);

#endif // __NS_H__
