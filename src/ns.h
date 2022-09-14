#ifndef __NS_H__
#define __NS_H__

#include "scopetypes.h"

bool nsIsPidInSeparateMntNs(pid_t, bool *);
bool nsIsPidInChildNs(pid_t, pid_t *);
int nsForkAndExec(pid_t, pid_t);
int nsConfigure(pid_t, void*, size_t);
int nsService(pid_t, const char*);

#endif // __NS_H__
