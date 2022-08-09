#ifndef __NS_H__
#define __NS_H__

#include "scopetypes.h"

bool nsIsPidInChildNs(pid_t, pid_t *);
int nsForkAndExec(pid_t, pid_t);

#endif // __NS_H__
