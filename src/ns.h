#ifndef __NS_H__
#define __NS_H__

#include "scopetypes.h"

bool nsIsPidInChildNs(pid_t, pid_t *);
int nsForkAndExec(pid_t, pid_t);
int nsConfigure(pid_t, void *, size_t);
service_status_t nsService(pid_t, const char *);
int nsHostStart(void);

#endif // __NS_H__
