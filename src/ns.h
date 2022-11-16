#ifndef __NS_H__
#define __NS_H__

#include "scopetypes.h"

// NS Action types
typedef enum {
    START = 0,
    STOP = 1,
} ns_action_t;

// Operation performed from host to container
int nsForkAndExec(pid_t, pid_t, char);
int nsConfigure(pid_t, void *, size_t);
service_status_t nsService(pid_t, const char *);

// Operation performed from container to host
int nsHostStart(void);
int nsHostStop(void);

#endif // __NS_H__
