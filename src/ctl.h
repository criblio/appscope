#ifndef __CTL_H__
#define __CTL_H__

#include "transport.h"

typedef struct _ctl_t ctl_t;

// Constructors Destructors
ctl_t *             ctlCreate();
void                ctlDestroy(ctl_t **);

void                ctlSendMsg(ctl_t *, char *);
void                ctlFlush(ctl_t *);

int                 ctlNeedsConnection(ctl_t *);
int                 ctlConnection(ctl_t *);
int                 ctlConnect(ctl_t *);
int                 ctlClose(ctl_t *);
void                ctlTransportSet(ctl_t *, transport_t *);

#endif // _CTL_H__
