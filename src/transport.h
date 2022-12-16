#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__
#include <stdint.h>
#include "scopetypes.h"

typedef struct _transport_t transport_t;

// Constructors Destructors
transport_t*        transportCreateUdp(const char *, const char *);
transport_t*        transportCreateTCP(const char *, const char *,
                                       unsigned int, unsigned int, const char*);
transport_t*        transportCreateFile(const char *, cfg_buffer_t);
transport_t*        transportCreateUnix(const char *);
transport_t*        transportCreateEdge(void);
transport_t*        transportCreateSyslog(void);
transport_t*        transportCreateShm(void);
void                transportDestroy(transport_t **);

// Accessors
int                 transportSend(transport_t *, const char *, size_t);
int                 transportFlush(transport_t *);
int                 transportNeedsConnection(transport_t *);
int                 transportConnect(transport_t *);
int                 transportConnection(transport_t *);
int                 transportDisconnect(transport_t *);
int                 transportReconnect(transport_t *);
cfg_transport_t     transportType(transport_t *);
int                 transportSupportsCommandControl(transport_t *);
void                transportLogConnectionStatus(transport_t *, const char *);

// Misc
void                transportRegisterForExitNotification(void (*fn)(void));
#endif // __TRANSPORT_H__
