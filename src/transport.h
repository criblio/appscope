#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__
#include "scopetypes.h"

typedef struct _transport_t transport_t;

// Constructors Destructors
transport_t*        transportCreateUdp(const char *, const char *);
transport_t*        transportCreateTCP(const char *, const char *);
transport_t*        transportCreateFile(const char *, cfg_buffer_t);
transport_t*        transportCreateUnix(const char *);
transport_t*        transportCreateSyslog(void);
transport_t*        transportCreateShm(void);
void                transportDestroy(transport_t **);

// Supplemental configuration
void                transportConfigureTls(transport_t *,
                          unsigned int, unsigned int, const char*);

// Accessors
int                 transportSend(transport_t *, const char *, size_t);
int                 transportFlush(transport_t *);
int                 transportNeedsConnection(transport_t *);
int                 transportConnect(transport_t *);
int                 transportConnection(transport_t *);
int                 transportDisconnect(transport_t *);
int                 transportReconnect(transport_t *);
cfg_transport_t     transportType(transport_t *);
int                 transportSetFD(int, transport_t *);

#endif // __TRANSPORT_H__
