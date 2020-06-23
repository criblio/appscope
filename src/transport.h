#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__
#include "scopetypes.h"

typedef struct _transport_t transport_t;

// Constructors Destructors
transport_t*        transportCreateUdp(const char* host, const char* port);
transport_t*        transportCreateTCP(const char* host, const char* port);
transport_t*        transportCreateFile(const char* path, cfg_buffer_t buf);
transport_t*        transportCreateUnix(const char* path);
transport_t*        transportCreateSyslog(void);
transport_t*        transportCreateShm();
void                transportDestroy(transport_t**);

// Accessors
int                 transportSend(transport_t*, const char* msg);
int                 transportFlush(transport_t*);
int                 transportNeedsConnection(transport_t *);
int                 transportConnect(transport_t*);
int                 transportConnection(transport_t *);
int                 transportDisconnect(transport_t *);
cfg_transport_t     transportType(transport_t *);

#endif // __TRANSPORT_H__
