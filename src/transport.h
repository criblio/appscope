#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__
#include <stdint.h>
#include "scopetypes.h"

typedef struct {
    const char *configString;       // Human-readable transport representation
    bool isConnected;               // Indicator about connection status
    uint64_t connectAttemptCount;   // Useful if isConnected is FALSE
    const char *failureString;      // May be provided when isConnected is FALSE
} transport_status_t;

typedef struct _transport_t transport_t;

// Constructors Destructors
transport_t*        transportCreateUdp(const char *, const char *);
transport_t*        transportCreateTCP(const char *, const char *,
                                       unsigned int, unsigned int, const char*);
transport_t*        transportCreateFile(const char *, cfg_buffer_t);
transport_t*        transportCreateUnix(const char *);
transport_t*        transportCreateEdge(void);
void                transportDestroy(transport_t **);

// Accessors
int                 transportSend(transport_t *, const char *, size_t);
int                 transportFlush(transport_t *);
bool                transportNeedsConnection(transport_t *);
int                 transportConnect(transport_t *);
int                 transportConnection(transport_t *);
int                 transportDisconnect(transport_t *);
int                 transportReconnect(transport_t *);
cfg_transport_t     transportType(transport_t *);
bool                transportSupportsCommandControl(transport_t *);
transport_status_t  transportConnectionStatus(transport_t *);

// Misc
void                transportRegisterForExitNotification(void (*fn)(void));

#endif // __TRANSPORT_H__
