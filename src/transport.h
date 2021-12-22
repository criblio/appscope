#ifndef __TRANSPORT_H__
#define __TRANSPORT_H__
#include <stdint.h>
#include "scopetypes.h"

typedef struct _transport_t transport_t;

typedef enum {
    NO_FAIL, // No known failures
    CONN_FAIL, // Connection failure
    TLS_CERT_FAIL, // TLS certificate failure
    TLS_CONN_FAIL, // TLS connection failure
    TLS_CONTEXT_FAIL, // TLS context failure
    TLS_SESSION_FAIL, // TLS session failure
    TLS_SOCKET_FAIL, // TLS socket failure
    TLS_VERIFY_FAIL, // TLS verification failure
} net_fail_t;

// Constructors Destructors
transport_t*        transportCreateUdp(const char *, const char *);
transport_t*        transportCreateTCP(const char *, const char *,
                                       unsigned int, unsigned int, const char*);
transport_t*        transportCreateFile(const char *, cfg_buffer_t);
transport_t*        transportCreateUnix(const char *);
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
uint64_t            transportConnectAttempts(transport_t *);
net_fail_t          transportFailureReason(transport_t *);

// Misc
void                transportRegisterForExitNotification(void (*fn)(void));
#endif // __TRANSPORT_H__
