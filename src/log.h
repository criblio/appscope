#ifndef __LOG_H__
#define __LOG_H__
#include "scopetypes.h"
#include "transport.h"

typedef struct _log_t log_t;

// Constructors Destructors
log_t *logCreate();
void logDestroy(log_t **);

// Accessors
int logSend(log_t *, const char *msg, cfg_log_level_t level);
cfg_log_level_t logLevel(log_t *);
void logFlush(log_t *);

// Setters (modifies log_t, but does not persist modifications)
int logNeedsConnection(log_t *);
int logConnect(log_t *);
int logConnection(log_t *);
int logDisconnect(log_t *);
int logReconnect(log_t *);
void logTransportSet(log_t *, transport_t *);
void logLevelSet(log_t *, cfg_log_level_t);

#endif // __LOG_H__
