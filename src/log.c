#include <stddef.h>
#include <stdlib.h>
#include "log.h"

struct _log_t
{
    transport_t* transport;
    cfg_log_level_t level;
};

log_t*
logCreate()
{
    log_t* log = calloc(1, sizeof(log_t));
    if (!log) return NULL;

    log->level = DEFAULT_LOG_LEVEL;

    return log;
}

void
logDestroy(log_t** log)
{
    if (!log || !*log) return;
    log_t* l = *log;
    transportDestroy(&l->transport);
    free(l);
    *log = NULL;
}

int
logSend(log_t* log, const char* msg, cfg_log_level_t mlevel)
{
    if (!log || !msg) return -1;

    if ((log->level == CFG_LOG_NONE) || (log->level > mlevel)) return 0;
    
    return transportSend(log->transport, msg);
}

cfg_log_level_t
logLevel(log_t* log)
{
    return (log) ? log->level : DEFAULT_LOG_LEVEL;
}

void
logTransportSet(log_t* log, transport_t* transport)
{
    if (!log) return;

    // Don't leak if logTransportSet is called repeatedly
    transportDestroy(&log->transport);
    log->transport = transport;
}

void
logLevelSet(log_t* log, cfg_log_level_t level)
{
    if (!log) return;
    log->level = level;
}


