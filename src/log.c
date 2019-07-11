#include <stddef.h>
#include "log.h"

struct _log_t
{

};

log_t*
logCreate()
{
    return NULL;
}

void
logDestroy(log_t** log)
{
}

int
logSend(log_t* log, char* msg)
{
    return -1;
}

cfg_log_level_t
logLevel(log_t* log)
{
    return CFG_LOG_NONE;
}

void
logSetTransport(log_t* log, transport_t* transport)
{
}

void
logLevelSet(log_t* log, cfg_log_level_t level)
{
}


