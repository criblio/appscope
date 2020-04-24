#ifndef __RUNTIMECFG_H__
#define __RUNTIMECFG_H__

#include "cfg.h"

typedef struct rtconfig_t {
    int urls;
    int blockconn;
    config_t *staticfg;
} rtconfig;

typedef enum
{
    PROT_START,
    HTTP1,
    HTTP2,
    REDIS,
    MONGO,
    NOSCAN,
    PROT_END
} protocol_type;

#endif // __RUNTIMECFG_H__
