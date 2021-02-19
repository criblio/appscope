#ifndef __RUNTIMECFG_H__
#define __RUNTIMECFG_H__

#include "cfg.h"

typedef struct rtconfig_t {
    int urls;
    int blockconn;
    config_t *staticfg;
} rtconfig;

extern rtconfig g_cfg;

#endif // __RUNTIMECFG_H__
