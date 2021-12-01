#ifndef __CFGUTILS_H__
#define __CFGUTILS_H__
#include "cfg.h"
#include "ctl.h"
#include "evtformat.h"
#include "linklist.h"
#include "log.h"
#include "mtc.h"
#include <stdio.h>

// cfgPath returns a pointer to a malloc()'d buffer.
// The caller is responsible for deallocating with free().
char *cfgPath(void);

// reads cfg from yaml file
config_t *cfgRead(const char *);
void destroyProtEntry(void *);

// reads cfg from a string (containing json or yaml)
config_t *cfgFromString(const char *);

// constructs a cJSON object hierarchy or json string
cJSON *jsonObjectFromCfg(config_t *);
char *jsonStringFromCfg(config_t *);

// modify cfg per environment variables
void cfgProcessEnvironment(config_t *);

// modify cfg per environment variable syntax in a file
void cfgProcessCommands(config_t *, FILE *);

log_t *initLog(config_t *);
mtc_t *initMtc(config_t *);
evt_fmt_t *initEvtFormat(config_t *);
ctl_t *initCtl(config_t *);

int cfgLogStreamDefault(config_t *);
int singleChannelSet(ctl_t *, mtc_t *);

#endif // __CFGUTILS_H__
