#ifndef __CFGUTILS_H__
#define __CFGUTILS_H__
#include <stdio.h>
#include "cfg.h"
#include "ctl.h"
#include "log.h"
#include "mtc.h"
#include "evtformat.h"
#include "linklist.h"

// cfgPath returns a pointer to a scope_malloc()'d buffer.
// The caller is responsible for deallocating with scope_free().
char * cfgPath(void);

// reads cfg from yaml file
config_t * cfgRead(const char *);
void destroyProtEntry(void *);

// reads cfg from a string (containing json or yaml)
config_t * cfgFromString(const char *);

// constructs a cJSON object hierarchy or json string
cJSON * jsonObjectFromCfg(config_t *);
char * jsonStringFromCfg(config_t *);

// modify cfg per environment variables
void cfgProcessEnvironment(config_t *);

// modify cfg per environment variable syntax in a file
void cfgProcessCommands(config_t *, FILE *);

typedef enum {
    RULES_ERROR,           // error with rules operation
    RULES_SCOPED,          // process will be scoped
    RULES_SCOPED_WITH_CFG, // process will be scoped with cfg from the rules file
    RULES_NOT_SCOPED       // process will not be scoped
} rules_status_t;

// Rules Handling
// Parse rules file and optionally reads a cfg from rules file
rules_status_t cfgRulesStatus(const char *, const char *, const char *, config_t *);
bool cfgRulesFileIsValid(const char*);
const char *cfgRulesFilePath(void);
char *cfgRulesUnixPath(void);

log_t * initLog(config_t *);
mtc_t * initMtc(config_t *);
evt_fmt_t * initEvtFormat(config_t *);
ctl_t * initCtl(config_t *);

int cfgLogStreamDefault(config_t *);
int singleChannelSet(ctl_t *, mtc_t *);

#endif // __CFGUTILS_H__
