#ifndef __CFGUTILS_H__
#define __CFGUTILS_H__
#include <stdio.h>
#include "cfg.h"
#include "log.h"
#include "out.h"
#include "evt.h"

// cfgPath returns a pointer to a malloc()'d buffer.
// The caller is responsible for deallocating with free().
char* cfgPath(void);

// reads cfg from yaml file
config_t* cfgRead(const char* path);

// modify cfg per environment variables
void cfgProcessEnvironment(config_t* cfg);

// modify cfg per environment variable syntax in a file
void cfgProcessCommands(config_t* cfg, FILE* file);

log_t* initLog(config_t* cfg);
out_t* initOut(config_t* cfg);
evt_t* initEvt(config_t* cfg);


#endif // __CFGUTILS_H__
