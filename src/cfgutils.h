#ifndef __CFGUTILS_H__
#define __CFGUTILS_H__
#include "cfg.h"
#include "log.h"
#include "out.h"

// cfgPath returns a pointer to a malloc()'d buffer.
// The caller is responsible for deallocating with free().
char* cfgPath(const char* cfgname);

out_t* initOut(config_t* cfg);
log_t* initLog(config_t* cfg);


#endif // __CFGUTILS_H__
