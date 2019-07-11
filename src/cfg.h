#ifndef __CFG_H__
#define __CFG_H__
#include "scopetypes.h"

typedef struct _config_t config_t;

// Constructors Destructors
config_t*           cfgCreateDefault();
config_t*           cfgRead(char* path);        // reads config from yaml file
void                cfgDestroy(config_t**);

// Accessors
cfg_out_format_t    cfgOutFormat(config_t*);
char*               cfgOutStatsDPrefix(config_t*);
cfg_transport_t     cfgTransportType(config_t*, which_transport_t);
char*               cfgTransportHost(config_t*, which_transport_t);
int                 cfgTransportPort(config_t*, which_transport_t);
char*               cfgTransportPath(config_t*, which_transport_t);
char**              cfgFuncFilters(config_t*);
int                 cfgFuncIsFiltered(config_t*, char* funcName);
cfg_log_level_t     cfgLogLevel(config_t*);


// Setters (modifies config_t, but does not persist modifications)
void                cfgOutFormatSet(config_t*, cfg_out_format_t);
void                cfgOutStatsDPrefixSet(config_t*, char*);
void                cfgTransportTypeSet(config_t*, which_transport_t, cfg_transport_t);
void                cfgTransportHostSet(config_t*, which_transport_t, char*);
void                cfgTransportPortSet(config_t*, which_transport_t, int);
void                cfgTransportPathSet(config_t*, which_transport_t, char*);
void                cfgFuncFiltersAdd(config_t*, char*);
void                cfgLogLevelSet(config_t*, cfg_log_level_t);

#endif // __CFG_H__
