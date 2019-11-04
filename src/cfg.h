#ifndef __CFG_H__
#define __CFG_H__
#include "scopetypes.h"

typedef struct _config_t config_t;

typedef struct {
    char* name;
    char* value;
} custom_tag_t;

// Constructors Destructors
config_t*           cfgCreateDefault();
void                cfgDestroy(config_t**);

// Accessors
cfg_out_format_t    cfgOutFormat(config_t*);
const char*         cfgOutStatsDPrefix(config_t*);
unsigned            cfgOutStatsDMaxLen(config_t*);
unsigned            cfgOutPeriod(config_t*);
const char*         cfgCmdDir(config_t*);
unsigned            cfgOutVerbosity(config_t*);
cfg_out_format_t    cfgEventFormat(config_t*);
const char*         cfgEventLogFileFilter(config_t*);
unsigned            cfgEventSource(config_t*, cfg_evt_t);
cfg_transport_t     cfgTransportType(config_t*, which_transport_t);
const char*         cfgTransportHost(config_t*, which_transport_t);
const char*         cfgTransportPort(config_t*, which_transport_t);
const char*         cfgTransportPath(config_t*, which_transport_t);
cfg_buffer_t        cfgTransportBuf(config_t*, which_transport_t);
custom_tag_t**      cfgCustomTags(config_t*);
const char*         cfgCustomTagValue(config_t*, const char*);
cfg_log_level_t     cfgLogLevel(config_t*);

// Setters (modifies config_t, but does not persist modifications)
void                cfgOutFormatSet(config_t*, cfg_out_format_t);
void                cfgOutStatsDPrefixSet(config_t*, const char*);
void                cfgOutStatsDMaxLenSet(config_t*, unsigned);
void                cfgOutPeriodSet(config_t*, unsigned);
void                cfgCmdDirSet(config_t*, const char*);
void                cfgOutVerbositySet(config_t*, unsigned);
void                cfgEventFormatSet(config_t*, cfg_out_format_t);
void                cfgEventLogFileFilterSet(config_t*, const char*);
void                cfgEventSourceSet(config_t*, cfg_evt_t, unsigned);
void                cfgTransportTypeSet(config_t*, which_transport_t, cfg_transport_t);
void                cfgTransportHostSet(config_t*, which_transport_t, const char*);
void                cfgTransportPortSet(config_t*, which_transport_t, const char*);
void                cfgTransportPathSet(config_t*, which_transport_t, const char*);
void                cfgTransportBufSet(config_t*, which_transport_t, cfg_buffer_t);
void                cfgCustomTagAdd(config_t*, const char*, const char*);
void                cfgLogLevelSet(config_t*, cfg_log_level_t);
#endif // __CFG_H__
