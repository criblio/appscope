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
unsigned            cfgMtcEnable(config_t*);
cfg_mtc_format_t    cfgMtcFormat(config_t*);
const char*         cfgMtcStatsDPrefix(config_t*);
unsigned            cfgMtcStatsDMaxLen(config_t*);
unsigned            cfgMtcPeriod(config_t*);
const char*         cfgCmdDir(config_t*);
unsigned            cfgMtcVerbosity(config_t*);
unsigned            cfgEvtEnable(config_t*);
cfg_mtc_format_t    cfgEventFormat(config_t*);
unsigned            cfgEvtRateLimit(config_t*);
unsigned            cfgEnhanceFs(config_t*);
const char*         cfgEvtFormatValueFilter(config_t*, watch_t);
const char*         cfgEvtFormatFieldFilter(config_t*, watch_t);
const char*         cfgEvtFormatNameFilter(config_t*, watch_t);
unsigned            cfgEvtFormatSourceEnabled(config_t*, watch_t);
cfg_transport_t     cfgTransportType(config_t*, which_transport_t);
const char*         cfgTransportHost(config_t*, which_transport_t);
const char*         cfgTransportPort(config_t*, which_transport_t);
const char*         cfgTransportPath(config_t*, which_transport_t);
cfg_buffer_t        cfgTransportBuf(config_t*, which_transport_t);
custom_tag_t**      cfgCustomTags(config_t*);
const char*         cfgCustomTagValue(config_t*, const char*);
cfg_log_level_t     cfgLogLevel(config_t*);

// Setters (modifies config_t, but does not persist modifications)
void                cfgMtcEnableSet(config_t*, unsigned);
void                cfgMtcFormatSet(config_t*, cfg_mtc_format_t);
void                cfgMtcStatsDPrefixSet(config_t*, const char*);
void                cfgMtcStatsDMaxLenSet(config_t*, unsigned);
void                cfgMtcPeriodSet(config_t*, unsigned);
void                cfgCmdDirSet(config_t*, const char*);
void                cfgMtcVerbositySet(config_t*, unsigned);
void                cfgEvtEnableSet(config_t*, unsigned);
void                cfgEventFormatSet(config_t*, cfg_mtc_format_t);
void                cfgEvtRateLimitSet(config_t*, unsigned);
void                cfgEnhanceFsSet(config_t*, unsigned);
void                cfgEvtFormatValueFilterSet(config_t*, watch_t, const char*);
void                cfgEvtFormatFieldFilterSet(config_t*, watch_t, const char*);
void                cfgEvtFormatNameFilterSet(config_t*, watch_t, const char*);
void                cfgEvtFormatSourceEnabledSet(config_t*, watch_t, unsigned);
void                cfgTransportTypeSet(config_t*, which_transport_t, cfg_transport_t);
void                cfgTransportHostSet(config_t*, which_transport_t, const char*);
void                cfgTransportPortSet(config_t*, which_transport_t, const char*);
void                cfgTransportPathSet(config_t*, which_transport_t, const char*);
void                cfgTransportBufSet(config_t*, which_transport_t, cfg_buffer_t);
void                cfgCustomTagAdd(config_t*, const char*, const char*);
void                cfgLogLevelSet(config_t*, cfg_log_level_t);
#endif // __CFG_H__
