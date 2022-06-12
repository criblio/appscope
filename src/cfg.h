#ifndef __CFG_H__
#define __CFG_H__
#include <pcre2posix.h>

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
unsigned            cfgMtcWatchEnable(config_t *, metric_watch_t);
const char*         cfgCmdDir(config_t*);
unsigned            cfgSendProcessStartMsg(config_t*);
unsigned            cfgMtcVerbosity(config_t*);
unsigned            cfgEvtEnable(config_t*);
cfg_mtc_format_t    cfgEventFormat(config_t*);
unsigned            cfgEvtRateLimit(config_t*);
unsigned            cfgEnhanceFs(config_t*);
cfg_backtrace_t     cfgBacktrace(config_t*);
const char*         cfgBacktraceFilterFile(config_t*);
const char*         cfgEvtFormatValueFilter(config_t*, watch_t);
const char*         cfgEvtFormatFieldFilter(config_t*, watch_t);
const char*         cfgEvtFormatNameFilter(config_t*, watch_t);
unsigned            cfgEvtFormatSourceEnabled(config_t*, watch_t);
cfg_transport_t     cfgTransportType(config_t*, which_transport_t);
const char*         cfgTransportHost(config_t*, which_transport_t);
const char*         cfgTransportPort(config_t*, which_transport_t);
const char*         cfgTransportPath(config_t*, which_transport_t);
cfg_buffer_t        cfgTransportBuf(config_t*, which_transport_t);
unsigned            cfgTransportTlsEnable(config_t *, which_transport_t);
unsigned            cfgTransportTlsValidateServer(config_t *, which_transport_t);
const char*         cfgTransportTlsCACertPath(config_t *, which_transport_t);
custom_tag_t**      cfgCustomTags(config_t*);
const char*         cfgCustomTagValue(config_t*, const char*);
cfg_log_level_t     cfgLogLevel(config_t*);
unsigned int        cfgPayEnable(config_t*);
const char *        cfgPayDir(config_t*);
const char *        cfgEvtFormatHeader(config_t *, int);
unsigned            cfgEvtAllowBinaryConsole(config_t *);
unsigned            cfgLogStreamEnable(config_t *);
unsigned            cfgLogStreamCloud(config_t *);
size_t              cfgEvtFormatNumHeaders(config_t *);
regex_t *           cfgEvtFormatHeaderRe(config_t *, int);
const char *        cfgAuthToken(config_t *);

// Setters (modifies config_t, but does not persist modifications)
void                cfgMtcEnableSet(config_t*, unsigned);
void                cfgMtcFormatSet(config_t*, cfg_mtc_format_t);
void                cfgMtcStatsDPrefixSet(config_t*, const char*);
void                cfgMtcStatsDMaxLenSet(config_t*, unsigned);
void                cfgMtcPeriodSet(config_t*, unsigned);
void                cfgMtcWatchEnableSet(config_t *, unsigned, metric_watch_t);
void                cfgCmdDirSet(config_t*, const char*);
void                cfgSendProcessStartMsgSet(config_t*, unsigned);
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
void                cfgTransportTlsEnableSet(config_t *, which_transport_t, unsigned);
void                cfgTransportTlsValidateServerSet(config_t *, which_transport_t, unsigned);
void                cfgTransportTlsCACertPathSet(config_t *, which_transport_t, const char *);
void                cfgCustomTagAdd(config_t*, const char*, const char*);
void                cfgLogLevelSet(config_t*, cfg_log_level_t);
void                cfgBacktraceSet(config_t*, cfg_backtrace_t);
void                cfgBacktraceFilterFileSet(config_t*, const char*);
void                cfgPayEnableSet(config_t*, unsigned int);
void                cfgPayDirSet(config_t*, const char *);
void                cfgEvtFormatHeaderSet(config_t *, const char *);
void                cfgEvtAllowBinaryConsoleSet(config_t *, unsigned);
void                cfgLogStreamEnableSet(config_t *, unsigned);
void                cfgLogStreamCloudSet(config_t *, unsigned);
void                cfgAuthTokenSet(config_t *, const char *);

#endif // __CFG_H__
