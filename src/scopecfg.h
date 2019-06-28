#ifndef __SCOPECFG_H__
#define __SCOPECFG_H__

typedef struct _config_t config_t;
typedef enum {CFG_EXPANDED_STATSD, CFG_NEWLINE_DELIMITED} cfg_out_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG} cfg_out_tx_t;
typedef enum {CFG_LOG_UDP, CFG_LOG_SYSLOG, CFG_LOG_SHM} cfg_log_tx_t;
typedef enum {CFG_LOG_DEBUG, 
              CFG_LOG_INFO, 
              CFG_LOG_WARN, 
              CFG_LOG_ERROR, 
              CFG_LOG_NONE} cfg_log_level_t;

// Constructors Destructors
config_t*           cfgCreateDefault();
config_t*           cfgRead(char* path);        // reads config from yaml file
void                cfgDestroy(config_t**);

// Accessors
cfg_out_format_t    cfgOutFormat(config_t*);
char*               cfgOutStatsDPrefix(config_t*);
cfg_out_tx_t        cfgOutTransportType(config_t*);
char*               cfgOutTransportHost(config_t*);
int                 cfgOutTransportPort(config_t*);
char*               cfgOutTransportPath(config_t*);
char**              cfgFuncFilters(config_t*);
int                 cfgFuncIsFiltered(config_t*, char* funcName);
int                 cfgLoggingEnabled(config_t*);
int                 cfgLogTransportEnabled(config_t*, cfg_log_tx_t);
cfg_log_level_t     cfgLogLevel(config_t*);


// Setters (modifies config_t, but does not persist modifications)
void                cfgOutFormatSet(config_t*, cfg_out_format_t);
void                cfgOutStatsDPrefixSet(config_t*, char*);
void                cfgOutTransportTypeSet(config_t*, cfg_out_tx_t);
void                cfgOutTransportHostSet(config_t*, char*);
void                cfgOutTransportPortSet(config_t*, int);
void                cfgOutTransportPathSet(config_t*, char*);
void                cfgFuncFiltersAdd(config_t*, char*);
void                cfgLogTransportEnabledSet(config_t*, cfg_log_tx_t, int);
void                cfgLogLevelSet(config_t*, cfg_log_level_t);

#endif // __SCOPECFG_H__
