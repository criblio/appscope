#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

typedef enum {CFG_EXPANDED_STATSD, CFG_NEWLINE_DELIMITED, CFG_FORMAT_MAX} cfg_out_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM} cfg_transport_t;
typedef enum {CFG_OUT, CFG_LOG, CFG_WHICH_MAX} which_transport_t;
typedef enum {CFG_LOG_TRACE,
              CFG_LOG_DEBUG,
              CFG_LOG_INFO,
              CFG_LOG_WARN,
              CFG_LOG_ERROR,
              CFG_LOG_NONE} cfg_log_level_t;

#define CFG_MAX_VERBOSITY 9
#define CFG_FILE_NAME "scope.cfg"

#define DEFAULT_OUT_FORMAT CFG_EXPANDED_STATSD
#define DEFAULT_STATSD_MAX_LEN 512
#define DEFAULT_STATSD_PREFIX ""
#define DEFAULT_CUSTOM_TAGS NULL
#define DEFAULT_OUT_VERBOSITY 4
#define DEFAULT_COMMAND_PATH "/tmp"
#define DEFAULT_LOG_LEVEL CFG_LOG_ERROR
#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_FD 999
#define DEFAULT_MIN_FD 200
#define DEFAULT_BADFD -2

#endif // __SCOPETYPES_H__

