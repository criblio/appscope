#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

typedef enum {CFG_METRIC_STATSD,
              CFG_METRIC_JSON,
              CFG_EVENT_ND_JSON,
              CFG_FORMAT_MAX} cfg_out_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM, CFG_TCP} cfg_transport_t;
typedef enum {CFG_OUT, CFG_EVT, CFG_LOG, CFG_WHICH_MAX} which_transport_t;
typedef enum {CFG_LOG_TRACE,
              CFG_LOG_DEBUG,
              CFG_LOG_INFO,
              CFG_LOG_WARN,
              CFG_LOG_ERROR,
              CFG_LOG_NONE} cfg_log_level_t;
typedef enum {CFG_BUFFER_FULLY, CFG_BUFFER_LINE} cfg_buffer_t;
typedef enum {CFG_SRC_FILE,
              CFG_SRC_CONSOLE,
              CFG_SRC_SYSLOG,
              CFG_SRC_METRIC,
              CFG_SRC_MAX} cfg_evt_t;


#define CFG_MAX_VERBOSITY 9
#define CFG_FILE_NAME "scope.yml"

#define DEFAULT_OUT_FORMAT CFG_METRIC_STATSD
#define DEFAULT_STATSD_MAX_LEN 512
#define DEFAULT_STATSD_PREFIX ""
#define DEFAULT_CUSTOM_TAGS NULL
#define DEFAULT_OUT_VERBOSITY 4
#define DEFAULT_COMMAND_DIR "/tmp"
#define DEFAULT_LOG_LEVEL CFG_LOG_ERROR
#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_FD 999
#define DEFAULT_MIN_FD 200
#define DEFAULT_BADFD -2
#define DEFAULT_EVT_FORMAT CFG_EVENT_ND_JSON
#define DEFAULT_SRC_FILE 0
#define DEFAULT_SRC_CONSOLE 0
#define DEFAULT_SRC_SYSLOG 0
#define DEFAULT_SRC_METRIC 0
#define DEFAULT_OUT_PORT "8125"
#define DEFAULT_EVT_PORT "9109"
#define DEFAULT_LOG_FILE_FILTER ".*log.*"
#define DEFAULT_METRIC_NAME_FILTER ".*"
#define DEFAULT_METRIC_VALUE_FILTER ".*"
#define DEFAULT_METRIC_FIELD_FILTER ".*host.*"
#define DEFAULT_CBUF_SIZE 5000

#endif // __SCOPETYPES_H__

