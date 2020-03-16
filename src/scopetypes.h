#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

#include <unistd.h>

typedef enum {CFG_METRIC_STATSD,
              CFG_METRIC_JSON,
              CFG_EVENT_ND_JSON,
              CFG_FORMAT_MAX} cfg_mtc_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM, CFG_TCP} cfg_transport_t;
typedef enum {CFG_MTC, CFG_CTL, CFG_LOG, CFG_WHICH_MAX} which_transport_t;
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
              CFG_SRC_MAX} watch_t;


#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128
#define DEFAULT_CMD_SIZE 32
#define MAX_ID 512

typedef struct
{
    pid_t pid;
    pid_t ppid;
    char hostname[MAX_HOSTNAME];
    char procname[MAX_PROCNAME];
    char *cmd;
    char id[MAX_ID];
} proc_id_t;


#define TRUE 1
#define FALSE 0

#ifndef bool
typedef unsigned int bool;
#endif

#define CFG_MAX_VERBOSITY 9
#define CFG_FILE_NAME "scope.yml"

#define DEFAULT_MTC_FORMAT CFG_METRIC_STATSD
#define DEFAULT_STATSD_MAX_LEN 512
#define DEFAULT_STATSD_PREFIX ""
#define DEFAULT_CUSTOM_TAGS NULL
#define DEFAULT_MTC_VERBOSITY 4
#define DEFAULT_COMMAND_DIR "/tmp"
#define DEFAULT_LOG_LEVEL CFG_LOG_ERROR
#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_FD 999
#define DEFAULT_MIN_FD 200
#define DEFAULT_CTL_FORMAT CFG_EVENT_ND_JSON
#define DEFAULT_SRC_FILE_VALUE ".*"
#define DEFAULT_SRC_CONSOLE_VALUE ".*"
#define DEFAULT_SRC_SYSLOG_VALUE ".*"
#define DEFAULT_SRC_METRIC_VALUE ".*"
#define DEFAULT_SRC_FILE_FIELD ".*"
#define DEFAULT_SRC_CONSOLE_FIELD ".*"
#define DEFAULT_SRC_SYSLOG_FIELD ".*"
#define DEFAULT_SRC_METRIC_FIELD "^[^h]+"
#define DEFAULT_SRC_FILE_NAME ".*log.*"
#define DEFAULT_SRC_CONSOLE_NAME "(stdout)|(stderr)"
#define DEFAULT_SRC_SYSLOG_NAME ".*"
#define DEFAULT_SRC_METRIC_NAME ".*"
#define DEFAULT_MTC_IPPORT_VERBOSITY 6
#define DEFAULT_SRC_FILE 0
#define DEFAULT_SRC_CONSOLE 0
#define DEFAULT_SRC_SYSLOG 0
#define DEFAULT_SRC_METRIC 0
#define DEFAULT_MTC_PORT "8125"
#define DEFAULT_CTL_PORT "9109"
#define MAXEVENTSPERSEC 10000
#define DEFAULT_PORTBLOCK 0
#define DEFAULT_METRIC_CBUF_SIZE 50 * 1024

/*
 * This calculation is not what we need in the long run.
 * Not all events are rate limited; only metric events at this point.
 * The correct size is empirical at the moment. This calculation
 * results in a value large enough to support what we are aware
 * of as requirements. SO, we'll extend this over time.
 */
#define DEFAULT_CBUF_SIZE (MAXEVENTSPERSEC * DEFAULT_SUMMARY_PERIOD)
#define DEFAULT_CONFIG_SIZE 30 * 1024

#endif // __SCOPETYPES_H__

