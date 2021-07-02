#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

#include <unistd.h>

typedef enum {CFG_FMT_STATSD,
              CFG_FMT_NDJSON,
              CFG_FORMAT_MAX} cfg_mtc_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM, CFG_TCP} cfg_transport_t;
typedef enum {CFG_MTC, CFG_CTL, CFG_LOG, CFG_LS, CFG_WHICH_MAX} which_transport_t;
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
              CFG_SRC_HTTP,
              CFG_SRC_NET,
              CFG_SRC_FS,
              CFG_SRC_DNS,
              CFG_SRC_MAX} watch_t;
typedef enum {CFG_LOGSTREAM_NONE,
              CFG_LOGSTREAM,
              CFG_LOGSTREAM_CLOUD,
              CFG_LOGSTREAM_MAX} cfg_logstream_t;

#define ROUND_DOWN(num, unit) ((num) & ~((unit) - 1))
#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))

#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128
#define DEFAULT_CMD_SIZE 32
#define MAX_ID 512
#define MAX_CGROUP 512
#define MODE_STR 16

typedef struct
{
    pid_t pid;
    pid_t ppid;
    uid_t uid;
    gid_t gid;
    char hostname[MAX_HOSTNAME];
    char procname[MAX_PROCNAME];
    char *cmd;
    char id[MAX_ID];
    char cgroup[MAX_CGROUP];
} proc_id_t;

#define TRUE 1
#define FALSE 0

#ifndef bool
typedef unsigned int bool;
#endif

#define CFG_MAX_VERBOSITY 9
#define CFG_FILE_NAME "scope.yml"
#define PROTOCOL_FILE_NAME "scope_protocol.yml"

#define DEFAULT_MTC_ENABLE TRUE
#define DEFAULT_MTC_FORMAT CFG_FMT_STATSD
#define DEFAULT_STATSD_MAX_LEN 512
#define DEFAULT_STATSD_PREFIX ""
#define DEFAULT_CUSTOM_TAGS NULL
#define DEFAULT_NUM_TAGS 8
#define DEFAULT_MTC_VERBOSITY 4
#define DEFAULT_COMMAND_DIR "/tmp"
#define DEFAULT_LOG_LEVEL CFG_LOG_WARN
#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_FD 999
#define DEFAULT_MIN_FD 200
#define DEFAULT_EVT_ENABLE TRUE
#define DEFAULT_CTL_FORMAT CFG_FMT_NDJSON
#define DEFAULT_SRC_FILE_VALUE ".*"
#define DEFAULT_SRC_CONSOLE_VALUE ".*"
#define DEFAULT_SRC_SYSLOG_VALUE ".*"
#define DEFAULT_SRC_METRIC_VALUE ".*"
#define DEFAULT_SRC_HTTP_VALUE ".*"
#define DEFAULT_SRC_NET_VALUE ".*"
#define DEFAULT_SRC_FS_VALUE ".*"
#define DEFAULT_SRC_DNS_VALUE ".*"
#define DEFAULT_SRC_FILE_FIELD ".*"
#define DEFAULT_SRC_CONSOLE_FIELD ".*"
#define DEFAULT_SRC_SYSLOG_FIELD ".*"
#define DEFAULT_SRC_METRIC_FIELD "^[^h]+"
#define DEFAULT_SRC_HTTP_FIELD ".*"
#define DEFAULT_SRC_NET_FIELD ".*"
#define DEFAULT_SRC_FS_FIELD ".*"
#define DEFAULT_SRC_DNS_FIELD ".*"
#define DEFAULT_SRC_FILE_NAME ".*log.*"
#define DEFAULT_SRC_CONSOLE_NAME "(stdout)|(stderr)"
#define DEFAULT_SRC_SYSLOG_NAME ".*"
#define DEFAULT_SRC_METRIC_NAME ".*"
#define DEFAULT_SRC_HTTP_NAME ".*"
#define DEFAULT_SRC_HTTP_HEADER NULL
#define DEFAULT_SRC_NET_NAME ".*"
#define DEFAULT_SRC_FS_NAME ".*"
#define DEFAULT_SRC_DNS_NAME ".*"
#define DEFAULT_MTC_IPPORT_VERBOSITY 1

#define DEFAULT_SRC_FILE TRUE
#define DEFAULT_SRC_CONSOLE TRUE
#define DEFAULT_SRC_SYSLOG FALSE
#define DEFAULT_SRC_METRIC FALSE
#define DEFAULT_SRC_HTTP TRUE
#define DEFAULT_SRC_NET TRUE
#define DEFAULT_SRC_FS TRUE
#define DEFAULT_SRC_DNS TRUE

#define DEFAULT_MAXEVENTSPERSEC 100000
#define DEFAULT_ENHANCE_FS TRUE
#define DEFAULT_PORTBLOCK 0
#define DEFAULT_METRIC_CBUF_SIZE 50 * 1024
#define DEFAULT_PROCESS_START_MSG TRUE
#define DEFAULT_PAYLOAD_ENABLE FALSE
#define DEFAULT_PAYLOAD_DIR "/tmp"

#define DEFAULT_MTC_TYPE CFG_UDP
#define DEFAULT_MTC_HOST "127.0.0.1"
#define DEFAULT_MTC_PORT "8125"
#define DEFAULT_MTC_PATH NULL
#define DEFAULT_MTC_BUF CFG_BUFFER_LINE
#define DEFAULT_CTL_TYPE CFG_TCP
#define DEFAULT_CTL_HOST "127.0.0.1"
#define DEFAULT_CTL_PORT "9109"
#define DEFAULT_CTL_PATH NULL
#define DEFAULT_CTL_BUF CFG_BUFFER_LINE
#define DEFAULT_LOG_TYPE CFG_FILE
#define DEFAULT_LS_TYPE CFG_TCP
#define DEFAULT_LS_HOST "127.0.0.1"
#define DEFAULT_LS_PORT "10090"
#define DEFAULT_LS_PATH NULL
#define DEFAULT_LS_BUF CFG_BUFFER_LINE
#define DEFAULT_LOG_HOST NULL
#define DEFAULT_LOG_PORT NULL
#define DEFAULT_LOG_PATH "/tmp/scope.log"
#define DEFAULT_LOG_BUF CFG_BUFFER_LINE
#define DEFAULT_TLS_ENABLE FALSE
#define DEFAULT_TLS_VALIDATE_SERVER TRUE
#define DEFAULT_TLS_CA_CERT NULL

#define DEFAULT_LOGSTREAM CFG_LOGSTREAM_NONE
#define DEFAULT_LOGSTREAM_LOGMSG "The following settings have been overridden by a LogStream connection: event, metric and payload transport, "

/*
 * This calculation is not what we need in the long run.
 * Not all events are rate limited; only metric events at this point.
 * The correct size is empirical at the moment. This calculation
 * results in a value large enough to support what we are aware
 * of as requirements. SO, we'll extend this over time.
 */
#define DEFAULT_CBUF_SIZE (DEFAULT_MAXEVENTSPERSEC * DEFAULT_SUMMARY_PERIOD)
#define DEFAULT_PAYLOAD_RING_SIZE 10000
#define DEFAULT_CONFIG_SIZE 30 * 1024

// Unpublished scope env vars that are not processed by config:
//    SCOPE_APP_TYPE                 internal use only
//    SCOPE_EXEC_TYPE                internal use only
//    SCOPE_EXECVE                   "false" disables scope of child procs
//    SCOPE_EXEC_PATH                specifies path to ldscope executable
//    SCOPE_CRIBL_NO_BREAKER         adds breaker property to process start message
//    SCOPE_LIB_PATH                 specifies path to libscope.so library
//    SCOPE_GO_STRUCT_PATH           for internal testing
//    SCOPE_HTTP_SERIALIZE_ENABLE    "true" adds guard for race condition
//    SCOPE_NO_SIGNAL                if defined, timer for USR2 is not set
//    SCOPE_PERF_PRESERVE            "true" processes at 10s instead of 1ms
//    SCOPE_SWITCH                   for internal go debugging
//    SCOPE_PID                      provided by library
//    SCOPE_PAYLOAD_HEADER           write payload headers to files

#define SCOPE_PID_ENV "SCOPE_PID"
#define PRESERVE_PERF_REPORTING "SCOPE_PERF_PRESERVE"

// TLS protocol refs that have been useful:
// https://tools.ietf.org/html/rfc5246
// http://blog.fourthbit.com/2014/12/23/traffic-analysis-of-an-ssl-slash-tls-session/
// https://tls13.ulfheim.net/
#define PAYLOAD_BYTESRC 8
#define PAYLOAD_REGEX "^16030[0-3].{4}0[12]"

// libmusl requires LD_LIBRARY_PATH
#define LD_LIB_ENV "LD_LIBRARY_PATH"
#define LD_LIB_DIR "libscope-v"

#endif // __SCOPETYPES_H__

