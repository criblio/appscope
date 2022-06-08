#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

#include <unistd.h>

typedef enum {CFG_FMT_STATSD,
              CFG_FMT_NDJSON,
              CFG_FORMAT_MAX} cfg_mtc_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM, CFG_TCP, CFG_EDGE} cfg_transport_t;
typedef enum {CFG_MTC, CFG_CTL, CFG_LOG, CFG_LS, CFG_WHICH_MAX} which_transport_t;
typedef enum {CFG_LOG_TRACE,
              CFG_LOG_DEBUG,
              CFG_LOG_INFO,
              CFG_LOG_WARN,
              CFG_LOG_ERROR,
              CFG_LOG_NONE} cfg_log_level_t;
typedef enum {CFG_BACKTRACE_FULL,
              CFG_BACKTRACE_FILTER,
              CFG_BACKTRACE_OPENAT,
              CFG_BACKTRACE_NONE} cfg_backtrace_t;
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

typedef enum {CFG_MTC_FS,
              CFG_MTC_NET,
              CFG_MTC_HTTP,
              CFG_MTC_DNS,
              CFG_MTC_PROC, 
              CFG_MTC_STATSD} metric_watch_t;

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
    char *username;
    char *groupname;
} proc_id_t;

#define TRUE 1
#define FALSE 0

#ifndef bool
typedef unsigned int bool;
#endif

#define CFG_MAX_VERBOSITY 9
#define CFG_FILE_NAME "scope.yml"

#define DEFAULT_MTC_ENABLE TRUE
#define DEFAULT_MTC_FORMAT CFG_FMT_STATSD
#define DEFAULT_MTC_FS_ENABLE TRUE
#define DEFAULT_MTC_NET_ENABLE TRUE
#define DEFAULT_MTC_HTTP_ENABLE TRUE
#define DEFAULT_MTC_DNS_ENABLE TRUE
#define DEFAULT_MTC_PROC_ENABLE TRUE
#define DEFAULT_STATSD_MAX_LEN 512
#define DEFAULT_STATSD_PREFIX ""
#define DEFAULT_MTC_STATSD_ENABLE TRUE
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
#define DEFAULT_SRC_FILE_NAME "(\\/logs?\\/)|(\\.log$)|(\\.log[.\\d])"
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

#define DEFAULT_MAXEVENTSPERSEC 10000
#define DEFAULT_ENHANCE_FS TRUE
#define DEFAULT_BACKTRACE_OPTION CFG_BACKTRACE_NONE
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
#define DEFAULT_LS_TYPE CFG_EDGE
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

#define DEFAULT_LOGSTREAM_ENABLE TRUE
#define DEFAULT_LOGSTREAM_CLOUD  FALSE
#define DEFAULT_LOGSTREAM_LOGMSG "The following settings have been overridden by a LogStream connection: event, metric and payload transport, "

/*
 * This calculation is not what we need in the long run.
 * Not all events are rate limited; only metric events at this point.
 * The correct size is empirical at the moment. This calculation
 * results in a value large enough to support what we are aware
 * of as requirements. SO, we'll extend this over time.
 */
#define DEFAULT_CBUF_SIZE (DEFAULT_MAXEVENTSPERSEC * DEFAULT_SUMMARY_PERIOD)
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
//    SCOPE_ALLOW_CONSTRUCT_DBG      allows debug inside the constructor
//    SCOPE_ERROR_SIGNAL_HANDLER     allows to register SIGSEGV&SIGBUS&SIGABRT handler
//    SCOPE_QUEUE_LENGTH             override default circular buffer sizes
//    SCOPE_ALLOW_BINARY_CONSOLE     "true" outputs all console data, always

#define SCOPE_PID_ENV "SCOPE_PID"
#define PRESERVE_PERF_REPORTING "SCOPE_PERF_PRESERVE"

// TLS protocol refs that have been useful:
//   https://tools.ietf.org/html/rfc5246
//   http://blog.fourthbit.com/2014/12/23/traffic-analysis-of-an-ssl-slash-tls-session/
//   https://tls13.ulfheim.net/
//
// The definitions below were the original config we were using but during work
// on #543, we found some cases where the first 6 bytes of the stream were
// split across two different payloads and doProtocol() is not doing any
// buffering. TLS detection wasn't working.
//
// A little background... The TLS record header is 5 bytes. The header for
// handshake records starts with 0x16 and, after the remaining 4 bytes of the
// record header, is followed by a handshake header. The "client hello" and
// "server hello" handshake headers start with 0x01 and 0x02 respectively. The
// original regex below is looking for the TLS record header followed by the
// first byte of either the "client hello" or "server hello" handshake header.
//
//#define PAYLOAD_BYTESRC 6
//#define PAYLOAD_REGEX "^16030[0-3].{4}0[12]"
//
// We found that we're only getting the first 5 bytes in one payload to
// doProtocol() in some cases. It's because the read operation was only for 5
// bytes or some network fragmentation happened or wonky stream buffering or
// whatever. In any case, we're changing this to only look for the 5-byte
// record header and not the handshake header that follows.
//
// If you end up here because you're tracking down a partial payload issue in
// the future, consider adding some buffering of initial payload data on a
// channel until you have enough to satisfy TLS and other protocol detection.
//
//#define PAYLOAD_BYTESRC 5
//#define PAYLOAD_REGEX "^16030[0-3].{4}"
//
// Another iteration after finding the Java SSLSocketClient program in our
// oracle-java integration tests was connecting to the server using SSL2
// instead of SSL3. SSL3 is what we know as TLS. SSL2 appears to use the same
// records except they can use a different header. The updated regex below
// looks for either form though the "magic" for SSL2 is pretty weak.
//
// The updated regex is also now using non-capturing groups to improve
// performance of TLS detection.
//
#define PAYLOAD_BYTESRC 5
#define PAYLOAD_REGEX "^(?:(?:16030[0-3].{4})|(?:8[0-9a-fA-F]{3}01))"

// libmusl requires LD_LIBRARY_PATH
#define LD_LIB_ENV "LD_LIBRARY_PATH"
#define LD_LIB_DIR "libscope-v"

// We've seen TLS connections hang when the remote side drops during the
// establishTlsSession() process...  this helps ensure we won't hang
// processes forever while waiting for a single connection to complete.
#define MAX_TLS_CONNECT_SECONDS 5

#endif // __SCOPETYPES_H__

