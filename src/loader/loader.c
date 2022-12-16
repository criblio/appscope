#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <elf.h>
#include <libgen.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/utsname.h>

#include "scopestdlib.h"
#include "scopetypes.h"
#include "libdir.h"
#include "loaderop.h"
#include "nsinfo.h"
#include "nsfile.h"
#include "ns.h"
#include "setup.h"
#include "loaderutils.h"
#include "inject.h"
#include "loader.h"
 
// maybe set this from a cmd line switch?
int g_log_level = CFG_LOG_WARN;

/* 
 * This avoids a segfault when code using shm_open() is compiled statically.
 * For some reason, compiling the code statically causes the __shm_directory()
 * function calls in librt.a to not reach the implementation in libpthread.a.
 * Implementing the function ourselves fixes this issue.
 *
 * See https://stackoverflow.com/a/47914897
 */
#ifndef  SHM_MOUNT
#define  SHM_MOUNT "/dev/shm/"
#endif
static const char  shm_mount[] = SHM_MOUNT;
const char *__shm_directory(size_t *len)
{
    if (len)
        *len = strlen(shm_mount);
    return shm_mount;
}

static const char help_overview[] =
"  OVERVIEW:\n"
"    The Scope library supports extraction of data from within applications.\n"
"    As a general rule, applications consist of one or more processes.\n"
"    The Scope library can be loaded into any process as the\n"
"    process starts.\n"
"    The primary way to define which processes include the Scope library\n"
"    is by exporting the environment variable LD_PRELOAD, which is set to point\n"
"    to the path name of the Scope library. E.g.: \n"
"    export LD_PRELOAD=./libscope.so\n"
"\n"
"    Scope emits data as metrics and/or events.\n"
"    Scope is fully configurable by means of a configuration file (scope.yml)\n"
"    and/or environment variables.\n"
"\n"
"    Metrics are emitted in StatsD format, over a configurable link. By default,\n"
"    metrics are sent over a UDP socket using localhost and port 8125.\n"
"\n"
"    Events are emitted in JSON format over a configurable link. By default,\n"
"    events are sent over a TCP socket using localhost and port 9109.\n"
"\n"
"    Scope logs to a configurable destination, at a configurable\n"
"    verbosity level. The default verbosity setting is level 4, and the\n"
"    default destination is the file `/tmp/scope.log`.\n";

static const char help_configuration[] =
"  CONFIGURATION:\n"
"    Configuration File:\n"
"        A YAML config file (named scope.yml) enables control of all available\n"
"        settings. The config file is optional. Environment variables take\n"
"        precedence over settings in a config file.\n"
"\n"
"    Config File Resolution\n"
"        If the SCOPE_CONF_PATH env variable is defined, and points to a\n"
"        file that can be opened, it will use this as the config file.\n"
"        Otherwise, AppScope searches for the config file in this priority\n"
"        order, using the first one it finds:\n"
"\n"
"            $SCOPE_HOME/conf/scope.yml\n"
"            $SCOPE_HOME/scope.yml\n"
"            /etc/scope/scope.yml\n"
"            ~/conf/scope.yml\n"
"            ~/scope.yml\n"
"            ./conf/scope.yml\n"
"            ./scope.yml\n"
"\n"
"    Environment Variables:\n"
"        SCOPE_CONF_PATH\n"
"            Directly specify config file's location and name.\n"
"            Used only at start time.\n"
"        SCOPE_HOME\n"
"            Specify a directory from which conf/scope.yml or ./scope.yml can\n"
"            be found. Used only at start time, and only if SCOPE_CONF_PATH\n"
"            does not exist. For more info, see Config File Resolution below.\n"
"        SCOPE_METRIC_ENABLE\n"
"            Single flag to make it possible to disable all metric output.\n"
"            true,false  Default is true.\n"
"        SCOPE_METRIC_VERBOSITY\n"
"            0-9 are valid values. Default is 4.\n"
"            For more info, see Metric Verbosity below.\n"
"        SCOPE_METRIC_FS\n"
"            Create metrics describing file connectivity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_NET\n"
"            Create metrics describing network connectivity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_HTTP\n"
"            Create metrics describing HTTP communication.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_DNS\n"
"            Create metrics describing DNS activity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_PROC\n"
"            Create metrics describing process.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_STATSD\n"
"            When true, statsd metrics sent or received by this application\n"
"            will be handled as appscope-native metrics.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_DEST\n"
"            Default is udp://localhost:8125\n"
"            Format is one of:\n"
"                file:///tmp/output.log\n"
"                    Output to a file. Use file://stdout, file://stderr for\n"
"                    STDOUT or STDERR\n"
"                udp://host:port\n"
"                tcp://host:port\n"
"                    Send to a TCP or UDP server. \"host\" is the hostname or\n"
"                    IP address and \"port\" is the port number of service name.\n"
"                unix://socketpath\n"
"                    Output to a unix domain server using TCP.\n"
"                    Use unix://@abstractname, unix:///var/run/mysock for\n"
"                    abstract address or filesystem address.\n"
"        SCOPE_METRIC_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_METRIC_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_METRIC_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_METRIC_FORMAT\n"
"            statsd, ndjson\n"
"            Default is statsd.\n"
"        SCOPE_STATSD_PREFIX\n"
"            Specify a string to be prepended to every scope metric.\n"
"        SCOPE_STATSD_MAXLEN\n"
"            Default is 512.\n"
"        SCOPE_SUMMARY_PERIOD\n"
"            Number of seconds between output summarizations. Default is 10.\n"
"        SCOPE_EVENT_ENABLE\n"
"            Single flag to make it possible to disable all event output.\n"
"            true,false  Default is true.\n"
"        SCOPE_EVENT_DEST\n"
"            Same format as SCOPE_METRIC_DEST above.\n"
"            Default is tcp://localhost:9109\n"
"        SCOPE_EVENT_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_EVENT_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_EVENT_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_EVENT_FORMAT\n"
"            ndjson\n"
"            Default is ndjson.\n"
"        SCOPE_EVENT_LOGFILE\n"
"            Create events from writes to log files.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_LOGFILE_NAME\n"
"            An extended regex to filter log file events by file name.\n"
"            Used only if SCOPE_EVENT_LOGFILE is true. Default is .*log.*\n"
"        SCOPE_EVENT_LOGFILE_VALUE\n"
"            An extended regex to filter log file events by field value.\n"
"            Used only if SCOPE_EVENT_LOGFILE is true. Default is .*\n"
"        SCOPE_EVENT_CONSOLE\n"
"            Create events from writes to stdout, stderr.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_CONSOLE_NAME\n"
"            An extended regex to filter console events by event name.\n"
"            Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"        SCOPE_EVENT_CONSOLE_VALUE\n"
"            An extended regex to filter console events by field value.\n"
"            Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"        SCOPE_EVENT_METRIC\n"
"            Create events from metrics.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_METRIC_NAME\n"
"            An extended regex to filter metric events by event name.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_METRIC_FIELD\n"
"            An extended regex to filter metric events by field name.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_METRIC_VALUE\n"
"            An extended regex to filter metric events by field value.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_HTTP\n"
"            Create events from HTTP headers.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_HTTP_NAME\n"
"            An extended regex to filter http events by event name.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_FIELD\n"
"            An extended regex to filter http events by field name.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_VALUE\n"
"            An extended regex to filter http events by field value.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_HEADER\n"
"            An extended regex that defines user defined headers\n"
"            that will be extracted. Default is NULL\n"
"        SCOPE_EVENT_NET\n"
"            Create events describing network connectivity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_NET_NAME\n"
"            An extended regex to filter network events by event name.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_NET_FIELD\n"
"            An extended regex to filter network events by field name.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_NET_VALUE\n"
"            An extended regex to filter network events by field value.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_FS\n"
"            Create events describing file connectivity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_FS_NAME\n"
"            An extended regex to filter file events by event name.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_FS_FIELD\n"
"            An extended regex to filter file events by field name.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_FS_VALUE\n"
"            An extended regex to filter file events by field value.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_DNS\n"
"            Create events describing DNS activity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_DNS_NAME\n"
"            An extended regex to filter dns events by event name.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_DNS_FIELD\n"
"            An extended regex to filter DNS events by field name.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_DNS_VALUE\n"
"            An extended regex to filter dns events by field value.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_MAXEPS\n"
"            Limits number of events that can be sent in a single second.\n"
"            0 is 'no limit'; 10000 is the default.\n"
"        SCOPE_ENHANCE_FS\n"
"            Controls whether uid, gid, and mode are captured for each open.\n"
"            Used only if SCOPE_EVENT_FS is true. true,false Default is true.\n"
"        SCOPE_LOG_LEVEL\n"
"            debug, info, warning, error, none. Default is error.\n"
"        SCOPE_LOG_DEST\n"
"            same format as SCOPE_METRIC_DEST above.\n"
"            Default is file:///tmp/scope.log\n"
"        SCOPE_LOG_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_LOG_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_LOG_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_TAG_\n"
"            Specify a tag to be applied to every metric and event.\n"
"            Environment variable expansion is available, \n"
"            e.g.: SCOPE_TAG_user=$USER\n"
"        SCOPE_CMD_DIR\n"
"            Specifies a directory to look for dynamic configuration files.\n"
"            See Dynamic Configuration below.\n"
"            Default is /tmp\n"
"        SCOPE_PAYLOAD_ENABLE\n"
"            Flag that enables payload capture.  true,false  Default is false.\n"
"        SCOPE_PAYLOAD_DIR\n"
"            Specifies a directory where payload capture files can be written.\n"
"            Default is /tmp\n"
"        SCOPE_CRIBL_ENABLE\n"
"            Single flag to make it possible to disable cribl backend.\n"
"            true,false  Default is true.\n"
"        SCOPE_CRIBL_CLOUD\n"
"            Intended as an alternative to SCOPE_CRIBL below. Identical\n"
"            behavior, except that where SCOPE_CRIBL can have TLS settings\n"
"            modified via related SCOPE_CRIBL_TLS_* environment variables,\n"
"            SCOPE_CRIBL_CLOUD hardcodes TLS settings as though these were\n"
"            individually specified:\n"
"                SCOPE_CRIBL_TLS_ENABLE=true\n"
"                SCOPE_CRIBL_TLS_VALIDATE_SERVER=true\n"
"                SCOPE_CRIBL_TLS_CA_CERT_PATH=\"\"\n"
"            As a note, library behavior will be undefined if this variable is\n"
"            set simultaneously with SCOPE_CRIBL, or any of SCOPE_CRIBL_TLS_*.\n"
"        SCOPE_CRIBL\n"
"            Defines a connection with Cribl LogStream\n"
"            Default is NULL\n"
"            Format is one of:\n"
"                tcp://host:port\n"
"                    If no port is provided, defaults to 10090\n"
"                unix://socketpath\n"
"                    Output to a unix domain server using TCP.\n"
"                    Use unix://@abstractname, unix:///var/run/mysock for\n"
"                    abstract address or filesystem address.\n"
"        SCOPE_CRIBL_AUTHTOKEN\n"
"            Authentication token provided by Cribl.\n"
"            Default is an empty string.\n"
"        SCOPE_CRIBL_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_CRIBL_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_CRIBL_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_CONFIG_EVENT\n"
"            Sends a single process-identifying event, when a transport\n"
"            connection is established.  true,false  Default is true.\n"
"\n"
"    Dynamic Configuration:\n"
"        Dynamic Configuration allows configuration settings to be\n"
"        changed on the fly after process start time. At every\n"
"        SCOPE_SUMMARY_PERIOD, the library looks in SCOPE_CMD_DIR to\n"
"        see if a file scope.<pid> exists. If it exists, the library processes\n"
"        every line, looking for environment variable–style commands\n"
"        (e.g., SCOPE_CMD_DBG_PATH=/tmp/outfile.txt). The library changes the\n"
"        configuration to match the new settings, and deletes the\n"
"        scope.<pid> file when it's complete.\n"
"\n"
"        While every environment variable defined here can be changed via\n"
"        Dynamic Configuration, there are a few environment variable-style\n"
"        commands that are only accepted during Dynamic Configuration.\n"
"        These will be ignored if specified as actual environment variables.\n"
"        They are listed here:\n"
"            SCOPE_CMD_ATTACH\n"
"                Flag that controls whether interposed functions are\n"
"                processed by AppScope (true), or not (false).\n"
"            SCOPE_CMD_DBG_PATH\n"
"                Causes AppScope to write debug information to the\n"
"                specified file path. Absolute paths are recommended.\n"
"            SCOPE_CONF_RELOAD\n"
"                Causes AppScope to reload its configuration file. If\n"
"                the value is \"true\", it is reloaded according to the\n"
"                Config File Resolution above. If any other value is\n"
"                specified, it is handled like SCOPE_CONF_PATH, but without\n"
"                the \"Used only at start time\" limitation.\n"
"\n"
"    Certificate Authority Resolution\n"
"        If SCOPE_*_TLS_ENABLE and SCOPE_*_TLS_VALIDATE_SERVER are true then\n"
"        AppScope performs TLS server validation. For this to be successful\n"
"        a CA certificate must be found that can authenticate the certificate\n"
"        the server provides during the TLS handshake process.\n"
"        If SCOPE_*_TLS_CA_CERT_PATH is set, AppScope will use this file path\n"
"        which is expected to contain CA certificates in PEM format. If this\n"
"        env variable is an empty string or not set, AppScope searches for a\n"
"        usable root CA file on the local filesystem, using the first one\n"
"        found from this list:\n"
"\n"
"            /etc/ssl/certs/ca-certificates.crt\n"
"            /etc/pki/tls/certs/ca-bundle.crt\n"
"            /etc/ssl/ca-bundle.pem\n"
"            /etc/pki/tls/cacert.pem\n"
"            /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem\n"
"            /etc/ssl/cert.pem\n";

static const char help_metrics[] =
"  METRICS:\n"
"    Metrics can be enabled or disabled with a single config element\n"
"    (metric: enable: true|false). Specific types of metrics, and specific \n"
"    field content, are managed with a Metric Verbosity setting.\n"
"\n"
"    Metric Verbosity\n"
"        Controls two different aspects of metric output – \n"
"        Tag Cardinality and Summarization.\n"
"\n"
"        Tag Cardinality\n"
"            0   No expanded StatsD tags\n"
"            1   adds 'data', 'unit'\n"
"            2   adds 'class', 'proto'\n"
"            3   adds 'op'\n"
"            4   adds 'pid', 'host', 'proc', 'http_status'\n"
"            5   adds 'domain', 'file'\n"
"            6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'\n"
"            7   adds 'fd', 'args'\n"
"            8   adds 'duration','numops','req_per_sec','req','resp','protocol'\n"
"\n"
"        Summarization\n"
"            0-4 has full event summarization\n"
"            5   turns off 'error'\n"
"            6   turns off 'filesystem open/close' and 'dns'\n"
"            7   turns off 'filesystem stat' and 'network connect'\n"
"            8   turns off 'filesystem seek'\n"
"            9   turns off 'filesystem read/write' and 'network send/receive'\n"
"\n"
"    The http.status metric is emitted when the http watch type has been\n"
"    enabled as an event. The http.status metric is not controlled with\n"
"    summarization settings.\n";

static const char help_events[] =
"  EVENTS:\n"
"    All events can be enabled or disabled with a single config element\n"
"    (event: enable: true|false). Unlike metrics, event content is not \n"
"    managed with verbosity settings. Instead, you use regex filters that \n"
"    manage which field types and field values to include.\n"
"\n"
"     Events are organized as 7 watch types: \n"
"     1) File Content. Provide a pathname, and all data written to the file\n"
"        will be organized in JSON format and emitted over the event channel.\n"
"     2) Console Output. Select stdin and/or stdout, and all data written to\n"
"        these endpoints will be formatted in JSON and emitted over the event\n"
"        channel.\n"
"     3) Metrics. Event metrics provide the greatest level of detail from\n"
"        libscope. Events are created for every read, write, send, receive,\n"
"        open, close, and connect. These raw events are configured with regex\n"
"        filters to manage which event, which specific fields within an event,\n"
"        and which value patterns within a field to include.\n"
"     4) HTTP Headers. HTTP headers are extracted, formatted in JSON, and\n"
"        emitted over the event channel. Three types of events are created\n"
"        for HTTP headers: 1) HTTP request events, 2) HTTP response events,\n"
"        and 3) a metric event corresponding to the request and response\n"
"        sequence. A response event includes the corresponding request,\n"
"        status and duration fields. An HTTP metric event provides fields\n"
"        describing bytes received, requests per second, duration, and status.\n"
"        Any header defined as X-appscope (case insensitive) will be emitted.\n"
"        User defined headers are extracted by using the headers field.\n"
"        The headers field is a regular expression.\n"
"     5) File System. Events are formatted in JSON for each file system open,\n"
"        including file name, permissions, and cgroup. Events for file system\n"
"        close add a summary of the number of bytes read and written, the\n"
"        total number of read and write operations, and the total duration\n"
"        of read and write operations. The specific function performing open\n"
"        and close is reported as well.\n"
"     6) Network. Events are formatted in JSON for network connections and \n"
"        corresponding disconnects, including type of protocol used, and \n"
"        local and peer IP:port. Events for network disconnect add a summary\n"
"        of the number of bytes sent and received, and the duration of the\n"
"        sends and receives while the connection was active. The reason\n"
"        (source) for disconnect is provided as local or remote. \n"
"     7) DNS. Events are formatted in JSON for DNS requests and responses.\n"
"        The event provides the domain name being resolved. On DNS response,\n"
"        the event provides the duration of the DNS operation.\n";

static const char help_protocol[] =
"  PROTOCOL DETECTION:\n"
"    Scope can detect any defined network protocol. You provide protocol\n"
"    definitions in the \"protocol\" section of the config file. You describe \n"
"    protocol specifics in one or more regex definitions. PCRE2 regular \n"
"    expressions are supported. The stock scope.yml file for examples.\n"
"\n"
"    Scope detects binary and string protocols. Detection events, \n"
"    formatted in JSON, are emitted over the event channel unless the \"detect\"\n"
"    property is set to \"false\".\n"
"\n"
"  PAYLOAD EXTRACTION:\n"
"    When enabled, libscope extracts payload data from network operations.\n"
"    Payloads are emitted in binary. No formatting is applied to the data.\n"
"    Payloads are emitted to either a local file or the LogStream channel.\n"
"    Configuration elements for libscope support defining a path for payload\n"
"    data.\n";

static int
showHelp(const char *section)
{
    printf(
      "Cribl AppScope Static Loader %s\n"
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n",
      SCOPE_VER
    );

    if (!section || !strcasecmp(section, "all")) {
        puts(help_overview);
        puts(help_configuration);
        puts(help_metrics);
        puts(help_events);
        puts(help_protocol);
    } else if (!strcasecmp(section, "overview")) {
        puts(help_overview);
    } else if (!strcasecmp(section, "configuration") || !strcasecmp(section, "config")) {
        puts(help_configuration);
    } else if (!strcasecmp(section, "metrics")) {
        puts(help_metrics);
    } else if (!strcasecmp(section, "events")) {
        puts(help_events);
    } else if (!strcasecmp(section, "protocols")) {
        puts(help_protocol);
    } else {
        fprintf(stderr, "error: invalid help section\n\n");
        return -1;
    }
    fflush(stdout);
    return 0;
}

static void
showUsage(char *prog)
{
    printf(
      "\n"
      "Cribl AppScope Static Loader %s\n" 
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n"
      "usage: %s [OPTIONS] [--] EXECUTABLE [ARGS...]\n"
      "       %s [OPTIONS] --attach PID\n"
      "       %s [OPTIONS] --detach PID\n"
      "       %s [OPTIONS] --configure FILTER_PATH --namespace PID\n"
      "       %s [OPTIONS] --service SERVICE --namespace PID\n"
      "\n"
      "options:\n"
      "  -u, --usage                  display this info\n"
      "  -h, --help [SECTION]         display all or the specified help section\n"
      "  -l, --libbasedir DIR         specify parent for the library directory (default: /tmp)\n"
      "  -f DIR                       alias for \"-l DIR\" for backward compatibility\n"
      "  -a, --attach PID             attach to the specified process ID\n"
      "  -d, --detach PID             detach from the specified process ID\n"
      "  -c, --configure FILTER_PATH  configure scope environment with FILTER_PATH\n"
      "  -w, --unconfigure            unconfigure scope environment\n"
      "  -s, --service SERVICE        setup specified service NAME\n"
      "  -v, --unservice              remove scope from all service configurations\n"
      "  -n  --namespace PID          perform service/configure operation on specified container PID\n"
      "  -p, --patch SO_FILE          patch specified libscope.so\n"
      "  -r, --starthost              execute the scope start command in a host context (must be run in the container)\n"
      "  -x, --stophost               execute the scope stop command in a host context (must be run in the container)\n"
      "\n"
      "Help sections are OVERVIEW, CONFIGURATION, METRICS, EVENTS, and PROTOCOLS.\n"
      "\n"
      "See `scope` if you are new to AppScope as it provides a simpler and more\n"
      "user-friendly experience as you come up to speed.\n"
      "\n"
      "User docs are at https://appscope.dev/docs/. The project is hosted at\n"
      "https://github.com/criblio/appscope. Please direct feature requests and\n"
      "defect reports there.\n"
      "\n",
      SCOPE_VER, prog, prog, prog, prog, prog
    );
    fflush(stdout);
}

static int
attach(pid_t pid, char *scopeLibPath)
{
    char *exe_path = NULL;
    elf_buf_t *ebuf;

    if (geteuid()) {
        printf("error: --attach requires root\n");
        return EXIT_FAILURE;
    }

    if (getExePath(pid, &exe_path) == -1) {
        fprintf(stderr, "error: can't get path to executable for pid %d\n", pid);
        return EXIT_FAILURE;
    }

    if ((ebuf = getElf(exe_path)) == NULL) {
        free(exe_path);
        fprintf(stderr, "error: can't read the executable %s\n", exe_path);
        return EXIT_FAILURE;
    }

    if (is_static(ebuf->buf) == TRUE) {
        fprintf(stderr, "error: can't attach to the static executable: %s\nNote that the executable can be 'scoped' using the command 'scope run -- %s'\n", exe_path, exe_path);
        free(exe_path);
        freeElf(ebuf->buf, ebuf->len);
        return EXIT_FAILURE;
    }

    free(exe_path);
    freeElf(ebuf->buf, ebuf->len);

    printf("Attaching to process %d\n", pid);
    int ret = injectScope(pid, scopeLibPath);

    // done
    return ret;
}

/*
 * Given the ability to separate lib load and interposition
 * we enable a detach capability as well as 3 types
 * of attach cases. 4 commands in all.
 *
 * Load and attach:
 * libscope is not loaded.  This case is
 * handled in function attach().
 *
 * First attach:
 * libscope is loaded and we are not interposing
 * functions, not scoped. This is the first time
 * libscope will have been attached.
 *
 * Reattach:
 * libscope is loaded, the process has been attached
 * in one form at least once previously.
 *
 * Detach:
 * libscope is loaded and we are interposing functions.
 * Remove all interpositions and stop scoping.
 */
static int
attachCmd(pid_t pid, bool attach)
{
    int fd, sfd, mfd;
    bool first_attach = FALSE;
    export_sm_t *exaddr;
    char buf[PATH_MAX] = {0};

    /*
     * The SM segment is used in the first attach case where
     * we've never been attached before. The segment is
     * populated with state including the address of the
     * reattach command in the lib. The segment is read
     * only and the size of the segment can't be modified.
     *
     * We use the presence of the segment to identify the
     * state of the lib. The segment is deleted when a
     * first attach is performed.
     */
    sfd = findFd(pid, SM_NAME);   // e.g. memfd:anon
    if (sfd > 0) {
        first_attach = TRUE;
    }

    /*
     * On command detach if the SM segment exists, we have never been
     * attached. Return and do not create the command file as it
     * will never be deleted until attached and then causes an
     * unintended detach.
     */
    if ((attach == FALSE) && (first_attach == TRUE)) {
        printf("Already detached from pid %d\n", pid);
        return EXIT_SUCCESS;
    }

    /*
     * Before executing any command, create and populate the dyn config file.
     * It is used for all cases:
     *   First attach: no attach command, include env vars, reload command
     *   Reattach: attach command = true, include env vars, reload command
     *   Detach: attach command = false, no env vars, no reload command
     */
    snprintf(buf, sizeof(buf), "%s/%s.%d",
                   DYN_CONFIG_CLI_DIR, DYN_CONFIG_CLI_PREFIX, pid);

    /*
     * Unlink a possible existing file before creating a new one
     * due to a fact that open will fail if the file is
     * sealed (still processed on library side).
     * File sealing is supported on tmpfs - /dev/shm (DYN_CONFIG_CLI_DIR).
     */
    unlink(buf);


    uid_t nsUid = nsInfoTranslateUid(pid);
    gid_t nsGid = nsInfoTranslateGid(pid);

    fd = nsFileOpen(buf, O_WRONLY|O_CREAT, nsUid, nsGid, geteuid(), getegid());
    if (fd == -1) {
        return EXIT_FAILURE;
    }

    if (fchmod(fd, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH) == -1) {
        perror("fchmod() failed");
        return EXIT_FAILURE;
    }

    /*
     * Ensuring that the process being operated on can remove
     * the dyn config file being created here.
     * In the case where a root user executes the command,
     * we need to change ownership of dyn config file.
     */

    if (geteuid() == 0) {
        uid_t euid = -1;
        gid_t egid = -1;

        if (getProcUidGid(pid, &euid, &egid) == -1) {
            fprintf(stderr, "error: osGetProcUidGid() failed\n");
            close(fd);
            return EXIT_FAILURE;
        }

        if (fchown(fd, euid, egid) == -1) {
            perror("fchown() failed");
            close(fd);
            return EXIT_FAILURE;
        }
    }

    if (first_attach == FALSE) {
        const char *cmd = (attach == TRUE) ? "SCOPE_CMD_ATTACH=true\n" : "SCOPE_CMD_ATTACH=false\n";
        if (write(fd, cmd, strlen(cmd)) <= 0) {
            perror("write() failed");
            close(fd);
            return EXIT_FAILURE;
        }
    }

    if (attach == TRUE) {
        int i;

        if (first_attach == TRUE) {
            printf("First attach to pid %d\n", pid);
        } else {
            printf("Reattaching to pid %d\n", pid);
        }

        /*
        * Reload the configuration during first attach & reattach if we want to apply
        * config from a environment variable
        * Handle SCOPE_CONF_RELOAD in first order because of "processReloadConfig" logic
        * is done in two steps:
        * - first - create a configuration based on path (default one or custom one)
        * - second - process env variables existing in the process (cfgProcessEnvironment)
        * We append rest of the SCOPE_ variables after since in this way we ovewrite the ones
        * which was set by cfgProcessEnvironment in second step.
        * TODO: Handle the file and env variables
        */
        char *scopeConfReload = getenv("SCOPE_CONF_RELOAD");
        if (!scopeConfReload) {
            dprintf(fd, "SCOPE_CONF_RELOAD=TRUE\n");
        } else {
            dprintf(fd, "SCOPE_CONF_RELOAD=%s\n", scopeConfReload);
        }

        for (i = 0; environ[i]; ++i) {
            size_t envLen = strlen(environ[i]);
            if ((envLen > 6) &&
                (strncmp(environ[i], "SCOPE_", 6) == 0) &&
                (!strstr(environ[i], "SCOPE_CONF_RELOAD"))) {
                    dprintf(fd, "%s\n", environ[i]);
            }
        }
    } else {
        printf("Detaching from pid %d\n", pid);
    }

    close(fd);

    int rc = EXIT_SUCCESS;

    if (first_attach == TRUE) {
        memset(buf, 0, PATH_MAX);
        // the only command we do in this case is first attach
        snprintf(buf, sizeof(buf), "/proc/%d/fd/%d", pid, sfd);
        if ((mfd = open(buf, O_RDONLY)) == -1) {
            close(sfd);
            perror("open");
            return EXIT_FAILURE;
        }

        if ((exaddr = mmap(NULL, sizeof(export_sm_t), PROT_READ,
                                 MAP_SHARED, mfd, 0)) == MAP_FAILED) {
            close(sfd);
            close(mfd);
            perror("mmap");
            return EXIT_FAILURE;
        }

        if (injectFirstAttach(pid, exaddr->cmdAttachAddr) == EXIT_FAILURE) {
            fprintf(stderr, "error: %s: you must have administrator privileges to run this command\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }

        //fprintf(stderr, "%s: %s 0x%lx\n", __FUNCTION__, buf, exaddr->cmdAttachAddr);

        close(sfd);
        close(mfd);

        if (munmap(exaddr, sizeof(export_sm_t))) {
            fprintf(stderr, "error: %s: unmapping in the the reattach command failed\n", __FUNCTION__);
            rc = EXIT_FAILURE;
        }
    }
        return rc;
}

// long aliases for short options
static struct option opts[] = {
    { "usage",       no_argument,       0, 'u'},
    { "help",        optional_argument, 0, 'h' },
    { "attach",      required_argument, 0, 'a' },
    { "detach",      required_argument, 0, 'd' },
    { "namespace",   required_argument, 0, 'n' },
    { "configure",   required_argument, 0, 'c' },
    { "unconfigure", no_argument,       0, 'w' },
    { "service",     required_argument, 0, 's' },
    { "unservice",   no_argument,       0, 'v' },
    { "libbasedir",  required_argument, 0, 'l' },
    { "patch",       required_argument, 0, 'p' },
    { "starthost",   no_argument,       0, 'r' },
    { "stophost",    no_argument,       0, 'x' },
    { "loader",      no_argument,       0, 'z' },
    { 0, 0, 0, 0 }
};

int
loader(int argc, char **argv, char **env)
{
    //printf("loader called\n");

    char *attachArg = NULL;
    char *configFilterPath = NULL;
    char *serviceName = NULL;
    char *nsPidArg = NULL;
    char *scopeLibPath = NULL;
    char path[PATH_MAX] = {0};
    pid_t pid = -1;
    char attachType = 'u';
    uid_t eUid = geteuid();
    gid_t eGid = getegid();
    uid_t nsUid = eUid;
    uid_t nsGid = eGid;
    bool unconfigure = FALSE;
    bool unservice = FALSE;

    // process command line
    for (;;) {
        int index = 0;
        //
        // The `+` here enables POSIX mode where the first non-option found
        // stops option processing so `ldscope foo -a 123` will not process the
        // `-a 123` here and instead pass it through.
        //
        // The initial `:` lets us handle options with optional values like
        // `-h` and `-h SECTION`.
        //
        int opt = getopt_long(argc, argv, "+:uh:a:d:n:l:f:p:c:s:rz", opts, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'u':
                showUsage(basename(argv[0]));
                return EXIT_SUCCESS;
            case 'h':
                // handle `-h SECTION`
                if (showHelp(optarg)) {
                    showUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                return EXIT_SUCCESS;
            case 'a':
                attachArg = optarg;
                attachType = 'a';
                break;
            case 'd':
                attachArg = optarg;
                attachType = 'd';
                break;
            case 'n':
                nsPidArg = optarg;
                break;
            case 'c':
                configFilterPath = optarg;
                break;
            case 'w':
                unconfigure = TRUE;
                break;
            case 's':
                serviceName = optarg;
                break;
            case 'v':
                unservice = TRUE;
                break;
            case 'f':
                // accept -f as alias for -l for BC
            case 'l':
                if (libdirSetLibraryBase(optarg))
                {
                    return EXIT_FAILURE;
                }
                break;
            case 'p':
                return (loaderOpPatchLibrary(optarg) == PATCH_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE;
                break;
            case 'r':
                return nsHostStart();
                break;
            case 'x':
                return nsHostStop();
                break;
            case 'z':
                // Ignore
                break;
            case ':':
                // options missing their value end up here
                switch (optopt) {
                    case 'h':
                        // handle `-h` without the section value
                        showHelp(0);
                        return EXIT_SUCCESS;
                    default: 
                        fprintf(stderr, "error: missing required value for -%c option\n", optopt);
                        showUsage(basename(argv[0]));
                        return EXIT_FAILURE;
                }
                break;
            case '?':
            default:
                fprintf(stderr, "error: invalid option: -%c\n", optopt);
                showUsage(basename(argv[0]));
                return EXIT_FAILURE;
        }
    }

    // either --attach, --detach, --configure, --unconfigure, --service, --unservice or a command are required
    if (!attachArg && !configFilterPath && !unconfigure && !serviceName && !unservice && optind >= argc) {
        fprintf(stderr, "error: missing --attach, --detach, --configure, --unconfigure, --service, --unservice option or EXECUTABLE argument\n");
        showUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }

    if (attachArg && (serviceName || unservice)) {
        fprintf(stderr, "error: --attach/--detach and --service/--unservice cannot be used together\n");
        showUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }

    if (attachArg && (configFilterPath || unconfigure)) {
        fprintf(stderr, "error: --attach/--detach and --configure/--unconfigure cannot be used together\n");
        showUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }

    if ((configFilterPath || unconfigure) && (serviceName || unservice)) {
        fprintf(stderr, "error: --configure/--unconfigure and --service/--unservice cannot be used together\n");
        showUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }

    if (nsPidArg && ((configFilterPath == NULL && !unconfigure) && (serviceName == NULL && !unservice))) {
        fprintf(stderr, "error: --namespace option required --configure/--unconfigure or --service/--unservice option\n");
        showUsage(basename(argv[0]));
        return EXIT_FAILURE;
    }

    // use --attach, --detach, --configure, --unconfigure, --service, --unservice ignore executable and args
    if (optind < argc) {
        if (attachArg) {
            fprintf(stderr, "warning: ignoring EXECUTABLE argument with --attach, --detach option\n");
        } else if (configFilterPath || unconfigure) {
            fprintf(stderr, "warning: ignoring EXECUTABLE argument with --configure/--unconfigure option\n");
        } else if (serviceName) {
            fprintf(stderr, "warning: ignoring EXECUTABLE argument with --service/--unservice option\n");
        }
    }

    if (serviceName) {
        // must be root
        if (eUid) {
            printf("error: --service requires root\n");
            return EXIT_FAILURE;
        }

        if (nsPidArg) {
            pid = atoi(nsPidArg);
            if (pid < 1) {
                fprintf(stderr, "error: invalid --namespace PID: %s\n", nsPidArg);
                return EXIT_FAILURE;
            }
        }
        if (pid == -1) {
            // Service on Host
            return setupService(serviceName, eUid, eGid);
        } else {
            // Service on Container
            pid_t nsContainerPid = 0;
            if ((nsInfoGetPidNs(pid, &nsContainerPid) == TRUE) ||
                (nsInfoIsPidInSameMntNs(pid) == FALSE)) {
                return nsService(pid, serviceName);
            }
        }
        return EXIT_FAILURE;
    }

    if (unservice) {
        // must be root
        if (eUid) {
            printf("error: --unservice requires root\n");
            return EXIT_FAILURE;
        }

        if (nsPidArg) {
            pid = atoi(nsPidArg);
            if (pid < 1) {
                fprintf(stderr, "error: invalid --namespace PID: %s\n", nsPidArg);
                return EXIT_FAILURE;
            }
        }
        if (pid == -1) {
            // Service on Host
            return setupUnservice();
        } else {
            // Service on Container
            pid_t nsContainerPid = 0;
            if ((nsInfoGetPidNs(pid, &nsContainerPid) == TRUE) ||
                (nsInfoIsPidInSameMntNs(pid) == FALSE)) {
                return nsUnservice(pid);
            }
        }
        return EXIT_FAILURE;
    }

    if (configFilterPath) {
        int status = EXIT_FAILURE;
        // must be root
        if (eUid) {
            printf("error: --configure requires root\n");
            return EXIT_FAILURE;
        }

        if (nsPidArg) {
            pid = atoi(nsPidArg);
            if (pid < 1) {
                fprintf(stderr, "error: invalid --namespace PID: %s\n", nsPidArg);
                return EXIT_FAILURE;
            }
        }

        size_t configFilterSize = 0;
        void *confgFilterMem = setupLoadFileIntoMem(&configFilterSize, configFilterPath);
        if (confgFilterMem == NULL) {
            fprintf(stderr, "error: Load filter file into memory %s\n", configFilterPath);
            return EXIT_FAILURE;
        }

        if (pid == -1) {
            // Configure on Host
            status = setupConfigure(confgFilterMem, configFilterSize, eUid, eGid);
        } else {
            // Configure on Container
            pid_t nsContainerPid = 0;
            if ((nsInfoGetPidNs(pid, &nsContainerPid) == TRUE) ||
                (nsInfoIsPidInSameMntNs(pid) == FALSE)) {
                status = nsConfigure(pid, confgFilterMem, configFilterSize);
            }
        }
        if (confgFilterMem) {
            munmap(confgFilterMem, configFilterSize);
        }

        return status;
    }

    if (unconfigure) {
        int status = EXIT_FAILURE;
        // must be root
        if (eUid) {
            printf("error: --unconfigure requires root\n");
            return EXIT_FAILURE;
        }

        if (nsPidArg) {
            pid = atoi(nsPidArg);
            if (pid < 1) {
                fprintf(stderr, "error: invalid --namespace PID: %s\n", nsPidArg);
                return EXIT_FAILURE;
            }
        }
        if (pid == -1) {
            // Configure on Host
            status = setupUnconfigure();
        } else {
            // Configure on Container
            pid_t nsContainerPid = 0;
            if ((nsInfoGetPidNs(pid, &nsContainerPid) == TRUE) ||
                (nsInfoIsPidInSameMntNs(pid) == FALSE)) {
                status = nsUnconfigure(pid);
            }
        }
        return status;
    }

    // Extract the library
    if (pid != -1) {
        nsUid = nsInfoTranslateUid(pid);
        nsGid = nsInfoTranslateGid(pid);
    }

    if (libdirExtract(LIBRARY_FILE, nsUid, nsGid)) {
        fprintf(stderr, "error: failed to extract library\n");
        return EXIT_FAILURE;
    }

    // Set SCOPE_LIB_PATH
    scopeLibPath = (char *)libdirGetPath(LIBRARY_FILE);

    if (access(scopeLibPath, R_OK|X_OK)) {
        fprintf(stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        return EXIT_FAILURE;
    }

    if (setenv("SCOPE_LIB_PATH", scopeLibPath, 1)) {
        perror("setenv(SCOPE_LIB_PATH) failed");
        return EXIT_FAILURE;
    }

    // Set SCOPE_PID_ENV
    setPidEnv(getpid());

    /*
     * Attach & Detach
     *
     * Get the pid as int
     * Validate that the process exists
     * Perform namespace switch if required
     * Is the library currently loaded in the target process
     * Attach using ptrace or a dynamic command, depending on lib presence
     * Return at end of the operation
     */
    if (attachArg) {
        // target process must exist
        pid = atoi(attachArg);
        if (pid < 1) {
            printf("error: invalid --attach, --detach PID: %s\n", attachArg);
            return EXIT_FAILURE;
        }

        pid_t nsAttachPid = 0;

        snprintf(path, sizeof(path), "/proc/%d", pid);
        if (access(path, F_OK)) {
            printf("error: --attach, --detach PID not a current process: %d\n", pid);
            return EXIT_FAILURE;
        }

        uint64_t rc = findLibrary("libscope.so", pid, FALSE);

        /*
        * If the expected process exists in different PID namespace (container)
        * we do a following switch context sequence:
        * - load static loader file into memory
        * - [optionally] save the configuration file pointed by SCOPE_CONF_PATH into memory
        * - switch the namespace from parent
        * - save previously loaded static loader into new namespace
        * - [optionally] save previously loaded configuration file into new namespace
        * - fork & execute static loader attach one more time with updated PID
        */
        if (nsInfoGetPidNs(pid, &nsAttachPid) == TRUE) {
            // must be root to switch namespace
            if (eUid) {
                printf("error: --attach requires root\n");
                return EXIT_FAILURE;
            }
            return nsForkAndExec(pid, nsAttachPid, attachType);
        /*
        * Process can exists in same PID namespace but in different mnt namespace
        * we do a simillar sequecne like above but without switching PID namespace
        * and updating PID.
        */
        } else if (nsInfoIsPidInSameMntNs(pid) == FALSE) {
            // must be root to switch namespace
            if (eUid) {
                printf("error: --attach requires root\n");
                return EXIT_FAILURE;
            }
            return nsForkAndExec(pid, pid, attachType);
        }

        // This is an attach command
        if (attachType == 'a') {
            int ret;
            uint64_t rc;
            char path[PATH_MAX];

            // create /dev/shm/${PID}.env when attaching, for the library to load
            snprintf(path, sizeof(path), "/attach_%d.env", pid);
            int fd = nsFileShmOpen(path, O_RDWR|O_CREAT, S_IRUSR|S_IRGRP|S_IROTH, nsUid, nsGid, eUid, eGid);
            if (fd == -1) {
                fprintf(stderr, "nsFileShmOpen failed\n");
                return EXIT_FAILURE;
            }

            // add the env vars we want in the library
            dprintf(fd, "SCOPE_LIB_PATH=%s\n", libdirGetPath(LIBRARY_FILE));

            int i;
            for (i = 0; environ[i]; i++) {
                if (strlen(environ[i]) > 6 && strncmp(environ[i], "SCOPE_", 6) == 0) {
                    dprintf(fd, "%s\n", environ[i]);
                }
            }

            // done
            close(fd);

            pid = atoi(attachArg);
            if (pid < 1) {
                fprintf(stderr, "error: invalid PID for --attach, --detach\n");
                return EXIT_FAILURE;
            }

            rc = findLibrary("libscope.so", pid, FALSE);
            if (rc == -1) {
                fprintf(stderr, "error: can't get path to executable for pid %d\n", pid);
                ret = EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist, a load & attach
                ret = attach(pid, scopeLibPath);
            } else {
                // libscope exists, a first time attach or a reattach
                ret = attachCmd(pid, TRUE);
            }

            // remove the env var file
            snprintf(path, sizeof(path), "/attach_%d.env", pid);
            shm_unlink(path);
            return ret;
        } else if (attachType == 'd') {
            // pid & libscope need to exist before moving forward
            if (rc == -1) {
                fprintf(stderr, "error: pid %d does not exist\n", pid);
                return EXIT_FAILURE;
            } else if (rc == 0) {
                // proc exists, libscope does not exist.
                fprintf(stderr, "error: pid %d has never been attached\n", pid);
                return EXIT_FAILURE;
            }
            return attachCmd(pid, FALSE);
        } else {
            fprintf(stderr, "error: attach or detach with invalid option\n");
            showUsage(basename(argv[0]));
            return EXIT_FAILURE;
        }
    }

    // Execute and scope the app
    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    void *handle = NULL;
    char *inferior_command = NULL;

    inferior_command = getpath(argv[2]);  // TODO: find the arg
    //printf("%s:%d %s\n", __FUNCTION__, __LINE__, inferior_command);
    if (!inferior_command) {
        fprintf(stderr,"%s could not find or execute command `%s`.  Exiting.\n", argv[0], argv[optind]);
        exit(EXIT_FAILURE);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }
    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", scopeLibPath, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }

        //printf("%s:%d %d: %s %s %s %s\n", __FUNCTION__, __LINE__,
        //       argc, argv[0], argv[1], argv[2], argv[3]);
        execve(inferior_command, &argv[2], environ); // TODO
        perror("execve");
        goto err;
    }

    // Static executable path
    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        execve(argv[0], argv, environ);
    }

    program_invocation_short_name = basename(argv[1]);  // TODO

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(argv[optind], &argv[optind], environ);
    }

    if ((handle = dlopen(scopeLibPath, RTLD_LAZY)) == NULL) {
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        goto err;
    }

    sys_exec(ebuf, inferior_command, argc-optind, &argv[optind], env);

    /*
     * We should not return from sys_exec unless there was an error loading the static exec.
     * In this case, just start the exec without being scoped.
     * Was wondering if we should free the mapped elf image.
     * But, since we exec on failure to load, it doesn't matter.
     */
    execve(argv[optind], &argv[optind], environ);

err:
    if (ebuf) free(ebuf);
    exit(EXIT_FAILURE);
}