/*
 * Load and run static executables
 *
 * objcopy -I binary -O elf64-x86-64 -B i386 ./lib/linux/libscope.so ./lib/linux/libscope.o
 * gcc -Wall -g src/scope.c -ldl -lrt -o scope ./lib/linux/libscope.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <limits.h>
#include <errno.h>

#include "fn.h"
#include "dbg.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include "os.h"
#include "utils.h"
#include "inject.h"

#define DEVMODE 0
#define __NR_memfd_create   319
#define _MFD_CLOEXEC		0x0001U
#define SHM_NAME            "libscope"
#define PARENT_PROC_NAME "start_scope"
#define GO_ENV_VAR "GODEBUG"
#define GO_ENV_SERVER_VALUE "http2server"
#define GO_ENV_CLIENT_VALUE "http2client"

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

typedef struct {
    char *path;
    char *shm_name;
    int fd;
    int use_memfd;
} libscope_info_t;

// Wrapper to call memfd_create syscall
static inline int _memfd_create(const char *name, unsigned int flags) {
	return syscall(__NR_memfd_create, name, flags);
}

static const char scope_help_overview[] =
"    OVERVIEW:\n"
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
"    default destination is the file `/tmp/scope.log`.\n"
"\n";

static const char scope_help_configuration[] =
"    CONFIGURATION:\n"
"    Configuration File:\n"
"       A YAML config file (named scope.yml) enables control of all available\n"
"       settings. The config file is optional. Environment variables take\n"
"       precedence over settings in a config file.\n"
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
"        \n"
"    Environment Variables:\n"
"    SCOPE_CONF_PATH\n"
"        Directly specify config file's location and name.\n"
"        Used only at start time.\n"
"    SCOPE_HOME\n"
"        Specify a directory from which conf/scope.yml or ./scope.yml can\n"
"        be found. Used only at start time, and only if SCOPE_CONF_PATH does\n"
"        not exist. For more info, see Config File Resolution below.\n"
"    SCOPE_METRIC_ENABLE\n"
"        Single flag to make it possible to disable all metric output.\n"
"        true,false  Default is true.\n"
"    SCOPE_METRIC_VERBOSITY\n"
"        0-9 are valid values. Default is 4.\n"
"        For more info, see Metric Verbosity below.\n"
"    SCOPE_METRIC_DEST\n"
"        Default is udp://localhost:8125\n"
"        Format is one of:\n"
"            file:///tmp/output.log   (file://stdout, file://stderr are\n"
"                                      special allowed values)\n"
"            udp://<server>:<123>         (<server> is servername or address;\n"
"                                      <123> is port number or service name)\n"
"    SCOPE_METRIC_FORMAT\n"
"        statsd, ndjson\n"
"        Default is statsd.\n"
"    SCOPE_STATSD_PREFIX\n"
"        Specify a string to be prepended to every scope metric.\n"
"    SCOPE_STATSD_MAXLEN\n"
"        Default is 512.\n"
"    SCOPE_SUMMARY_PERIOD\n"
"        Number of seconds between output summarizations. Default is 10.\n"
"    SCOPE_EVENT_ENABLE\n"
"        Single flag to make it possible to disable all event output.\n"
"        true,false  Default is true.\n"
"    SCOPE_EVENT_DEST\n"
"        Same format as SCOPE_METRIC_DEST above.\n"
"        Default is tcp://localhost:9109\n"
"    SCOPE_EVENT_FORMAT\n"
"        ndjson\n"
"        Default is ndjson.\n"
"    SCOPE_EVENT_LOGFILE\n"
"        Create events from writes to log files.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_LOGFILE_NAME\n"
"        An extended regex to filter log file events by file name.\n"
"        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*log.*\n"
"    SCOPE_EVENT_LOGFILE_VALUE\n"
"        An extended regex to filter log file events by field value.\n"
"        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*\n"
"    SCOPE_EVENT_CONSOLE\n"
"        Create events from writes to stdout, stderr.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_CONSOLE_NAME\n"
"        An extended regex to filter console events by event name.\n"
"        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"    SCOPE_EVENT_CONSOLE_VALUE\n"
"        An extended regex to filter console events by field value.\n"
"        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"    SCOPE_EVENT_METRIC\n"
"        Create events from metrics.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_METRIC_NAME\n"
"        An extended regex to filter metric events by event name.\n"
"        Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_METRIC_FIELD\n"
"        An extended regex to filter metric events by field name.\n"
"        Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_METRIC_VALUE\n"
"        An extended regex to filter metric events by field value.\n"
"        Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_HTTP\n"
"        Create events from HTTP headers.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_HTTP_NAME\n"
"        An extended regex to filter http events by event name.\n"
"        Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_EVENT_HTTP_FIELD\n"
"        An extended regex to filter http events by field name.\n"
"        Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_EVENT_HTTP_VALUE\n"
"        An extended regex to filter http events by field value.\n"
"        Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_EVENT_HTTP_HEADER\n"
"        An extended regex that defines user defined headers\n"
"        that will be extracted. Default is NULL\n"
"    SCOPE_EVENT_NET\n"
"        Create events describing network connectivity.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_NET_NAME\n"
"        An extended regex to filter network events by event name.\n"
"        Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"    SCOPE_EVENT_NET_FIELD\n"
"        An extended regex to filter network events by field name.\n"
"        Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"    SCOPE_EVENT_NET_VALUE\n"
"        An extended regex to filter network events by field value.\n"
"        Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"    SCOPE_EVENT_FS\n"
"        Create events describing file connectivity.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_FS_NAME\n"
"        An extended regex to filter file events by event name.\n"
"        Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"    SCOPE_EVENT_FS_FIELD\n"
"        An extended regex to filter file events by field name.\n"
"        Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"    SCOPE_EVENT_FS_VALUE\n"
"        An extended regex to filter file events by field value.\n"
"        Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"    SCOPE_EVENT_DNS\n"
"        Create events describing DNS activity.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_DNS_NAME\n"
"        An extended regex to filter dns events by event name.\n"
"        Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"    SCOPE_EVENT_DNS_FIELD\n"
"        An extended regex to filter DNS events by field name.\n"
"        Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"    SCOPE_EVENT_DNS_VALUE\n"
"        An extended regex to filter dns events by field value.\n"
"        Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"    SCOPE_EVENT_MAXEPS\n"
"        Limits number of events that can be sent in a single second.\n"
"        0 is 'no limit'; 10000 is the default.\n"
"    SCOPE_ENHANCE_FS\n"
"        Controls whether uid, gid, and mode are captured for each open.\n"
"        Used only if SCOPE_EVENT_FS is true. true,false Default is true.\n"
"    SCOPE_LOG_LEVEL\n"
"        debug, info, warning, error, none. Default is error.\n"
"    SCOPE_LOG_DEST\n"
"        same format as SCOPE_METRIC_DEST above.\n"
"        Default is file:///tmp/scope.log\n"
"    SCOPE_TAG_\n"
"        Specify a tag to be applied to every metric and event.\n"
"        Environment variable expansion is available, \n"
"        e.g.: SCOPE_TAG_user=$USER\n"
"    SCOPE_CMD_DIR\n"
"        Specifies a directory to look for dynamic configuration files.\n"
"        See Dynamic Configuration below.\n"
"        Default is /tmp\n"
"    SCOPE_PAYLOAD_ENABLE\n"
"        Flag that enables payload capture.  true,false  Default is false.\n"
"    SCOPE_PAYLOAD_DIR\n"
"        Specifies a directory where payload capture files can be written.\n"
"        Default is /tmp\n"
"    SCOPE_CRIBL\n"
"        Defines a connection with Cribl LogStream\n"
"        Default is NULL\n"
"        Format is:\n"
"            tcp://host:port\n"
"            If no port is provided, defaults to 10090\n"
"    SCOPE_CONFIG_EVENT\n"
"        Sends a single process-identifying event, when a transport\n"
"        connection is established.  true,false  Default is true.\n"
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
"\n";

static const char scope_help_metrics[] =
"    METRICS:\n"
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
"    summarization settings.\n"
"\n";

static const char scope_help_events[] =
"    EVENTS:\n"
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
"        the event provides the duration of the DNS operation.\n"
"\n";

static const char scope_help_protocol[] =
"     PROTOCOL DETECTION:\n"
"     Scope can detect any defined network protocol. You provide protocol\n"
"     definitions in a separate YAML config file (which should be named \n"
"     scope_protocol.yml). You describe protocol specifics in one or more regex \n"
"     definitions. PCRE2 regular expressions are supported. You can find a \n"
"     sample config file at\n"
"     https://github.com/criblio/appscope/blob/master/conf/scope_protocol.yml.\n"
"\n"
"     Scope detects binary and string protocols. Detection events, \n"
"     formatted in JSON, are emitted over the event channel. Enable the \n"
"     event metric watch type to allow protocol detection.\n"
"\n"
"     The protocol detection config file should be named scope_protocol.yml.\n"
"     Place the protocol definitions config file (scope_protocol.yml) in the \n"
"     directory defined by the SCOPE_HOME environment variable. If Scope \n"
"     does not find the protocol definitions file in that directory, it will\n"
"     search for it, in the same search order as described for config files.\n"
"\n"
"\n"
"     PAYLOAD EXTRACTION:\n"
"     When enabled, libscope extracts payload data from network operations.\n"
"     Payloads are emitted in binary. No formatting is applied to the data.\n"
"     Payloads are emitted to either a local file or the LogStream channel.\n"
"     Configuration elements for libscope support defining a path for payload\n"
"     data.\n"
"\n";

typedef struct {
    const char* cmd;
    const char* text;
} help_list_t;

static const help_list_t help_list[] = {
    {"overview", scope_help_overview},
    {"configuration", scope_help_configuration},
    {"metrics", scope_help_metrics},
    {"events", scope_help_events},
    {"protocol", scope_help_protocol},
    {NULL, NULL}
};

static void
print_version(char *path)
{
    printf("Scope Version: %s\n\n", SCOPE_VER);

    if (strstr(path, "libscope")) {
        printf("    Usage: LD_PRELOAD=%s <command name>\n", path);
        printf("    For more info: %s help\n\n", path);
    }
}

static void
print_help(void)
{
    int i;
    printf( "Welcome to the help for AppScope!\n\n"
            "    For this message:\n        ldscope --help\n"
            "    To print all help:\n       ldscope --help all\n"
            "    Or to print help by section:\n");
    for (i=0; help_list[i].cmd; i++) {
        printf("        ldscope --help %s\n", help_list[i].cmd);
    }

    putchar('\n');
}

static void
do_help(char *prog, libscope_info_t *info, int argc, char **argv)
{
    int i, j;

    print_version(info->path);

    if (argc > 2) {
        // print all help sections
        if (!strcmp(argv[2], "all")) {
            for (i=0; help_list[i].text; i++) {
                printf("%s", help_list[i].text);
            }
            return;
        }

        // print matching help sections
        for (i=2; i < argc; i++) {
            for (j=0; help_list[j].cmd; j++) {
                if (!strcmp(argv[i], help_list[j].cmd)) {
                    printf("%s", help_list[j].text);
                    return;
                }
            }
        }
    }

    print_help();
}

/**
 * Checks if kernel version is >= 3.17
 */
static int
check_kernel_version(void)
{
    struct utsname buffer;
    char *token;
    char *separator = ".";
    int val;

    if (uname(&buffer)) {
        return 0;
    }
    token = strtok(buffer.release, separator);
    val = atoi(token);
    if (val < 3) {
        return 0;
    } else if (val > 3){
        return 1;
    }

    token = strtok(NULL, separator);
    val = atoi(token);
    return (val < 17) ? 0 : 1;
}

static void
release_libscope(libscope_info_t **info_ptr) {
    if (!info_ptr || !*info_ptr) return;

    libscope_info_t *info = *info_ptr;

    if (info->fd != -1) close(info->fd);
    if (info->shm_name) {
        if (info->fd != -1) shm_unlink(info->shm_name);
        free(info->shm_name);
    }
    if (info->path) free(info->path);
    free(info);
    *info_ptr = NULL;
}

static libscope_info_t *
setup_libscope(bool inject)
{
    libscope_info_t *info = NULL;
    int everything_successful = FALSE;

    if (!(info = calloc(1, sizeof(libscope_info_t)))) {
        perror("setup_libscope:calloc");
        goto err;
    }

    info->fd = -1;
    info->use_memfd = !inject && check_kernel_version();

    if (info->use_memfd) {
        info->fd = _memfd_create(SHM_NAME, _MFD_CLOEXEC);
    } else {
        if (asprintf(&info->shm_name, "%s%i", inject ? SHM_NAME_INJECT : SHM_NAME, getpid()) == -1) {
            perror("setup_libscope:shm_name");
            info->shm_name = NULL; // failure leaves info->shm_name undefined
            goto err;
        }
        info->fd = shm_open(info->shm_name, O_RDWR | O_CREAT, S_IRWXU | S_IROTH | S_IXOTH);
    }
    if (info->fd == -1) {
        perror(info->use_memfd ? "setup_libscope:memfd_create" : "setup_libscope:shm_open");
        goto err;
    }

    size_t libsize = (size_t) (&_binary___lib_linux_libscope_so_end - &_binary___lib_linux_libscope_so_start);
    if (write(info->fd, &_binary___lib_linux_libscope_so_start, libsize) != libsize) {
        perror("setup_libscope:write");
        goto err;
    }

    int rv;
    if (info->use_memfd) {
        rv = asprintf(&info->path, "/proc/%i/fd/%i", getpid(), info->fd);
    } else {
        rv = asprintf(&info->path, "/dev/shm/%s", info->shm_name);
    }
    if (rv == -1) {
        perror("setup_libscope:path");
        info->path = NULL; // failure leaves info->path undefined
        goto err;
    }

/*
 * DEVMODE is here only to help with gdb. The debugger has
 * a problem reading symbols from a /proc pathname.
 * This is expected to be enabled only by developers and
 * only when using the debugger.
 */
#if DEVMODE == 1
    asprintf(&info->path, "./lib/linux/libscope.so");
    printf("LD_PRELOAD=%s\n", info->path);
#endif

    everything_successful = TRUE;

err:
    if (!everything_successful) release_libscope(&info);
    return info;
}

// If possible, we want to set GODEBUG=http2server=0,http2client=0
// This tells go not to upgrade to http2, which allows
// our http1 protocol capture stuff to do it's thing.
// We consider this temporary, because when we support http2
// it will not be necessary.
static void
setGoHttpEnvVariable(void)
{
    if (checkEnv("SCOPE_GO_HTTP1", "false") == TRUE) return;

    char *cur_val = getenv(GO_ENV_VAR);

    // If GODEBUG isn't set, try to set it to http2server=0,http2client=0
    if (!cur_val) {
        if (setenv(GO_ENV_VAR, GO_ENV_SERVER_VALUE "=0," GO_ENV_CLIENT_VALUE "=0", 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        return;
    }

    // GODEBUG is set.
    // If http2server wasn't specified, let's append ",http2server=0"
    if (!strstr(cur_val, GO_ENV_SERVER_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_SERVER_VALUE) == -1)) {
            perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) free(new_val);
    }

    cur_val = getenv(GO_ENV_VAR);

    // If http2client wasn't specified, let's append ",http2client=0"
    if (!strstr(cur_val, GO_ENV_CLIENT_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_CLIENT_VALUE) == -1)) {
            perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) free(new_val);
    }
}

int
main(int argc, char **argv, char **env)
{
    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    pid_t pid;
    void *handle = NULL;
    libscope_info_t *info;
    bool attach = FALSE;

    // Use dlsym to get addresses for everything in g_fn
    initFn();
    setPidEnv(getpid());

    if (argc == 3 && (strncmp(argv[1], "--attach", 8) == 0)) {
        attach = TRUE;
    }

    info = setup_libscope(attach);
    if (!info) {
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    //check command line arguments 
    char *scope_cmd = argv[0];
    if (strncmp(argv[1], "--help", 6) == 0) {
        do_help(scope_cmd, info, argc, argv);
        exit(EXIT_FAILURE);
    }

    if (attach) {
        int pid = atoi(argv[2]);
        printf("Attaching to process %d\n", pid);
        injectScope(pid, info->path);
        return 0;
    }

    char *inferior_command = getpath(argv[1]);
    if (!inferior_command) {
        fprintf(stderr,"%s could not find or execute command `%s`.  Exiting.\n", scope_cmd, argv[1]);
        exit(EXIT_FAILURE);
    }
    argv[1] = inferior_command; // update args with resolved inferior_command

    // before processing, try to set SCOPE_EXEC_PATH for execve
    char *sep;
    if (osGetExePath(&sep) == 0) {
        // doesn't overwrite an existing env var if already set
        setenv("SCOPE_EXEC_PATH", sep, 0);
        free(sep);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }

        setGoHttpEnvVariable();

    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", info->path, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }
        
        pid = fork();
        if (pid == -1) {
            perror("fork");
            goto err;
        } else if (pid > 0) {
            int status;
            int ret;
            do {
                ret = waitpid(pid, &status, 0);
            } while (ret == -1 && errno == EINTR);

            release_libscope(&info);
            if (WIFEXITED(status)) exit(WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        } else {
            execve(inferior_command, &argv[1], environ);
            perror("execve");
            goto err;
        }
    }

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    // Static executable path
    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        execve(argv[0], argv, environ);
    }

    program_invocation_short_name = basename(argv[1]);

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(argv[1], &argv[1], environ);
    }

    if ((handle = dlopen(info->path, RTLD_LAZY)) == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }

    release_libscope(&info);

    sys_exec(ebuf, inferior_command, argc, argv, env);

    return 0;
err:
    release_libscope(&info);
    if (ebuf) free(ebuf);
    exit(EXIT_FAILURE);
}
