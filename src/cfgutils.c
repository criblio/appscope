#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pwd.h>
#include "pcre2posix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cfgutils.h"
#include "dbg.h"
#include "mtcformat.h"
#include "scopetypes.h"
#include "com.h"
#include "utils.h"

#ifndef NO_YAML
#include "yaml.h"
#endif

#define METRIC_NODE          "metric"
#define ENABLE_NODE              "enable"
#define FORMAT_NODE              "format"
#define TYPE_NODE                    "type"
#define STATSDPREFIX_NODE            "statsdprefix"
#define STATSDMAXLEN_NODE            "statsdmaxlen"
#define VERBOSITY_NODE               "verbosity"
#define TAGS_NODE                    "tags"
#define TRANSPORT_NODE           "transport"
#define TYPE_NODE                    "type"
#define HOST_NODE                    "host"
#define PORT_NODE                    "port"
#define PATH_NODE                    "path"
#define BUFFERING_NODE               "buffering"

#define LIBSCOPE_NODE        "libscope"
#define LOG_NODE                 "log"
#define LEVEL_NODE                   "level"
#define TRANSPORT_NODE               "transport"
#define SUMMARYPERIOD_NODE       "summaryperiod"
#define COMMANDDIR_NODE          "commanddir"
#define CFGEVENT_NODE            "configevent"

#define EVENT_NODE           "event"
#define TRANSPORT_NODE           "transport"
#define FORMAT_NODE              "format"
#define TYPE_NODE                    "type"
#define MAXEPS_NODE                  "maxeventpersec"
#define ENHANCEFS_NODE               "enhancefs"
#define WATCH_NODE               "watch"
#define TYPE_NODE                    "type"
#define NAME_NODE                    "name"
#define FIELD_NODE                   "field"
#define VALUE_NODE                   "value"
#define EX_HEADERS                   "headers"

#define PAYLOAD_NODE          "payload"
#define ENABLE_NODE              "enable"
#define DIR_NODE                 "dir"


enum_map_t formatMap[] = {
    {"statsd",                CFG_FMT_STATSD},
    {"ndjson",                CFG_FMT_NDJSON},
    {NULL,                    -1}
};

enum_map_t transportTypeMap[] = {
    {"udp",                   CFG_UDP},
    {"tcp",                   CFG_TCP},
    {"unix",                  CFG_UNIX},
    {"file",                  CFG_FILE},
    {"syslog",                CFG_SYSLOG},
    {"shm",                   CFG_SHM},
    {NULL,                   -1}
};

enum_map_t logLevelMap[] = {
    {"debug",                 CFG_LOG_DEBUG},
    {"info",                  CFG_LOG_INFO},
    {"warning",               CFG_LOG_WARN},
    {"error",                 CFG_LOG_ERROR},
    {"none",                  CFG_LOG_NONE},
    {"trace",                 CFG_LOG_TRACE},
    {NULL,                    -1}
};

enum_map_t bufferMap[] = {
    {"line",                  CFG_BUFFER_LINE},
    {"full",                  CFG_BUFFER_FULLY},
    {NULL,                    -1}
};

enum_map_t watchTypeMap[] = {
    {"file",                  CFG_SRC_FILE},
    {"console",               CFG_SRC_CONSOLE},
    {"syslog",                CFG_SRC_SYSLOG},
    {"metric",                CFG_SRC_METRIC},
    {"http",                  CFG_SRC_HTTP},
    {"net",                   CFG_SRC_NET},
    {"fs",                    CFG_SRC_FS},
    {"dns",                   CFG_SRC_DNS},
    {NULL,                    -1}
};

enum_map_t boolMap[] = {
    {"true",                  TRUE},
    {"false",                 FALSE},
    {NULL,                    -1}
};

// forward declarations
void cfgMtcEnableSetFromStr(config_t*, const char*);
void cfgMtcFormatSetFromStr(config_t*, const char*);
void cfgMtcStatsDPrefixSetFromStr(config_t*, const char*);
void cfgMtcStatsDMaxLenSetFromStr(config_t*, const char*);
void cfgMtcPeriodSetFromStr(config_t*, const char*);
void cfgCmdDirSetFromStr(config_t*, const char*);
void cfgConfigEventSetFromStr(config_t*, const char*);
void cfgEvtEnableSetFromStr(config_t*, const char*);
void cfgEventFormatSetFromStr(config_t*, const char*);
void cfgEvtRateLimitSetFromStr(config_t*, const char*);
void cfgEnhanceFsSetFromStr(config_t*, const char*);
void cfgEvtFormatValueFilterSetFromStr(config_t*, watch_t, const char*);
void cfgEvtFormatFieldFilterSetFromStr(config_t*, watch_t, const char*);
void cfgEvtFormatNameFilterSetFromStr(config_t*, watch_t, const char*);
void cfgEvtFormatSourceEnabledSetFromStr(config_t*, watch_t, const char*);
void cfgMtcVerbositySetFromStr(config_t*, const char*);
void cfgTransportSetFromStr(config_t*, which_transport_t, const char*);
void cfgCustomTagAddFromStr(config_t*, const char*, const char*);
void cfgLogLevelSetFromStr(config_t*, const char*);
void cfgPayEnableSetFromStr(config_t*, const char*);
void cfgPayDirSetFromStr(config_t*, const char*);
void cfgEvtFormatHeaderSetFromStr(config_t *, const char *);
static void cfgSetFromFile(config_t *, const char *);
static void cfgEvtFormatLogStreamSetFromStr(config_t *, const char *);

// These global variables limits us to only reading one config file at a time...
// which seems fine for now, I guess.
static which_transport_t transport_context;
static watch_t watch_context;
static regex_t* g_regex = NULL;
static char g_logmsg[1024] = {};

static char*
cfgPathSearch(const char* cfgname)
{
    // in priority order:
    //   1) $SCOPE_HOME/conf/scope.yml
    //   2) $SCOPE_HOME/scope.yml
    //   3) /etc/scope/scope.yml
    //   4) ~/conf/scope.yml
    //   5) ~/scope.yml
    //   6) ./conf/scope.yml
    //   7) ./scope.yml

    char path[1024]; // Somewhat arbitrary choice for MAX_PATH
    static int (*ni_access)(const char *pathname, int mode);
    if (!ni_access) ni_access = dlsym(RTLD_NEXT, "access");
    if (!ni_access) return NULL;

    const char* homedir = getenv("HOME");
    const char* scope_home = getenv("SCOPE_HOME");
    if (scope_home &&
       (snprintf(path, sizeof(path), "%s/conf/%s", scope_home, cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (scope_home &&
       (snprintf(path, sizeof(path), "%s/%s", scope_home, cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "/etc/scope/%s", cfgname) > 0 ) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (homedir &&
       (snprintf(path, sizeof(path), "%s/conf/%s", homedir, cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (homedir &&
       (snprintf(path, sizeof(path), "%s/%s", homedir, cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "./conf/%s", cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "./%s", cfgname) > 0) &&
        !ni_access(path, R_OK)) {
        return realpath(path, NULL);
    }

    return NULL;
}

char*
cfgPath(void)
{
    const char* envPath = getenv("SCOPE_CONF_PATH");

    // If SCOPE_CONF_PATH is set, and the file can be opened, use it.
    char* path;
    if (envPath && (path = strdup(envPath))) {

        // ni for "non-interposed"...  a direct glibc call without scope.
        FILE *(*ni_fopen)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
        int (*ni_fclose)(FILE *) = dlsym(RTLD_NEXT, "fclose");
        FILE* f = NULL;
        if (ni_fopen && ni_fclose && (f = ni_fopen(path, "rb"))) {
            ni_fclose(f);
            return path;
        }

        // Couldn't open the file
        free(path);
    }

    // Otherwise, search for scope.yml
    return cfgPathSearch(CFG_FILE_NAME);
}

char *
protocolPath(void)
{
    return cfgPathSearch(PROTOCOL_FILE_NAME);
}

static void
processCustomTag(config_t* cfg, const char* e, const char* value)
{
    char name_buf[1024];
    strncpy(name_buf, e, sizeof(name_buf));

    char* name = name_buf + strlen("SCOPE_TAG_");

    // convert the "=" to a null delimiter for the name
    char* end = strchr(name, '=');
    if (end) {
        *end = '\0';
        cfgCustomTagAddFromStr(cfg, name, value);
    }

}

static regex_t*
envRegex()
{
    if (g_regex) return g_regex;

    if (!(g_regex = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        return g_regex;
    }

    if (regcomp(g_regex, "\\$[a-zA-Z0-9_]+", REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        free(g_regex);
        g_regex = NULL;
    }
    return g_regex;
}

// For testing.  Never executed in the real deal.
// Exists to make valgrind more readable when analyzing tests.
void
envRegexFree(void** state)
{
    if (!g_regex) return;
    regfree(g_regex);
    free(g_regex);
}

static char*
doEnvVariableSubstitution(const char* value)
{
    if (!value) return NULL;

    regex_t* re = envRegex();
    regmatch_t match = {0};

    int out_size = strlen(value) + 1;
    char* outval = calloc(1, out_size);
    if (!outval) {
        DBG("%s", value);
        return NULL;
    }

    char* outptr = outval;       // "tail" pointer where text can be appended
    char* inptr = (char*) value; // "current" pointer as we scan through value

    while (re && !regexec_wrapper(re, inptr, 1, &match, 0)) {

        int match_size = match.rm_eo - match.rm_so;

        // if the character before the match is '\', don't do substitution
        char* escape_indicator = &inptr[match.rm_so - 1];
        int escaped = (escape_indicator >= value) && (*escape_indicator == '\\');

        if (escaped) {
            // copy the part before the match, except the escape char '\'
            outptr = stpncpy(outptr, inptr, match.rm_so - 1);
            // copy the matching env variable name
            outptr = stpncpy(outptr, &inptr[match.rm_so], match_size);
            // move to the next part of the input value
            inptr = &inptr[match.rm_eo];
            continue;
        }

        // lookup the part that matched to see if we can substitute it
        char env_name[match_size + 1];
        strncpy(env_name, &inptr[match.rm_so], match_size);
        env_name[match_size] = '\0';
        char* env_value = getenv(&env_name[1]); // offset of 1 skips the $

        // Grow outval buffer any time env_value is bigger than env_name
        int size_growth = (!env_value) ? 0 : strlen(env_value) - match_size;
        if (size_growth > 0) {
            char* new_outval = realloc (outval, out_size + size_growth);
            if (new_outval) {
                out_size += size_growth;
                outptr = new_outval + (outptr - outval);
                outval = new_outval;
            } else {
                DBG("%s", value);
                free(outval);
                return NULL;
            }
        }

        // copy the part before the match
        outptr = stpncpy(outptr, inptr, match.rm_so);
        // either copy in the env value or the variable that wasn't found
        outptr = stpcpy(outptr, (env_value) ? env_value : env_name);
        // move to the next part of the input value
        inptr = &inptr[match.rm_eo];
    }

    // copy whatever is left
    strcpy(outptr, inptr);

    return outval;
}

static void
processCmdDebug(const char* path)
{
    if (!path || !path[0]) return;

    static FILE *(*ni_fopen)(const char *, const char *);
    static int (*ni_fclose)(FILE*);
    if (!ni_fopen) ni_fopen = dlsym(RTLD_NEXT, "fopen");
    if (!ni_fclose) ni_fclose = dlsym(RTLD_NEXT, "fclose");
    if (!ni_fopen || !ni_fclose) return;

    FILE* f;
    if (!(f = ni_fopen(path, "a"))) return;
    dbgDumpAll(f);
    ni_fclose(f);
}

static void
processReloadConfig(config_t *cfg, const char* value)
{
    if (!cfg || !value) return;
    unsigned int enable = strToVal(boolMap, value);
    if (enable != TRUE) return;

    char *path = cfgPath();
    cfgSetFromFile(cfg, path);
    if (path) free(path);

    cfgProcessEnvironment(cfg);

    if (cfgLogStream(cfg)) {
        cfgLogStreamDefault(cfg);
    }
}

static int
startsWith(const char* string, const char* substring)
{
    return (strncmp(string, substring, strlen(substring)) == 0);
}

//
// An example of this format: SCOPE_STATSD_MAXLEN=1024
//
// For completeness, scope env vars that are not processed here:
//    SCOPE_CONF_PATH (only used on startup to specify cfg file)
//    SCOPE_HOME      (only used on startup for searching for cfg file)
static void
processEnvStyleInput(config_t *cfg, const char *env_line)
{

    if (!cfg || !env_line) return;

    char* env_ptr, *value;
    if (!(env_ptr = strchr(env_line, '='))) return;
    if (!(value = doEnvVariableSubstitution(&env_ptr[1]))) return;

    if (startsWith(env_line, "SCOPE_METRIC_ENABLE")) {
        cfgMtcEnableSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_METRIC_FORMAT")) {
        cfgMtcFormatSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_PREFIX")) {
        cfgMtcStatsDPrefixSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_MAXLEN")) {
        cfgMtcStatsDMaxLenSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_SUMMARY_PERIOD")) {
        cfgMtcPeriodSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_CMD_DIR")) {
        cfgCmdDirSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_CONFIG_EVENT")) {
        cfgConfigEventSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_METRIC_VERBOSITY")) {
        cfgMtcVerbositySetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_LOG_LEVEL")) {
        cfgLogLevelSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_METRIC_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_MTC, value);
    } else if (startsWith(env_line, "SCOPE_LOG_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_LOG, value);
    } else if (startsWith(env_line, "SCOPE_TAG_")) {
        processCustomTag(cfg, env_line, value);
    } else if (startsWith(env_line, "SCOPE_PAYLOAD_ENABLE")) {
        cfgPayEnableSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_PAYLOAD_DIR")) {
        cfgPayDirSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_CMD_DBG_PATH")) {
        processCmdDebug(value);
    } else if (startsWith(env_line, "SCOPE_CONF_RELOAD")) {
        processReloadConfig(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_CTL, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_ENABLE")) {
        cfgEvtEnableSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FORMAT")) {
        cfgEventFormatSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_MAXEPS")) {
        cfgEvtRateLimitSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_ENHANCE_FS")) {
        cfgEnhanceFsSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_HTTP_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_HTTP_HEADER")) {
        cfgEvtFormatHeaderSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_NET_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FS_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DNS_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_HTTP_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_NET_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FS_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DNS_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_HTTP_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_NET_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FS_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DNS_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_HTTP")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_NET")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FS")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DNS")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (startsWith(env_line, "SCOPE_LOGSTREAM")) {
        cfgEvtFormatLogStreamSetFromStr(cfg, value);
    }


    free(value);
}


extern char** environ;

void
cfgProcessEnvironment(config_t* cfg)
{
    if (!cfg) return;
    char* e = NULL;
    int i = 0;
    while ((e = environ[i++])) {
        // Everything we care about starts with a capital 'S'.  Skip 
        // everything else for performance.
        if (e[0] != 'S') continue;

        // Some things should only be processed as commands, not as
        // environment variables.  Skip them here.
        if (startsWith(e, "SCOPE_CMD_DBG_PATH")) continue;
        if (startsWith(e, "SCOPE_CONF_RELOAD")) continue;

        // Process everything else.
        processEnvStyleInput(cfg, e);
    }
}

void
cfgProcessCommands(config_t* cfg, FILE* file)
{
    if (!cfg || !file) return;

    static ssize_t (*ni_getline)(char **, size_t *, FILE *);
    if (!ni_getline) ni_getline = dlsym(RTLD_NEXT, "getline");
    if (!ni_getline) return;

    char* e = NULL;
    size_t n = 0;

    while (ni_getline(&e, &n, file) != -1) {
        e[strcspn(e, "\r\n")] = '\0'; //overwrite first \r or \n with null
        processEnvStyleInput(cfg, e);
        e[0] = '\0';
    }

    if (e) free(e);
}

void
cfgMtcEnableSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgMtcEnableSet(cfg, strToVal(boolMap, value));
}

void
cfgMtcFormatSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgMtcFormatSet(cfg, strToVal(formatMap, value));
}

void
cfgMtcStatsDPrefixSetFromStr(config_t* cfg, const char* value)
{
    // A little silly to define passthrough function
    // but this keeps the interface consistent.
    cfgMtcStatsDPrefixSet(cfg, value);
}

void
cfgMtcStatsDMaxLenSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgMtcStatsDMaxLenSet(cfg, x);
}

void
cfgMtcPeriodSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgMtcPeriodSet(cfg, x);
}

void
cfgCmdDirSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgCmdDirSet(cfg, value);
}

void
cfgConfigEventSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgSendProcessStartMsgSet(cfg, strToVal(boolMap, value));
}

void
cfgEvtEnableSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgEvtEnableSet(cfg, strToVal(boolMap, value));
}

void
cfgEventFormatSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    // only ndjson is valid
    cfgEventFormatSet(cfg, CFG_FMT_NDJSON);
    //cfgEventFormatSet(cfg, strToVal(formatMap, value));
}

void
cfgEvtRateLimitSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgEvtRateLimitSet(cfg, x);
}

void
cfgEnhanceFsSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgEnhanceFsSet(cfg, strToVal(boolMap, value));
}

void
cfgEvtFormatValueFilterSetFromStr(config_t* cfg, watch_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEvtFormatValueFilterSet(cfg, src, value);
}

void
cfgEvtFormatFieldFilterSetFromStr(config_t* cfg, watch_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEvtFormatFieldFilterSet(cfg, src, value);
}

void
cfgEvtFormatNameFilterSetFromStr(config_t* cfg, watch_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEvtFormatNameFilterSet(cfg, src, value);
}

void
cfgEvtFormatHeaderSetFromStr(config_t *cfg, const char *value)
{
    if (!cfg || !value) return;
    cfgEvtFormatHeaderSet(cfg, value);
}

void
cfgEvtFormatSourceEnabledSetFromStr(config_t* cfg, watch_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEvtFormatSourceEnabledSet(cfg, src, strToVal(boolMap, value));
}

void
cfgMtcVerbositySetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgMtcVerbositySet(cfg, x);
}

void
cfgTransportSetFromStr(config_t *cfg, which_transport_t t, const char *value)
{
    if (!cfg || !value) return;

    // see if value starts with udp:// or file://
    if (value == strstr(value, "udp://")) {

        // copied to avoid directly modifing the process's env variable
        char value_cpy[1024];
        strncpy(value_cpy, value, sizeof(value_cpy));

        char *host = value_cpy + strlen("udp://");

        // convert the ':' to a null delimiter for the host
        // and move port past the null
        char *port = strrchr(host, ':');
        if (!port) return;  // port is *required*
        *port = '\0';
        port++;

        cfgTransportTypeSet(cfg, t, CFG_UDP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);

    } else if (value == strstr(value, "tcp://")) {

        // copied to avoid directly modifing the process's env variable
        char value_cpy[1024];
        strncpy(value_cpy, value, sizeof(value_cpy));

        char *host = value_cpy + strlen("tcp://");

        // convert the ':' to a null delimiter for the host
        // and move port past the null
        char *port = strrchr(host, ':');
        if (!port) return;  // port is *required*
        *port = '\0';
        port++;

        cfgTransportTypeSet(cfg, t, CFG_TCP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);

    } else if (value == strstr(value, "file://")) {
        const char *path = value + strlen("file://");
        cfgTransportTypeSet(cfg, t, CFG_FILE);
        cfgTransportPathSet(cfg, t, path);
    } else {
        // LS  is alwyas a TCP connection
        // there is a case where SCOPE_LOGSTREAM="host:port"
        // or SCOPE_LOGSTREAM="host"
        // if SCOPE_LOGSTREAM="tcp://host:port" it's handled above
        // if SCOPE_LOGSTREAM="udp://host:port" it will not connect
        char *host, *port;
        char value_cpy[1024];

        // copied to avoid directly modifing the process's env variable
        strncpy(value_cpy, value, sizeof(value_cpy));
        host = value_cpy;

        if ((port = strrchr(value_cpy, ':'))) {
            *port = '\0';
            port++;
        } else {
            port = strdup(DEFAULT_LS_PORT);
        }

        cfgTransportTypeSet(cfg, t, CFG_TCP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);
    }
}

void
cfgCustomTagAddFromStr(config_t* cfg, const char* name, const char* value)
{
    // A little silly to define passthrough function
    // but this keeps the interface consistent.
    cfgCustomTagAdd(cfg, name, value);
}

void
cfgLogLevelSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgLogLevelSet(cfg, strToVal(logLevelMap, value));
}

void
cfgPayEnableSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgPayEnableSet(cfg, strToVal(boolMap, value));
}

void
cfgPayDirSetFromStr(config_t *cfg, const char *value)
{
    if (!cfg || !value) return;
    cfgPayDirSet(cfg, value);
}

static void
cfgEvtFormatLogStreamSetFromStr(config_t *cfg, const char *value)
{
    cfgLogStreamSet(cfg);
    cfgTransportSetFromStr(cfg, CFG_LS, value);
    cfgTransportSetFromStr(cfg, CFG_CTL, value);
}

#ifndef NO_YAML

#define foreach(pair, pairs) \
    for (pair = pairs.start; pair != pairs.top; pair++)

typedef void (*node_fn)(config_t*, yaml_document_t*, yaml_node_t*);

static char*
stringVal(yaml_node_t* node)
{
    if (!node || (node->type != YAML_SCALAR_NODE)) return NULL;
    const char* nodeStr = (const char*) node->data.scalar.value;
    return doEnvVariableSubstitution(nodeStr);
}

typedef struct {
    yaml_node_type_t type;
    char* key;
    node_fn fn;
} parse_table_t;

static void
processKeyValuePair(parse_table_t* t, yaml_node_pair_t* pair, config_t* config, yaml_document_t* doc)
{
    yaml_node_t* key = yaml_document_get_node(doc, pair->key);
    yaml_node_t* value = yaml_document_get_node(doc, pair->value);
    if (key->type != YAML_SCALAR_NODE) return;

    // printf("key = %s, value = %s\n", key->data.scalar.value,
    //     (value->type == YAML_SCALAR_NODE) ? value->data.scalar.value : "X");

    // Scan through the parse_table_t for a matching type and key
    // If found, call the function that handles that.
    int i;
    for (i=0; t[i].type != YAML_NO_NODE; i++) {
        if ((value->type == t[i].type) &&
            (!strcmp((char*)key->data.scalar.value, t[i].key))) {
            t[i].fn(config, doc, value);
            break;
        }
    }
}

static void
processLevel(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgLogLevelSetFromStr(config, value);
    if (value) free(value);
}

static void
processTransportType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportTypeSet(config, c, strToVal(transportTypeMap, value));
    if (value) free(value);
}

static void
processHost(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportHostSet(config, c, value);
    if (value) free(value);
}

static void
processPort(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportPortSet(config, c, value);
    if (value) free(value);
}

static void
processPath(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportPathSet(config, c, value);
    if (value) free(value);
}

static void
processBuf(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportBufSet(config, c, strToVal(bufferMap, value));
    if (value) free(value);
}

static void
processTransport(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processTransportType},
        {YAML_SCALAR_NODE,    HOST_NODE,            processHost},
        {YAML_SCALAR_NODE,    PORT_NODE,            processPort},
        {YAML_SCALAR_NODE,    PATH_NODE,            processPath},
        {YAML_SCALAR_NODE,    BUFFERING_NODE,       processBuf},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processTransportMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    transport_context = CFG_MTC;
    processTransport(config, doc, node);
}

static void
processTransportLog(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    transport_context = CFG_LOG;
    processTransport(config, doc, node);
}

static void
processTransportCtl(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    transport_context = CFG_CTL;
    processTransport(config, doc, node);
}

static void
processLogging(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    LEVEL_NODE,           processLevel},
        {YAML_MAPPING_NODE,   TRANSPORT_NODE,       processTransportLog},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processTags(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        yaml_node_t* key = yaml_document_get_node(doc, pair->key);
        yaml_node_t* value = yaml_document_get_node(doc, pair->value);
        if (key->type != YAML_SCALAR_NODE) continue;
        if (value->type != YAML_SCALAR_NODE) continue;

        char* key_str = stringVal(key);
        char* value_str = stringVal(value);

        cfgCustomTagAddFromStr(config, key_str, value_str);
        if (key_str) free(key_str);
        if (value_str) free(value_str);
    }
}

static void
processFormatTypeMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcFormatSetFromStr(config, value);
    if (value) free(value);
}

static void
processFormatTypeEvent(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEventFormatSetFromStr(config, value);
    if (value) free(value);
}

static void
processFormatMaxEps(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEvtRateLimitSetFromStr(config, value);
    if (value) free(value);
}

static void
processEnhanceFs(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEnhanceFsSetFromStr(config, value);
    if (value) free(value);
}

static void
processStatsDPrefix(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcStatsDPrefixSetFromStr(config, value);
    if (value) free(value);
}

static void
processStatsDMaxLen(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcStatsDMaxLenSetFromStr(config, value);
    if (value) free(value);
}

static void
processVerbosity(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcVerbositySetFromStr(config, value);
    if (value) free(value);
}

static void
processMetricEnable(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcEnableSetFromStr(config, value);
    if (value) free(value);
}

static void
processFormat(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processFormatTypeMetric},
        {YAML_SCALAR_NODE,    STATSDPREFIX_NODE,    processStatsDPrefix},
        {YAML_SCALAR_NODE,    STATSDMAXLEN_NODE,    processStatsDMaxLen},
        {YAML_SCALAR_NODE,    VERBOSITY_NODE,       processVerbosity},
        {YAML_MAPPING_NODE,   TAGS_NODE,            processTags},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processSummaryPeriod(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcPeriodSetFromStr(config, value);
    if (value) free(value);
}

static void
processCommandDir(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgCmdDirSetFromStr(config, value);
    if (value) free(value);
}

static void
processConfigEvent(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgConfigEventSetFromStr(config, value);
    if (value) free(value);
}

static void
processMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processMetricEnable},
        {YAML_MAPPING_NODE,   FORMAT_NODE,          processFormat},
        {YAML_MAPPING_NODE,   TRANSPORT_NODE,       processTransportMetric},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processEvtEnable(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEvtEnableSetFromStr(config, value);
    if (value) free(value);
}

static void
processEvtFormat(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processFormatTypeEvent},
        {YAML_SCALAR_NODE,    MAXEPS_NODE,          processFormatMaxEps},
        {YAML_SCALAR_NODE,    ENHANCEFS_NODE,       processEnhanceFs},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processWatchType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    watch_context = strToVal(watchTypeMap, value);
    cfgEvtFormatSourceEnabledSet(config, watch_context, 1);
    if (value) free(value);
}

static void
processWatchName(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatNameFilterSetFromStr(config, watch_context, value);
    if (value) free(value);
}

static void
processWatchField(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatFieldFilterSetFromStr(config, watch_context, value);
    if (value) free(value);
}

static void
processWatchValue(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatValueFilterSetFromStr(config, watch_context, value);
    if (value) free(value);
}

static void
processWatchHeader(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    // watch header is only valid for http
    if (watch_context != CFG_SRC_HTTP) return;

    char *value = stringVal(node);
    cfgEvtFormatHeaderSet(config, value);
    if (value) free(value);
}

static int
isWatchType(yaml_document_t* doc, yaml_node_pair_t* pair)
{
    yaml_node_t* key = yaml_document_get_node(doc, pair->key);
    if (!key || (key->type != YAML_SCALAR_NODE)) return 0;
    return !strcmp((char*)key->data.scalar.value, TYPE_NODE);
}

static void
processSource(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processWatchType},
        {YAML_SCALAR_NODE,    NAME_NODE,            processWatchName},
        {YAML_SCALAR_NODE,    FIELD_NODE,           processWatchField},
        {YAML_SCALAR_NODE,    VALUE_NODE,           processWatchValue},
        {YAML_SCALAR_NODE,    EX_HEADERS,           processWatchHeader},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    watch_context = CFG_SRC_MAX;

    yaml_node_pair_t* pair;
    // process type first
    foreach(pair, node->data.mapping.pairs) {
        if (!isWatchType(doc, pair)) continue;
        processKeyValuePair(t, pair, config, doc);
        break;
    }
    // Then process everything else
    foreach(pair, node->data.mapping.pairs) {
        if (isWatchType(doc, pair)) continue;
        processKeyValuePair(t, pair, config, doc);
    }

}

static void
processWatch(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    // Type can be scalar or sequence.
    // It will be scalar there are zero entries, in which case we
    // clear all values and return.
    if ((node->type != YAML_SEQUENCE_NODE) &&
       (node->type != YAML_SCALAR_NODE)) return;

    // absence of one of these values means to clear it.
    // clear them all, then set values for whatever we find.
    watch_t x;
    for (x = CFG_SRC_FILE; x<CFG_SRC_MAX; x++) {
        cfgEvtFormatSourceEnabledSet(config, x, 0);
    }

    if (node->type != YAML_SEQUENCE_NODE) return;
    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* i = yaml_document_get_node(doc, *item);
        processSource(config, doc, i);
    }
}

static void
processEvent(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processEvtEnable},
        {YAML_MAPPING_NODE,   TRANSPORT_NODE,       processTransportCtl},
        {YAML_MAPPING_NODE,   FORMAT_NODE,          processEvtFormat},
        {YAML_SEQUENCE_NODE,  WATCH_NODE,           processWatch},
        {YAML_SCALAR_NODE,    WATCH_NODE,           processWatch},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processLibscope(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_MAPPING_NODE,   LOG_NODE,             processLogging},
        {YAML_SCALAR_NODE,    SUMMARYPERIOD_NODE,   processSummaryPeriod},
        {YAML_SCALAR_NODE,    COMMANDDIR_NODE,      processCommandDir},
        {YAML_SCALAR_NODE,    CFGEVENT_NODE,        processConfigEvent},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processPayloadEnable(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    cfgPayEnableSetFromStr(config, value);
    if (value) free(value);
}

static void
processPayloadDir(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    cfgPayDirSetFromStr(config, value);
    if (value) free(value);
}

static void
processPayload(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processPayloadEnable},
        {YAML_SCALAR_NODE,    DIR_NODE,             processPayloadDir},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
setConfigFromDoc(config_t* config, yaml_document_t* doc)
{
    yaml_node_t* node = yaml_document_get_root_node(doc);
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_MAPPING_NODE,   METRIC_NODE,          processMetric},
        {YAML_MAPPING_NODE,   LIBSCOPE_NODE,        processLibscope},
        {YAML_MAPPING_NODE,   PAYLOAD_NODE,         processPayload},
        {YAML_MAPPING_NODE,   EVENT_NODE,           processEvent},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
cfgSetFromFile(config_t *config, const char* path)
{
    FILE* f = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    yaml_parser_t parser;
    yaml_document_t doc;
    // ni for "not-interposed"... a direct glibc call without scope.
    FILE *(*ni_fopen)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
    int (*ni_fclose)(FILE*) = dlsym(RTLD_NEXT, "fclose");

    if (!ni_fopen || !ni_fclose) goto cleanup;

    if (!config) goto cleanup;
    if (!path) goto cleanup;
    f = ni_fopen(path, "rb");
    if (!f) goto cleanup;

    parser_successful = yaml_parser_initialize(&parser);
    if (!parser_successful) goto cleanup;

    yaml_parser_set_input_file(&parser, f);

    doc_successful = yaml_parser_load(&parser, &doc);
    if (!doc_successful) goto cleanup;

    // This is where the magic happens
    setConfigFromDoc(config, &doc);

cleanup:
    if (doc_successful) yaml_document_delete(&doc);
    if (parser_successful) yaml_parser_delete(&parser);
    if (f) ni_fclose(f);
}

config_t*
cfgFromString(const char* string)
{
    config_t* config = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    yaml_parser_t parser;
    yaml_document_t doc;

    if (!string) goto cleanup;

    config = cfgCreateDefault();
    if (!config) goto cleanup;

    parser_successful = yaml_parser_initialize(&parser);
    if (!parser_successful) goto cleanup;

    yaml_parser_set_input_string(&parser, (unsigned char*)string, strlen(string));

    doc_successful = yaml_parser_load(&parser, &doc);
    if (!doc_successful) goto cleanup;

    // This is where the magic happens
    setConfigFromDoc(config, &doc);

cleanup:
    if (doc_successful) yaml_document_delete(&doc);
    if (parser_successful) yaml_parser_delete(&parser);
    return config;
}

config_t*
cfgRead(const char *path)
{
    config_t *config = cfgCreateDefault();
    cfgSetFromFile(config, path);
    return config;
}

#else
config_t*
cfgRead(const char* path)
{
    return cfgCreateDefault();
}
config_t*
cfgFromString(const char* string)
{
    return cfgCreateDefault();
}
#endif

static cJSON*
createTransportJson(config_t* cfg, which_transport_t trans)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
         valToStr(transportTypeMap, cfgTransportType(cfg, trans)))) goto err;

    switch (cfgTransportType(cfg, trans)) {
        case CFG_TCP:
        case CFG_UDP:
            if (!cJSON_AddStringToObjLN(root, HOST_NODE,
                                     cfgTransportHost(cfg, trans))) goto err;
            if (!cJSON_AddStringToObjLN(root, PORT_NODE,
                                     cfgTransportPort(cfg, trans))) goto err;
            break;
        case CFG_UNIX:
            if (!cJSON_AddStringToObjLN(root, PATH_NODE,
                                     cfgTransportPath(cfg, trans))) goto err;
            break;
        case CFG_FILE:
            if (!cJSON_AddStringToObjLN(root, PATH_NODE,
                                     cfgTransportPath(cfg, trans))) goto err;
            if (!cJSON_AddStringToObjLN(root, BUFFERING_NODE,
                 valToStr(bufferMap, cfgTransportBuf(cfg, trans)))) goto err;
            break;
        case CFG_SYSLOG:
        case CFG_SHM:
            break;
        default:
            DBG(NULL);
    }
    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}


static cJSON*
createLogJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* transport;

    if (!(root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(root, LEVEL_NODE,
                     valToStr(logLevelMap, cfgLogLevel(cfg)))) goto err;

    if (!(transport = createTransportJson(cfg, CFG_LOG))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createTagsJson(config_t* cfg)
{
    cJSON* root = NULL;
    if (!(root = cJSON_CreateObject())) goto err;

    custom_tag_t **tags = cfgCustomTags(cfg);
    int i;
    if (tags) {
        for (i=0; tags[i]; i++) {
            if (!(cJSON_AddStringToObject(root, tags[i]->name, tags[i]->value))) {
                DBG("name:%s value:%s", tags[i]->name, tags[i]->value);
            }
        }
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createMetricFormatJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* tags;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
                     valToStr(formatMap, cfgMtcFormat(cfg)))) goto err;
    if (!cJSON_AddStringToObjLN(root, STATSDPREFIX_NODE,
                                    cfgMtcStatsDPrefix(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, STATSDMAXLEN_NODE,
                                    cfgMtcStatsDMaxLen(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, VERBOSITY_NODE,
                                       cfgMtcVerbosity(cfg))) goto err;

    if (!(tags = createTagsJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, TAGS_NODE, tags);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createMetricJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* transport, *format;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
                          valToStr(boolMap, cfgMtcEnable(cfg)))) goto err;

    if (!(transport = createTransportJson(cfg, CFG_MTC))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    if (!(format = createMetricFormatJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, FORMAT_NODE, format);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createWatchObjectJson(config_t* cfg, watch_t src)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
                                    valToStr(watchTypeMap, src))) goto err;
    if (!cJSON_AddStringToObjLN(root, NAME_NODE,
                                   cfgEvtFormatNameFilter(cfg, src))) goto err;
    if (!cJSON_AddStringToObjLN(root, FIELD_NODE,
                                  cfgEvtFormatFieldFilter(cfg, src))) goto err;
    if (!cJSON_AddStringToObjLN(root, VALUE_NODE,
                                  cfgEvtFormatValueFilter(cfg, src))) goto err;
    if (src == CFG_SRC_HTTP) {
        const char *header = cfgEvtFormatHeader(cfg);
        header = (header) ? header : "";
        if (!cJSON_AddStringToObjLN(root, EX_HEADERS, header)) goto err;
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createWatchArrayJson(config_t* cfg)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateArray())) goto err;

    watch_t src;
    for (src = CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        cJSON* item;
        if (!cfgEvtFormatSourceEnabled(cfg, src)) continue;
        if (!(item = createWatchObjectJson(cfg, src))) continue;
        cJSON_AddItemToArray(root, item);
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createEventFormatJson(config_t* cfg)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;
    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
                      valToStr(formatMap, cfgEventFormat(cfg)))) goto err;
    if (!cJSON_AddNumberToObjLN(root, MAXEPS_NODE,
                      cfgEvtRateLimit(cfg))) goto err;
    if (!cJSON_AddStringToObjLN(root, ENHANCEFS_NODE,
                      valToStr(boolMap, cfgEnhanceFs(cfg)))) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createEventJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* format, *watch, *transport;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
                          valToStr(boolMap, cfgEvtEnable(cfg)))) goto err;

    if (!(transport = createTransportJson(cfg, CFG_CTL))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    if (!(format = createEventFormatJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, FORMAT_NODE, format);

    if (!(watch = createWatchArrayJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, WATCH_NODE, watch);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createPayloadJson(config_t *cfg)
{
    cJSON *root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
                         valToStr(boolMap, cfgPayEnable(cfg)))) goto err;
    if (!cJSON_AddStringToObjLN(root, DIR_NODE,
                         cfgPayDir(cfg))) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createLibscopeJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON *log;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!(log = createLogJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, LOG_NODE, log);

    if (!cJSON_AddStringToObjLN(root, CFGEVENT_NODE,
                 valToStr(boolMap, cfgSendProcessStartMsg(cfg)))) goto err;

    if (!cJSON_AddNumberToObjLN(root, SUMMARYPERIOD_NODE,
                                      cfgMtcPeriod(cfg))) goto err;

    if (!cJSON_AddStringToObjLN(root, COMMANDDIR_NODE,
                                         cfgCmdDir(cfg))) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

cJSON*
jsonObjectFromCfg(config_t* cfg)
{
    cJSON* json_root = NULL;
    cJSON* metric, *libscope, *event, *payload;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (!(metric = createMetricJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, METRIC_NODE, metric);

    if (!(libscope = createLibscopeJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, LIBSCOPE_NODE, libscope);

    if (!(event = createEventJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, EVENT_NODE, event);

    if (!(payload = createPayloadJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, PAYLOAD_NODE, payload);

    return json_root;
err:
    if (json_root) cJSON_Delete(json_root);
    return NULL;
}

char*
jsonStringFromCfg(config_t* cfg)
{
    cJSON* json = jsonObjectFromCfg(cfg);
    if (!json) return NULL;

    char* string = cJSON_PrintUnformatted(json);
    cJSON_Delete(json);
    return string;
}

static transport_t*
initTransport(config_t* cfg, which_transport_t t)
{
    transport_t* transport = NULL;

    switch (cfgTransportType(cfg, t)) {
        case CFG_SYSLOG:
            transport = transportCreateSyslog();
            break;
        case CFG_FILE:
            transport = transportCreateFile(cfgTransportPath(cfg, t), cfgTransportBuf(cfg,t));
            break;
        case CFG_UNIX:
            transport = transportCreateUnix(cfgTransportPath(cfg, t));
            break;
        case CFG_UDP:
            transport = transportCreateUdp(cfgTransportHost(cfg, t), cfgTransportPort(cfg, t));
            break;
        case CFG_TCP:
            transport = transportCreateTCP(cfgTransportHost(cfg, t), cfgTransportPort(cfg, t));
            break;
        case CFG_SHM:
            transport = transportCreateShm();
            break;
        default:
            DBG("%d", cfgTransportType(cfg, t));
    }
    return transport;
}

static mtc_fmt_t*
initMtcFormat(config_t* cfg)
{
    mtc_fmt_t* fmt = mtcFormatCreate(cfgMtcFormat(cfg));
    if (!fmt) return NULL;

    mtcFormatStatsDPrefixSet(fmt, cfgMtcStatsDPrefix(cfg));
    mtcFormatStatsDMaxLenSet(fmt, cfgMtcStatsDMaxLen(cfg));
    mtcFormatVerbositySet(fmt, cfgMtcVerbosity(cfg));
    mtcFormatCustomTagsSet(fmt, cfgCustomTags(cfg));
    return fmt;
}

log_t*
initLog(config_t* cfg)
{
    log_t* log = logCreate();
    if (!log) return log;
    transport_t* t = initTransport(cfg, CFG_LOG);
    if (!t) {
        logDestroy(&log);
        return log;
    }
    logTransportSet(log, t);
    logLevelSet(log, cfgLogLevel(cfg));

    return log;
}

mtc_t*
initMtc(config_t* cfg)
{
    mtc_t* mtc = mtcCreate();
    if (!mtc) return mtc;

    mtcEnabledSet(mtc, cfgMtcEnable(cfg));

    transport_t* t = initTransport(cfg, CFG_MTC);
    if (!t) {
        mtcDestroy(&mtc);
        return mtc;
    }
    mtcTransportSet(mtc, t);

    mtc_fmt_t* f = initMtcFormat(cfg);
    if (!f) {
        mtcDestroy(&mtc);
        return mtc;
    }
    mtcFormatSet(mtc, f);

    return mtc;
}

evt_fmt_t *
initEvtFormat(config_t *cfg)
{
    evt_fmt_t *evt = evtFormatCreate();
    if (!evt) return evt;

    watch_t src;
    for (src = CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        evtFormatSourceEnabledSet(evt, src,
                                  cfgEvtEnable(cfg) &&
                                  cfgEvtFormatSourceEnabled(cfg, src));
        evtFormatNameFilterSet(evt, src, cfgEvtFormatNameFilter(cfg, src));
        evtFormatFieldFilterSet(evt, src, cfgEvtFormatFieldFilter(cfg, src));
        evtFormatValueFilterSet(evt, src, cfgEvtFormatValueFilter(cfg, src));
    }

    evtFormatHeaderFilterSet(evt, cfgEvtFormatHeader(cfg));

    evtFormatRateLimitSet(evt, cfgEvtRateLimit(cfg));
    evtFormatCustomTagsSet(evt, cfgCustomTags(cfg));

    return evt;
}

ctl_t *
initCtl(config_t *cfg)
{
    ctl_t *ctl = ctlCreate();
    if (!ctl) return ctl;

    /*
     * If the transport is TCP, the transport may not connect
     * at this point. If so, it will connect later. As such,
     * it should not be treated as fatal.
     */
    transport_t *trans = initTransport(cfg, CFG_CTL);
    if (!trans) {
        ctlDestroy(&ctl);
        return ctl;
    }
    ctlTransportSet(ctl, trans, CFG_CTL);

    if (cfgLogStream(cfg) && cfgPayEnable(cfg)) {
        transport_t *trans = initTransport(cfg, CFG_LS);
        if (!trans) {
            ctlDestroy(&ctl);
            return ctl;
        }

        ctlTransportSet(ctl, trans, CFG_LS);
    } else {
        ctlTransportSet(ctl, NULL, CFG_LS);
    }

    evt_fmt_t* evt = initEvtFormat(cfg);
    if (!evt) {
        ctlDestroy(&ctl);
        return ctl;
    }
    ctlEvtSet(ctl, evt);

    ctlEnhanceFsSet(ctl, cfgEnhanceFs(cfg));
    ctlPayEnableSet(ctl, cfgPayEnable(cfg));
    ctlPayDirSet(ctl,    cfgPayDir(cfg));

    return ctl;
}

/*
 * When connected to LogStream
 * internal configuration, overriding default config, env vars 
 * and the config file to:
 *
 * - use a single IP:port for events, metrics & remote commands
 * - set metrics to use ndjson
 * - use a separate connection over the single IP:port for payloads
 * - include the abbreviated json header for payloads
 * - watch types enabled for files, console, net, fs, http, dns
 * - log level warning
 * - all else uses defaults
 */
int
cfgLogStreamDefault(config_t *cfg)
{
    if (!cfg || (cfgLogStream(cfg) == FALSE)) return -1;

    const char *host = cfgTransportHost(cfg, CFG_LS);
    const char *port = cfgTransportPort(cfg, CFG_LS);

    if (!host || !port) return -1;

    snprintf(g_logmsg, sizeof(g_logmsg), DEFAULT_LOGSTREAM_LOGMSG);

    cfgTransportTypeSet(cfg, CFG_CTL, CFG_TCP);
    cfgTransportHostSet(cfg, CFG_CTL, host);
    cfgTransportPortSet(cfg, CFG_CTL, port);

    cfgTransportTypeSet(cfg, CFG_MTC, CFG_TCP);
    cfgTransportHostSet(cfg, CFG_MTC, host);
    cfgTransportPortSet(cfg, CFG_MTC, port);

    if (cfgMtcEnable(cfg) != TRUE) {
        strncat(g_logmsg, "Metrics enable, ", 20);
    }
    cfgMtcEnableSet(cfg, (unsigned)1);

    if (cfgMtcFormat(cfg) != TRUE) {
        strncat(g_logmsg, "Metrics format, ", 20);
    }
    cfgMtcFormatSet(cfg, CFG_FMT_NDJSON);

    if (cfgEvtEnable(cfg) != TRUE) {
        strncat(g_logmsg, "Event enable, ", 20);
    }
    cfgEvtEnableSet(cfg, (unsigned)1);

    if (cfgLogLevel(cfg) > CFG_LOG_WARN ) {
        strncat(g_logmsg, "Log level, ", 20);
        cfgLogLevelSet(cfg, CFG_LOG_WARN);
    }

    return 0;
}

int
singleChannelSet(ctl_t *ctl, mtc_t *mtc)
{
    if (!ctl || !mtc) return -1;

    // if any logs created during cfg send now
    if (g_logmsg[0] != '\0') {
        scopeLog(g_logmsg, -1, CFG_LOG_WARN);
    }

    transport_t *trans = ctlTransport(ctl, CFG_CTL);
    if (trans) {
        mtcTransportSet(mtc, trans);
        return 0;
    }

    return -1;
}

/*
 * It goes like this:
 * the protocol config file is 3 levels deep
 *
 * the root node is "protocol:"
 * 2nd level is an array of protocols: a mapping node
 * 3rd level is the definition of a specific protocol
 *
 * a specific protocol contains:
 * name: a string, display name of the protocol
 * binary: a string, true or false
 * regex: a string, the regex pattern
 * len: an integer, len is optional, should be supplied for a binary protocol
 */
bool
protocolRead(const char *path, list_t *plist)
{
    FILE *protFile = NULL;
    protocol_def_t *prot = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    int num_found = 0;
    bool name_found = FALSE;
    yaml_parser_t parser;
    yaml_document_t doc;
    yaml_node_t *node;
    yaml_node_pair_t *root_pair, *prot_pair;
    yaml_node_t *root_value, *root_key, *plist_key, *prot_key, *prot_value;
    yaml_node_item_t *pitem;

    // ni for "not-interposed"... a direct glibc call without scope.
    FILE *(*ni_fopen)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
    int (*ni_fclose)(FILE*) = dlsym(RTLD_NEXT, "fclose");

    if (!ni_fopen || !ni_fclose || !path) goto cleanup;

    protFile = ni_fopen(path, "r");
    if (!protFile) goto cleanup;

    parser_successful = yaml_parser_initialize(&parser);
    if (!parser_successful) goto cleanup;

    yaml_parser_set_input_file(&parser, protFile);

    doc_successful = yaml_parser_load(&parser, &doc);
    if (!doc_successful) goto cleanup;

    node = yaml_document_get_root_node(&doc);
    if (node->type != YAML_MAPPING_NODE) goto cleanup;

    /*
     * as defined, no need to loop on the root node
     * in case we need to add more to the config doc...
     * same for the 2nd level
     */
    foreach (root_pair, node->data.mapping.pairs) {
        // 1st level
        root_value = yaml_document_get_node(&doc, root_pair->value);
        if (root_value->type != YAML_SEQUENCE_NODE) goto cleanup;

        root_key = yaml_document_get_node(&doc, root_pair->key);
        if (!root_key || (root_key->type != YAML_SCALAR_NODE)) goto cleanup;
        if (strcmp((char *)root_key->data.scalar.value, "protocol") != 0) goto cleanup;

        foreach(pitem, root_value->data.sequence.items) {
            // 2nd level
            // get an item here instead of a node for the array
            // use the key because there is no value
            plist_key = yaml_document_get_node(&doc, *pitem);
            if (plist_key->type != YAML_MAPPING_NODE) goto cleanup;

            if ((prot = calloc(1, sizeof(protocol_def_t))) == NULL) goto cleanup;
            name_found = FALSE;

            foreach (prot_pair, (yaml_node_pair_t *)plist_key->data.sequence.items) {
                // 3rd level
                prot_key = yaml_document_get_node(&doc, prot_pair->key);
                if (!prot_key || (prot_key->type != YAML_SCALAR_NODE)) goto cleanup;

                prot_value = yaml_document_get_node(&doc, prot_pair->value);
                if (!prot_value) goto cleanup;

                if (!strcmp((char *)prot_key->data.scalar.value, "name")) {
                    if (prot->protname) free(prot->protname);
                    prot->protname = strdup((char *)prot_value->data.scalar.value);
                    name_found = TRUE; // at least need to have a name
                } else if (!strcmp((char *)prot_key->data.scalar.value, "regex")) {
                    if (prot->regex) free(prot->regex);
                    prot->regex = strdup((char *)prot_value->data.scalar.value);
                } else if (!strcmp((char *)prot_key->data.scalar.value, "binary")) {
                    prot->binary = (!strcmp((char *)prot_value->data.scalar.value, "false")) ?
                        FALSE : TRUE; // seems like it should default to true
                } else if (!strcmp((char *)prot_key->data.scalar.value, "len")) {
                    errno = 0;
                    prot->len = strtoull((char *)prot_value->data.scalar.value, NULL, 0);
                    if (errno != 0) prot->len = 0;
                } else {
                    continue;
                }
            }

            if (!name_found  || (lstInsert(plist, num_found, prot) == FALSE)) {
                destroyProtEntry(prot);
            } else {
                num_found++;
            }
            prot = NULL;
        }
    }

cleanup:
    if (prot) destroyProtEntry(prot);
    if (doc_successful) yaml_document_delete(&doc);
    if (parser_successful) yaml_parser_delete(&parser);
    if (protFile) ni_fclose(protFile);
    return TRUE;
}

void
destroyProtEntry(void *data)
{
    if (!data) return;

    protocol_def_t *pre = data;
    if (pre->re) pcre2_code_free(pre->re);
    if (pre->regex) free(pre->regex);
    if (pre->protname) free(pre->protname);
    free(pre);
}
