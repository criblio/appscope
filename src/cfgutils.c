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
#include <fcntl.h>

#include "cfgutils.h"
#include "dbg.h"
#include "mtcformat.h"
#include "scopetypes.h"
#include "com.h"
#include "utils.h"
#include "fn.h"
#include "state.h"
#include "scopestdlib.h"

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
#define WATCH_NODE               "watch"
#define TYPE_NODE                    "type"
#define TRANSPORT_NODE           "transport"
#define TYPE_NODE                    "type"
#define HOST_NODE                    "host"
#define PORT_NODE                    "port"
#define PATH_NODE                    "path"
#define BUFFERING_NODE               "buffering"
#define TLS_NODE                     "tls"
#define ENABLE_NODE                      "enable"
#define VALIDATE_NODE                    "validateserver"
#define CACERT_NODE                      "cacertpath"

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

#define PAYLOAD_NODE         "payload"
#define ENABLE_NODE              "enable"
#define DIR_NODE                 "dir"

#define CRIBL_NODE           "cribl"
#define ENABLE_NODE              "enable"
#define TRANSPORT_NODE           "transport"
#define AUTHTOKEN_NODE           "authtoken"

#define TAGS_NODE            "tags"

#define PROTOCOL_NODE        "protocol"
#define NAME_NODE                "name"
#define REGEX_NODE               "regex"
#define BINARY_NODE              "binary"
#define LEN_NODE                 "len"
#define DETECT_NODE              "detect"
#define PAYLOAD_NODE             "payload"

#define CUSTOM_NODE          "custom"
#define FILTER_NODE              "filter"
#define PROCNAME_NODE                "procname"
#define ARG_NODE                     "arg"
#define HOSTNAME_NODE                "hostname"
#define USERNAME_NODE                "username"
#define ENV_NODE                     "env"
#define ANCESTOR_NODE                "ancestor"
#define CONFIG_NODE              "config"

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
    {"edge",                  CFG_EDGE},
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
void cfgMtcStatsdEnableSetFromStr(config_t*, const char*);
void cfgMtcCategoryEnableSetFromStr(config_t*, const char*, metric_category_t);
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
void cfgTransportTlsEnableSetFromStr(config_t *, which_transport_t, const char *);
void cfgTransportTlsValidateServerSetFromStr(config_t *, which_transport_t, const char *);
void cfgTransportTlsCACertPathSetFromStr(config_t *, which_transport_t, const char *);
void cfgCustomTagAddFromStr(config_t*, const char*, const char*);
void cfgLogLevelSetFromStr(config_t*, const char*);
void cfgPayEnableSetFromStr(config_t*, const char*);
void cfgPayDirSetFromStr(config_t*, const char*);
void cfgAuthTokenSetFromStr(config_t*, const char*);
void cfgEvtFormatHeaderSetFromStr(config_t *, const char *);
void cfgCriblEnableSetFromStr(config_t *, const char *);
static void cfgSetFromFile(config_t *, const char *);

static void processRoot(config_t *, yaml_document_t *, yaml_node_t *);

// These global variables limits us to only reading one config file at a time...
// which seems fine for now, I guess.
static which_transport_t transport_context;
static watch_t watch_context;
static protocol_def_t *protocol_context = NULL;
static regex_t* g_regex = NULL;
static char g_logmsg[1024] = {};

// needed for custom filtering
extern proc_id_t g_proc;

// "state" of custom filtering
static unsigned custom_matched = FALSE;
static unsigned custom_match_count = 0;

static char *
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

    const char *homedir = getenv("HOME");
    const char *scope_home = getenv("SCOPE_HOME");
    if (scope_home &&
       (scope_snprintf(path, sizeof(path), "%s/conf/%s", scope_home, cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if (scope_home &&
       (scope_snprintf(path, sizeof(path), "%s/%s", scope_home, cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if ((scope_snprintf(path, sizeof(path), "/etc/scope/%s", cfgname) > 0 ) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if (homedir &&
       (scope_snprintf(path, sizeof(path), "%s/conf/%s", homedir, cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if (homedir &&
       (scope_snprintf(path, sizeof(path), "%s/%s", homedir, cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if ((scope_snprintf(path, sizeof(path), "./conf/%s", cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }
    if ((scope_snprintf(path, sizeof(path), "./%s", cfgname) > 0) &&
        !scope_access(path, R_OK)) {
        return scope_realpath(path, NULL);
    }

    return NULL;
}

char *
cfgPath(void)
{
    const char* envPath = getenv("SCOPE_CONF_PATH");

    // If SCOPE_CONF_PATH is set, and the file can be opened, use it.
    char *path;
    if (envPath && (path = scope_strdup(envPath))) {

        FILE *fp = scope_fopen(path, "rb");
        if (fp) {
            scope_fclose(fp);
            return path;
        }

        // Couldn't open the file
        scope_free(path);
    }

    // Otherwise, search for scope.yml
    return cfgPathSearch(CFG_FILE_NAME);
}

static void
processCustomTag(config_t* cfg, const char* e, const char* value)
{
    char name_buf[1024];
    scope_strncpy(name_buf, e, sizeof(name_buf));

    char* name = name_buf + (sizeof("SCOPE_TAG_") - 1);

    // convert the "=" to a null delimiter for the name
    char* end = scope_strchr(name, '=');
    if (end) {
        *end = '\0';
        cfgCustomTagAddFromStr(cfg, name, value);
    }

}

static regex_t*
envRegex()
{
    if (g_regex) return g_regex;

    if (!(g_regex = scope_calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        return g_regex;
    }

    if (regcomp(g_regex, "\\$[a-zA-Z0-9_]+", REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        scope_free(g_regex);
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
    scope_free(g_regex);
}

static char*
doEnvVariableSubstitution(const char* value)
{
    if (!value) return NULL;

    regex_t* re = envRegex();
    regmatch_t match = {0};

    int out_size = scope_strlen(value) + 1;
    char* outval = scope_calloc(1, out_size);
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
            outptr = scope_stpncpy(outptr, inptr, match.rm_so - 1);
            // copy the matching env variable name
            outptr = scope_stpncpy(outptr, &inptr[match.rm_so], match_size);
            // move to the next part of the input value
            inptr = &inptr[match.rm_eo];
            continue;
        }

        // lookup the part that matched to see if we can substitute it
        char env_name[match_size + 1];
        scope_strncpy(env_name, &inptr[match.rm_so], match_size);
        env_name[match_size] = '\0';
        char* env_value = getenv(&env_name[1]); // offset of 1 skips the $

        // Grow outval buffer any time env_value is bigger than env_name
        int size_growth = (!env_value) ? 0 : scope_strlen(env_value) - match_size;
        if (size_growth > 0) {
            char* new_outval = scope_realloc(outval, out_size + size_growth);
            if (new_outval) {
                out_size += size_growth;
                outptr = new_outval + (outptr - outval);
                outval = new_outval;
            } else {
                DBG("%s", value);
                scope_free(outval);
                return NULL;
            }
        }

        // copy the part before the match
        outptr = scope_stpncpy(outptr, inptr, match.rm_so);
        // either copy in the env value or the variable that wasn't found
        outptr = scope_stpcpy(outptr, (env_value) ? env_value : env_name);
        // move to the next part of the input value
        inptr = &inptr[match.rm_eo];
    }

    // copy whatever is left
    scope_strcpy(outptr, inptr);

    return outval;
}

static void
processCmdDebug(const char* path)
{
    if (!path || !path[0]) return;

    FILE* f;
    if (!(f = scope_fopen(path, "a"))) return;
    dbgDumpAll(f);
    scope_fclose(f);
}

static void
processReloadConfig(config_t *cfg, const char* value)
{
    if (!cfg || !value) return;
    unsigned int enable = strToVal(boolMap, value);

    if (enable == TRUE) {
        char *path = cfgPath();
        cfgSetFromFile(cfg, path);
        if (path) scope_free(path);
    } else {
        cfgSetFromFile(cfg, value);
    }

    cfgProcessEnvironment(cfg);

    if (cfgLogStreamEnable(cfg)) {
        cfgLogStreamDefault(cfg);
    }
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

    char *env_name = NULL;
    char *value = NULL;
    char *env_ptr;
    env_name = scope_strdup(env_line);
    if (!env_name) goto cleanup;
    if (!(env_ptr = scope_strchr(env_name, '='))) goto cleanup;
    *env_ptr = '\0'; // Delimiting env_name
    if (!(value = doEnvVariableSubstitution(&env_ptr[1]))) goto cleanup;

    if (!scope_strcmp(env_name, "SCOPE_METRIC_ENABLE")) {
        cfgMtcEnableSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_FORMAT")) {
        cfgMtcFormatSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_STATSD_PREFIX")) {
        cfgMtcStatsDPrefixSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_STATSD_MAXLEN")) {
        cfgMtcStatsDMaxLenSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_SUMMARY_PERIOD")) {
        cfgMtcPeriodSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_STATSD")) {
        cfgMtcStatsdEnableSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_FS")) {
        cfgMtcCategoryEnableSetFromStr(cfg, value, CFG_MTC_FS);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_NETWORK")) {
        cfgMtcCategoryEnableSetFromStr(cfg, value, CFG_MTC_NETWORK);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_HTTP")) {
        cfgMtcCategoryEnableSetFromStr(cfg, value, CFG_MTC_HTTP);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_DNS")) {
        cfgMtcCategoryEnableSetFromStr(cfg, value, CFG_MTC_DNS);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_PROC")) {
        cfgMtcCategoryEnableSetFromStr(cfg, value, CFG_MTC_PROC);
    } else if (!scope_strcmp(env_name, "SCOPE_CMD_DIR")) {
        cfgCmdDirSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CONFIG_EVENT")) {
        cfgConfigEventSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_VERBOSITY")) {
        cfgMtcVerbositySetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_LOG_LEVEL")) {
        cfgLogLevelSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_MTC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_TLS_ENABLE")) {
        cfgTransportTlsEnableSetFromStr(cfg, CFG_MTC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_TLS_VALIDATE_SERVER")) {
        cfgTransportTlsValidateServerSetFromStr(cfg, CFG_MTC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_METRIC_TLS_CA_CERT_PATH")) {
        cfgTransportTlsCACertPathSetFromStr(cfg, CFG_MTC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_LOG_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_LOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_LOG_TLS_ENABLE")) {
        cfgTransportTlsEnableSetFromStr(cfg, CFG_LOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_LOG_TLS_VALIDATE_SERVER")) {
        cfgTransportTlsValidateServerSetFromStr(cfg, CFG_LOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_LOG_TLS_CA_CERT_PATH")) {
        cfgTransportTlsCACertPathSetFromStr(cfg, CFG_LOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_PAYLOAD_ENABLE")) {
        cfgPayEnableSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_PAYLOAD_DIR")) {
        cfgPayDirSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CMD_DBG_PATH")) {
        processCmdDebug(value);
    } else if (!scope_strcmp(env_name, "SCOPE_CONF_RELOAD")) {
        processReloadConfig(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_CTL, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_TLS_ENABLE")) {
        cfgTransportTlsEnableSetFromStr(cfg, CFG_CTL, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_TLS_VALIDATE_SERVER")) {
        cfgTransportTlsValidateServerSetFromStr(cfg, CFG_CTL, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_TLS_CA_CERT_PATH")) {
        cfgTransportTlsCACertPathSetFromStr(cfg, CFG_CTL, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_ENABLE")) {
        cfgEvtEnableSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_FORMAT")) {
        cfgEventFormatSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_MAXEPS")) {
        cfgEvtRateLimitSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_ENHANCE_FS")) {
        cfgEnhanceFsSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_LOGFILE_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_CONSOLE_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_SYSLOG_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_METRIC_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_HTTP_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_HTTP_HEADER")) {
        cfgEvtFormatHeaderSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_NET_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_FS_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_DNS_NAME")) {
        cfgEvtFormatNameFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_LOGFILE_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_CONSOLE_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_SYSLOG_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_METRIC_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_HTTP_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_NET_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_FS_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_DNS_FIELD")) {
        cfgEvtFormatFieldFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_LOGFILE_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_CONSOLE_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_SYSLOG_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_METRIC_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_HTTP_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_NET_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_FS_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_DNS_VALUE")) {
        cfgEvtFormatValueFilterSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_LOGFILE")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_CONSOLE")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_SYSLOG")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_METRIC")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_HTTP")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_HTTP, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_NET")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_NET, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_FS")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_FS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_EVENT_DNS")) {
        cfgEvtFormatSourceEnabledSetFromStr(cfg, CFG_SRC_DNS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_ENABLE")) {
        cfgCriblEnableSetFromStr(cfg, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_TLS_ENABLE")) {
        cfgTransportTlsEnableSetFromStr(cfg, CFG_LS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_TLS_VALIDATE_SERVER")) {
        cfgTransportTlsValidateServerSetFromStr(cfg, CFG_LS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_TLS_CA_CERT_PATH")) {
        cfgTransportTlsCACertPathSetFromStr(cfg, CFG_LS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_CLOUD")) {
        cfgLogStreamCloudSet(cfg, TRUE);
        cfgTransportSetFromStr(cfg, CFG_LS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL")) {
        cfgLogStreamCloudSet(cfg, FALSE);
        cfgTransportSetFromStr(cfg, CFG_LS, value);
    } else if (!scope_strcmp(env_name, "SCOPE_CRIBL_AUTHTOKEN")) {
        cfgAuthTokenSetFromStr(cfg, value);
    } else if (startsWith(env_name, "SCOPE_TAG_")) {
        processCustomTag(cfg, env_line, value);
    }

cleanup:
    if (value) scope_free(value);
    if (env_name) scope_free(env_name);
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

    char *line = NULL;
    size_t len = 0;

    while (scope_getline(&line, &len, file) != -1) {
        line[scope_strcspn(line, "\r\n")] = '\0'; //overwrite first \r or \n with null
        processEnvStyleInput(cfg, line);
        line[0] = '\0';
    }

    if (line) scope_free(line);
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
    scope_errno = 0;
    char* endptr = NULL;
    unsigned long x = scope_strtoul(value, &endptr, 10);
    if (scope_errno || *endptr) return;

    cfgMtcStatsDMaxLenSet(cfg, x);
}

void
cfgMtcPeriodSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    scope_errno = 0;
    char* endptr = NULL;
    unsigned long x = scope_strtoul(value, &endptr, 10);
    if (scope_errno || *endptr) return;

    cfgMtcPeriodSet(cfg, x);
}

void
cfgMtcStatsdEnableSetFromStr(config_t *cfg, const char *value)
{
    if (!cfg || !value) return;
    cfgMtcStatsdEnableSet(cfg, strToVal(boolMap, value));
}

void
cfgMtcCategoryEnableSetFromStr(config_t *cfg, const char *value, metric_category_t type)
{
    if (!cfg || !value) return;
    cfgMtcCategoryEnableSet(cfg, strToVal(boolMap, value), type);
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
    scope_errno = 0;
    char* endptr = NULL;
    unsigned long x = scope_strtoul(value, &endptr, 10);
    if (scope_errno || *endptr) return;

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
    scope_errno = 0;
    char* endptr = NULL;
    unsigned long x = scope_strtoul(value, &endptr, 10);
    if (scope_errno || *endptr) return;

    cfgMtcVerbositySet(cfg, x);
}

void
cfgTransportSetFromStr(config_t *cfg, which_transport_t t, const char *value)
{
    if (!cfg || !value) return;

    // see if value starts with udp://, tcp://, file://, unix:// or equals edge
    if (value == scope_strstr(value, "udp://")) {

        // copied to avoid directly modifying the process's env variable
        char value_cpy[1024];
        scope_strncpy(value_cpy, value, sizeof(value_cpy));

        char *host = value_cpy + (sizeof("udp://") - 1);

        // convert the ':' to a null delimiter for the host
        // and move port past the null
        char *port = scope_strrchr(host, ':');
        if (!port) return;  // port is *required*
        *port = '\0';
        port++;

        cfgTransportTypeSet(cfg, t, CFG_UDP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);

    } else if (value == scope_strstr(value, "tcp://")) {

        // copied to avoid directly modifying the process's env variable
        char value_cpy[1024];
        scope_strncpy(value_cpy, value, sizeof(value_cpy));

        char *host = value_cpy + (sizeof("tcp://") - 1);

        // convert the ':' to a null delimiter for the host
        // and move port past the null
        char *port = scope_strrchr(host, ':');
        if (!port) return;  // port is *required*
        *port = '\0';
        port++;

        cfgTransportTypeSet(cfg, t, CFG_TCP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);

    } else if (value == scope_strstr(value, "file://")) {
        const char *path = value + (sizeof("file://") - 1);
        cfgTransportTypeSet(cfg, t, CFG_FILE);
        cfgTransportPathSet(cfg, t, path);
    } else if (value == scope_strstr(value, "unix://")) {
        const char *path = value + (sizeof("unix://") - 1);
        cfgTransportTypeSet(cfg, t, CFG_UNIX);
        cfgTransportPathSet(cfg, t, path);
    } else if (scope_strncmp(value, "edge", sizeof("edge") - 1) == 0) {
        cfgTransportTypeSet(cfg, t, CFG_EDGE);
    }
}

void
cfgTransportTlsEnableSetFromStr(config_t *cfg, which_transport_t t, const char *value)
{
    if (!cfg || !value) return;
    cfgTransportTlsEnableSet(cfg, t, strToVal(boolMap, value));
}

void
cfgTransportTlsValidateServerSetFromStr(config_t *cfg, which_transport_t t, const char *value)
{
    if (!cfg || !value) return;
    cfgTransportTlsValidateServerSet(cfg, t, strToVal(boolMap, value));
}
void
cfgTransportTlsCACertPathSetFromStr(config_t *cfg, which_transport_t t, const char *value)
{
    // A little silly to define passthrough function
    // but this keeps the interface consistent.
    if (!cfg || !value) return;
    cfgTransportTlsCACertPathSet(cfg, t, value);
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

void
cfgCriblEnableSetFromStr(config_t *cfg, const char *value)
{
    if (!cfg || !value) return;
    cfgLogStreamEnableSet(cfg, strToVal(boolMap, value));
}

void
cfgAuthTokenSetFromStr(config_t *cfg, const char *value)
{
    if (!cfg || !value) return;
    cfgAuthTokenSet(cfg, value);
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
            (!scope_strcmp((char*)key->data.scalar.value, t[i].key))) {
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
    if (value) scope_free(value);
}

static void
processTransportType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportTypeSet(config, c, strToVal(transportTypeMap, value));
    if (value) scope_free(value);
}

static void
processHost(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportHostSet(config, c, value);
    if (value) scope_free(value);
}

static void
processPort(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportPortSet(config, c, value);
    if (value) scope_free(value);
}

static void
processPath(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportPathSet(config, c, value);
    if (value) scope_free(value);
}

static void
processBuf(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportBufSet(config, c, strToVal(bufferMap, value));
    if (value) scope_free(value);
}

static void
processTlsEnable(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportTlsEnableSetFromStr(config, c, value);
    if (value) scope_free(value);
}

static void
processTlsValidate(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportTlsValidateServerSetFromStr(config, c, value);
    if (value) scope_free(value);
}

static void
processTlsCaCert(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    which_transport_t c = transport_context;
    cfgTransportTlsCACertPathSetFromStr(config, c, value);
    if (value) scope_free(value);
}

static void
processTls(config_t *config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processTlsEnable},
        {YAML_SCALAR_NODE,    VALIDATE_NODE,        processTlsValidate},
        {YAML_SCALAR_NODE,    CACERT_NODE,          processTlsCaCert},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t *pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
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
        {YAML_MAPPING_NODE,   TLS_NODE,             processTls},
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
        if (key_str) scope_free(key_str);
        if (value_str) scope_free(value_str);
    }
}

static void
processFormatTypeMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcFormatSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processFormatTypeEvent(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEventFormatSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processFormatMaxEps(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEvtRateLimitSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processEnhanceFs(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgEnhanceFsSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processStatsDPrefix(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcStatsDPrefixSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processStatsDMaxLen(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcStatsDMaxLenSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processVerbosity(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcVerbositySetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processMetricEnable(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcEnableSetFromStr(config, value);
    if (value) scope_free(value);
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
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processMtcWatchType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    if (!scope_strcmp(value, "statsd")) {
        cfgMtcStatsdEnableSet(config, TRUE);
    }
    if (value) scope_free(value);
}

static void
processMtcSource(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processMtcWatchType},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}
static void
processMtcWatch(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    // Type can be scalar or sequence.
    // It will be scalar if there are zero entries, in which case we
    // clear all values and return.
    if ((node->type != YAML_SEQUENCE_NODE) &&
       (node->type != YAML_SCALAR_NODE)) return;

    // absence of one of these values means to clear it.
    // clear them all, then set values for whatever we find.
    cfgMtcStatsdEnableSet(config, FALSE);

    if (node->type != YAML_SEQUENCE_NODE) return;
    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* i = yaml_document_get_node(doc, *item);
        processMtcSource(config, doc, i);
    }
}

static void
processSummaryPeriod(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgMtcPeriodSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processCommandDir(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgCmdDirSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processConfigEvent(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgConfigEventSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processMetricEnable},
        {YAML_MAPPING_NODE,   FORMAT_NODE,          processFormat},
        {YAML_SEQUENCE_NODE,  WATCH_NODE,           processMtcWatch},
        {YAML_SCALAR_NODE,    WATCH_NODE,           processMtcWatch},
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
    if (value) scope_free(value);
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
processEvtWatchType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    watch_context = strToVal(watchTypeMap, value);
    cfgEvtFormatSourceEnabledSet(config, watch_context, 1);
    if (value) scope_free(value);
}

static void
processEvtWatchName(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatNameFilterSetFromStr(config, watch_context, value);
    if (value) scope_free(value);
}

static void
processEvtWatchField(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatFieldFilterSetFromStr(config, watch_context, value);
    if (value) scope_free(value);
}

static void
processEvtWatchValue(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEvtFormatValueFilterSetFromStr(config, watch_context, value);
    if (value) scope_free(value);
}

static void
processEvtWatchHeader(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    if (node->type != YAML_SEQUENCE_NODE) return;

    // watch header is only valid for http
    if (watch_context != CFG_SRC_HTTP) return;

    yaml_node_item_t *item;

    foreach(item, node->data.sequence.items) {
        yaml_node_t *node = yaml_document_get_node(doc, *item);
        char *value = stringVal(node);
        cfgEvtFormatHeaderSet(config, value);
        if (value) scope_free(value);
    }
}

static int
isWatchType(yaml_document_t* doc, yaml_node_pair_t* pair)
{
    yaml_node_t* key = yaml_document_get_node(doc, pair->key);
    if (!key || (key->type != YAML_SCALAR_NODE)) return 0;
    return !scope_strcmp((char*)key->data.scalar.value, TYPE_NODE);
}

static void
processSource(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processEvtWatchType},
        {YAML_SCALAR_NODE,    NAME_NODE,            processEvtWatchName},
        {YAML_SCALAR_NODE,    FIELD_NODE,           processEvtWatchField},
        {YAML_SCALAR_NODE,    VALUE_NODE,           processEvtWatchValue},
        {YAML_SEQUENCE_NODE,  EX_HEADERS,           processEvtWatchHeader},
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
processEvtWatch(config_t* config, yaml_document_t* doc, yaml_node_t* node)
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
        {YAML_SEQUENCE_NODE,  WATCH_NODE,           processEvtWatch},
        {YAML_SCALAR_NODE,    WATCH_NODE,           processEvtWatch},
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
    if (value) scope_free(value);
}

static void
processPayloadDir(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    cfgPayDirSetFromStr(config, value);
    if (value) scope_free(value);
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
processCriblEnable(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char *value = stringVal(node);
    cfgCriblEnableSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processCriblTransport(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    transport_context = CFG_LS;
    processTransport(config, doc, node);
}

static void
processAuthToken(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    char* value = stringVal(node);
    cfgAuthTokenSetFromStr(config, value);
    if (value) scope_free(value);
}

static void
processCribl(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    ENABLE_NODE,          processCriblEnable},
        {YAML_MAPPING_NODE,   TRANSPORT_NODE,       processCriblTransport},
        {YAML_SCALAR_NODE,    AUTHTOKEN_NODE,       processAuthToken},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t *pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processProtocolName(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    if (protocol_context->protname) scope_free(protocol_context->protname);
    protocol_context->protname = stringVal(node);
}

static void
processProtocolRegex(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    if (protocol_context->regex) scope_free(protocol_context->regex);
    protocol_context->regex = stringVal(node);
}

static void
processProtocolBinary(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    char* sVal = stringVal(node);
    unsigned iVal = strToVal(boolMap, sVal);
    if (iVal <= 1) protocol_context->binary = iVal;
    if (sVal) scope_free(sVal);
}

static void
processProtocolLen(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    char *sVal = stringVal(node);
    char *endInt = NULL;
    scope_errno = 0;
    unsigned long iVal = scope_strtoul(sVal, &endInt, 10);
    if (!scope_errno && !*endInt) protocol_context->len = iVal;
    if (sVal) scope_free(sVal);
}

static void
processProtocolDetect(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    char* sVal = stringVal(node);
    unsigned iVal = strToVal(boolMap, sVal);
    if (iVal <= 1) protocol_context->detect = iVal;
    if (sVal) scope_free(sVal);
}

static void
processProtocolPayload(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE || !protocol_context) return;
    char* sVal = stringVal(node);
    unsigned iVal = strToVal(boolMap, sVal);
    if (iVal <= 1) protocol_context->payload = iVal;
    if (sVal) scope_free(sVal);
}

static void
processProtocolEntry(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    // protocol entries must be key/value maps
    if (node->type != YAML_MAPPING_NODE) {
        scopeLogWarn("WARN: ignoring non-map protocol entry\n");
        return;
    }

    // the protocol list should already be setup
    if (!g_protlist) {
        DBG(NULL);
        return;
    }

    // protocol object to populate
    protocol_context = scope_calloc(1, sizeof(protocol_def_t));
    if (!protocol_context) {
        DBG(NULL);
        return;
    }
    protocol_context->detect = TRUE; // non-zero default

    // process the entry
    parse_table_t t[] = {
        {YAML_SCALAR_NODE, NAME_NODE,    processProtocolName},
        {YAML_SCALAR_NODE, REGEX_NODE,   processProtocolRegex},
        {YAML_SCALAR_NODE, BINARY_NODE,  processProtocolBinary},
        {YAML_SCALAR_NODE, LEN_NODE,     processProtocolLen},
        {YAML_SCALAR_NODE, DETECT_NODE,  processProtocolDetect},
        {YAML_SCALAR_NODE, PAYLOAD_NODE, processProtocolPayload},
        {YAML_NO_NODE,     NULL,         NULL}
    };
    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }

    // require at least the name and regex
    if (!protocol_context->protname || !protocol_context->regex) {
        destroyProtEntry(protocol_context);
        protocol_context = NULL;
        scopeLogWarn("WARN: ignoring protocol entry missing name or regex\n");
        return;
    }

    // init the regex
    int errornumber;
    PCRE2_SIZE erroroffset;
    protocol_context->re = pcre2_compile(
            (PCRE2_SPTR)protocol_context->regex,
            PCRE2_ZERO_TERMINATED, 0,
            &errornumber, &erroroffset, NULL);
    if (!protocol_context->re) {
        scopeLogWarn("WARN: invalid regex for \"%s\" protocol entry; %s\n",
                 protocol_context->protname, protocol_context->regex);
        destroyProtEntry(protocol_context);
        protocol_context = NULL;
        return;
    }

    // replace if name matches existing entry
    for (list_key_t key = 0; key <= g_prot_sequence; ++key) {
        protocol_def_t *found = lstFind(g_protlist, key);
        if (found && !scope_strcmp(protocol_context->protname, found->protname)) {
            protocol_context->type = key;
            if (!lstDelete(g_protlist, key)) {
                DBG(NULL);
            }
            if (!lstInsert(g_protlist, key, protocol_context)) {
                DBG(NULL);
            }
            protocol_context = NULL;
            destroyProtEntry(found);
            break;
        }
    }

    // otherwise, add
    if (protocol_context) {
        protocol_context->type = ++g_prot_sequence;
        if (!lstInsert(g_protlist, g_prot_sequence, protocol_context)) {
            --g_prot_sequence;
            destroyProtEntry(protocol_context);
            DBG(NULL);
        }
        protocol_context = NULL;
    }
}

static void
processProtocol(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    if (node->type != YAML_SEQUENCE_NODE) return;

    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* node = yaml_document_get_node(doc, *item);
        processProtocolEntry(config, doc, node);
    }
}

static void
processCustomFilterProcname(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar procname value\n");
        custom_matched = FALSE;
        return;
    }

    char *valueStr = stringVal(node);
    if (valueStr && !scope_strcmp(valueStr, g_proc.procname)) {
        ++custom_match_count;
        scope_free(valueStr);
        return;
    }


    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilterArg(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar arg value\n");
        custom_matched = FALSE;
        return;
    }

    char *valueStr = stringVal(node);
    if (valueStr && scope_strstr(g_proc.cmd, valueStr)) {
        ++custom_match_count;
        scope_free(valueStr);
        return;
    }

    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilterHostname(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar hostname value\n");
        custom_matched = FALSE;
        return;
    }

    // Note hostname are not case sensitive so unlike other filters, this is a
    // case-insensitive comparison.
    char *valueStr = stringVal(node);
    if (valueStr && !scope_strcasecmp(valueStr, g_proc.hostname)) {
        ++custom_match_count;
        scope_free(valueStr);
        return;
    }

    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilterUsername(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar username value\n");
        custom_matched = FALSE;
        return;
    }

    char *valueStr = stringVal(node);
    struct passwd *pw = getpwuid(g_proc.uid);
    if (valueStr && pw && !scope_strcmp(valueStr, pw->pw_name)) {
        ++custom_match_count;
        scope_free(valueStr);
        return;
    }

    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilterEnv(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar env value\n");
        custom_matched = FALSE;
        return;
    }

    char *valueStr = stringVal(node);
    if (valueStr) {
        char *equal = scope_strchr(valueStr, '=');
        if (equal) *equal = '\0';
        char *envName = valueStr;
        char *envVal = equal ? equal+1 : NULL;
        char *env = getenv(envName);
        if (env) {
            if (envVal) {
                if (!scope_strcmp(env, envVal)) {
                    ++custom_match_count;
                    scope_free(valueStr);
                    return;
                }
            } else {
                ++custom_match_count;
                scope_free(valueStr);
                return;
            }
        }
    }

    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilterAncestor(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) {
        scopeLogWarn("WARN: non-scalar ancestor value\n");
        custom_matched = FALSE;
        return;
    }

    char *valueStr = stringVal(node);
    if (valueStr) {
        pid_t ppid = g_proc.ppid;
        while (ppid > 1) {
            char buf[PATH_MAX];
            if (scope_snprintf(buf, sizeof(buf), "/proc/%d/exe", ppid) < 0) {
                DBG(NULL);
                break;
            }

            char exe[PATH_MAX];
            size_t exeLen = scope_readlink(buf, exe, sizeof(exe));
            if (exeLen <= 0) {
                DBG(NULL);
                break;
            }
            exe[exeLen] = '\0';

            char* name = exe;
            name = scope_basename(exe);
            if (!scope_strcmp(valueStr, name)) {
                ++custom_match_count;
                scope_free(valueStr);
                return;
            }

            if (scope_snprintf(buf, sizeof(buf), "/proc/%d/stat", ppid) < 0) {
                DBG(NULL);
                break;
            }
            int fd = scope_open(buf, O_RDONLY);
            if (fd == -1) {
                DBG(NULL);
                break;
            }
            if (scope_read(fd, buf, sizeof(buf)) <= 0) { 
                DBG(NULL);
                scope_close(fd);
                break;
            }
            scope_strtok(buf,  " ");              // (1) pid   %d
            scope_strtok(NULL, " ");              // (2) comm  %s
            scope_strtok(NULL, " ");              // (3) state %s
            ppid = scope_atoi(scope_strtok(NULL, " ")); // (4) ppid  %d
            scope_close(fd);
        }
    }

    custom_matched = FALSE;
    if (valueStr) scope_free(valueStr);
}

static void
processCustomFilter(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    custom_matched = TRUE;
    custom_match_count = 0;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE, PROCNAME_NODE, processCustomFilterProcname},
        {YAML_SCALAR_NODE, ARG_NODE,      processCustomFilterArg},
        {YAML_SCALAR_NODE, HOSTNAME_NODE, processCustomFilterHostname},
        {YAML_SCALAR_NODE, USERNAME_NODE, processCustomFilterUsername},
        {YAML_SCALAR_NODE, ENV_NODE,      processCustomFilterEnv},
        {YAML_SCALAR_NODE, ANCESTOR_NODE, processCustomFilterAncestor},
        {YAML_NO_NODE,     NULL,          NULL}
    };

    yaml_node_pair_t *pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
        if (!custom_matched) break;
    }
}

static void
processCustomConfig(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    // All filters have to match and there must be more than one filter
    if (!custom_matched || !custom_match_count) {
        scopeLogInfo("INFO: skipping custom config\n");
        return;
    }

    processRoot(config, doc, node);
}

static void
processCustomEntry(config_t* config, yaml_document_t* doc, yaml_node_pair_t* pair)
{
    //yaml_node_t* name = yaml_document_get_node(doc, pair->key);
    yaml_node_t* node = yaml_document_get_node(doc, pair->value);

    if (node->type != YAML_MAPPING_NODE) {
        scopeLogWarn("WARN: ignoring non-map custom entry\n");
        return;
    }

    parse_table_t t[] = {
        {YAML_MAPPING_NODE, FILTER_NODE, processCustomFilter},
        {YAML_NO_NODE,      NULL,        NULL}
    };
    yaml_node_pair_t* nodePair;
    foreach(nodePair, node->data.mapping.pairs) {
        processKeyValuePair(t, nodePair, config, doc);
    }

    parse_table_t t2[] = {
        {YAML_MAPPING_NODE, CONFIG_NODE, processCustomConfig},
        {YAML_NO_NODE,      NULL,        NULL}
    };
    foreach(nodePair, node->data.mapping.pairs) {
        processKeyValuePair(t2, nodePair, config, doc);
    }
}

static void
processCustom(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processCustomEntry(config, doc, pair);
    }
}

static void
processRoot(config_t *config, yaml_document_t *doc, yaml_node_t *node)
{
    parse_table_t t[] = {
        {YAML_MAPPING_NODE,  METRIC_NODE,   processMetric},
        {YAML_MAPPING_NODE,  LIBSCOPE_NODE, processLibscope},
        {YAML_MAPPING_NODE,  PAYLOAD_NODE,  processPayload},
        {YAML_MAPPING_NODE,  EVENT_NODE,    processEvent},
        {YAML_MAPPING_NODE,  CRIBL_NODE,    processCribl},
        {YAML_MAPPING_NODE,  TAGS_NODE,     processTags},
        {YAML_SEQUENCE_NODE, PROTOCOL_NODE, processProtocol},
        {YAML_NO_NODE,       NULL,          NULL}
    };

    yaml_node_pair_t *pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
setConfigFromDoc(config_t* config, yaml_document_t* doc)
{
    yaml_node_t *node = yaml_document_get_root_node(doc);
    if (node->type != YAML_MAPPING_NODE) return;

    processRoot(config, doc, node);

    // process custom entries after the others
    parse_table_t t2[] = {
        {YAML_MAPPING_NODE, CUSTOM_NODE, processCustom},
        {YAML_NO_NODE,      NULL,        NULL}
    };

    yaml_node_pair_t *pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t2, pair, config, doc);
    }
}

static void
cfgSetFromFile(config_t *config, const char* path)
{
    FILE *fp = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    yaml_parser_t parser;
    yaml_document_t doc;

    if (!config) goto cleanup;
    if (!path) goto cleanup;
    fp = scope_fopen(path, "rb");
    if (!fp) goto cleanup;

    parser_successful = yaml_parser_initialize(&parser);
    if (!parser_successful) goto cleanup;

    yaml_parser_set_input_file(&parser, fp);

    doc_successful = yaml_parser_load(&parser, &doc);
    if (!doc_successful) goto cleanup;

    // This is where the magic happens
    setConfigFromDoc(config, &doc);

cleanup:
    if (doc_successful) yaml_document_delete(&doc);
    if (parser_successful) yaml_parser_delete(&parser);
    if (fp) scope_fclose(fp);
}

config_t *
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

    yaml_parser_set_input_string(&parser, (unsigned char*)string, scope_strlen(string));

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

static cJSON *
createTlsJson(config_t *cfg, which_transport_t trans)
{
    cJSON* root = NULL;
    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
         valToStr(boolMap, cfgTransportTlsEnable(cfg, trans)))) goto err;
    if (!cJSON_AddStringToObjLN(root, VALIDATE_NODE,
         valToStr(boolMap, cfgTransportTlsValidateServer(cfg, trans)))) goto err;

    // Represent NULL as an empty string
    const char *path = cfgTransportTlsCACertPath(cfg, trans);
    path = (path) ? path : "";
    if (!cJSON_AddStringToObjLN(root, CACERT_NODE, path)) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createTransportJson(config_t* cfg, which_transport_t trans)
{
    cJSON* root = NULL;
    cJSON* tls = NULL;

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

            if (!(tls = createTlsJson(cfg, trans))) goto err;
            cJSON_AddItemToObjectCS(root, TLS_NODE, tls);
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
        case CFG_EDGE:
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

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
                     valToStr(formatMap, cfgMtcFormat(cfg)))) goto err;
    if (!cJSON_AddStringToObjLN(root, STATSDPREFIX_NODE,
                                    cfgMtcStatsDPrefix(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, STATSDMAXLEN_NODE,
                                    cfgMtcStatsDMaxLen(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, VERBOSITY_NODE,
                                       cfgMtcVerbosity(cfg))) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createMetricWatchObjectJson(config_t *cfg)
{
    cJSON *root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE, "statsd")) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createMetricWatchArrayJson(config_t* cfg)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateArray())) goto err;

    if (cfgMtcStatsdEnable(cfg)) {
        cJSON* item;
        if (!(item = createMetricWatchObjectJson(cfg))) goto err;
        cJSON_AddItemToArray(root, item);
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}


static cJSON*
createMetricJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON *transport, *format, *watch;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
                          valToStr(boolMap, cfgMtcEnable(cfg)))) goto err;

    if (!(transport = createTransportJson(cfg, CFG_MTC))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    if (!(format = createMetricFormatJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, FORMAT_NODE, format);

    if (!(watch = createMetricWatchArrayJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, WATCH_NODE, watch);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createWatchObjectJson(config_t *cfg, watch_t src)
{
    cJSON *root = NULL;

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
        cJSON *headers = cJSON_CreateArray();
        if (!headers) goto err;

        int numhead;
        if ((numhead = cfgEvtFormatNumHeaders(cfg)) > 0) {
            int i;

            for (i = 0; i < numhead; i++) {
                char *hstr = (char *)cfgEvtFormatHeader(cfg, i);
                if (hstr) {
                    cJSON_AddStringToObjLN(headers, "headers", hstr);
                }
            }
        }

        cJSON_AddItemToObject(root, "headers", headers);
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

static cJSON*
createProtocolEntryJson(config_t* cfg, protocol_def_t* prot)
{
    cJSON *root = NULL;

    if (!prot) goto err;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, NAME_NODE, prot->protname)) goto err;
    if (!cJSON_AddStringToObjLN(root, REGEX_NODE, prot->regex)) goto err;
    if (!cJSON_AddStringToObjLN(root, BINARY_NODE, valToStr(boolMap, prot->binary))) goto err;
    if (!cJSON_AddNumberToObjLN(root, LEN_NODE, prot->len)) goto err;
    if (!cJSON_AddStringToObjLN(root, DETECT_NODE, valToStr(boolMap, prot->detect))) goto err;
    if (!cJSON_AddStringToObjLN(root, PAYLOAD_NODE, valToStr(boolMap, prot->payload))) goto err;

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createProtocolJson(config_t* cfg)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateArray())) goto err;

    for (unsigned key = 1; key <= g_prot_sequence; ++key) {
        protocol_def_t *prot = lstFind(g_protlist, key);
        if (prot) {
            cJSON *item = createProtocolEntryJson(cfg, prot);
            if (!item) goto err;
            cJSON_AddItemToArray(root, item);
        }
    }

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createCriblJson(config_t* cfg)
{
    cJSON *root = NULL;
    cJSON *transport;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, ENABLE_NODE,
                          valToStr(boolMap, cfgLogStreamEnable(cfg)))) goto err;

    if (!(transport = createTransportJson(cfg, CFG_LS))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    // Represent NULL as an empty string
    const char *token = cfgAuthToken(cfg);
    token = (token) ? token : "";
    if (!cJSON_AddStringToObjLN(root, AUTHTOKEN_NODE, token)) goto err;

    return root;
err:
    if(root) cJSON_Delete(root);
    return NULL;
}

cJSON*
jsonObjectFromCfg(config_t* cfg)
{
    cJSON *json_root = NULL;
    cJSON *metric, *libscope, *event, *payload, *tags, *protocol, *cribl;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (!(metric = createMetricJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, METRIC_NODE, metric);

    if (!(libscope = createLibscopeJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, LIBSCOPE_NODE, libscope);

    if (!(event = createEventJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, EVENT_NODE, event);

    if (!(payload = createPayloadJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, PAYLOAD_NODE, payload);

    if (!(tags = createTagsJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, TAGS_NODE, tags);

    if (!(protocol = createProtocolJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, PROTOCOL_NODE, protocol);

    if (!(cribl = createCriblJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, CRIBL_NODE, cribl);

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
        case CFG_EDGE:
        {
            transport = transportCreateEdge();
            break;
        }
        case CFG_UDP:
            transport = transportCreateUdp(cfgTransportHost(cfg, t), cfgTransportPort(cfg, t));
            break;
        case CFG_TCP:
            transport = transportCreateTCP(cfgTransportHost(cfg, t), cfgTransportPort(cfg, t),
                                           cfgTransportTlsEnable(cfg, t),
                                           cfgTransportTlsValidateServer(cfg, t),
                                           cfgTransportTlsCACertPath(cfg, t));
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

    evtFormatRateLimitSet(evt, cfgEvtRateLimit(cfg));
    evtFormatCustomTagsSet(evt, cfgCustomTags(cfg));

    return evt;
}

static bool
protocolDefinitionsUsePayloads()
{
    bool retVal = FALSE;
    protocol_def_t *protoDef = NULL;
    unsigned int ptype = 0;

    // Loop through all payload definitions.
    // If any has payload set, return TRUE
    for (ptype = 0; ptype <= g_prot_sequence; ptype++) {
        if ((protoDef = lstFind(g_protlist, ptype)) != NULL) {
            retVal |= protoDef->payload;
            if (retVal) break;
        }
    }
    return retVal;
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

    // We'll create a dedicated payload channel, conditionally.
    if (cfgLogStreamEnable(cfg)
            && (cfgPayEnable(cfg) || protocolDefinitionsUsePayloads())) {
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
 * - use a single IP:port/UNIX socket for events, metrics & remote commands
 * - set metrics to use ndjson
 * - use a separate connection over the single IP:port/UNIX socket for payloads
 * - include the abbreviated json header for payloads
 * - watch types enabled for files, console, net, fs, http, dns
 * - log level warning
 * - all else uses defaults
 */
int
cfgLogStreamDefault(config_t *cfg)
{
    if (!cfg || (cfgLogStreamEnable(cfg) == FALSE)) return -1;

    scope_snprintf(g_logmsg, sizeof(g_logmsg), DEFAULT_LOGSTREAM_LOGMSG);

    cfg_transport_t ls_type = cfgTransportType(cfg, CFG_LS);

    if (ls_type == CFG_UNIX) {
        const char *path = cfgTransportPath(cfg, CFG_LS);
        cfgTransportPathSet(cfg, CFG_CTL, path);
    } else if (ls_type == CFG_TCP) {
        // override the CFG_LS transport type to be TCP for type different than UNIX
        cfgTransportTypeSet(cfg, CFG_LS, CFG_TCP);
        // host is already set
        // port is already set

        // if cloud, override tls settings too
        if (cfgLogStreamCloud(cfg) == TRUE) {
            // TLS enabled, with Server Validation, using root certs (payload)
            cfgTransportTlsEnableSet(cfg, CFG_LS, TRUE);
            cfgTransportTlsValidateServerSet(cfg, CFG_LS, TRUE);
            cfgTransportTlsCACertPathSet(cfg, CFG_LS, NULL);
        }
        const char *host = cfgTransportHost(cfg, CFG_LS);
        cfgTransportHostSet(cfg, CFG_CTL, host);
        const char *port = cfgTransportPort(cfg, CFG_LS);
        cfgTransportPortSet(cfg, CFG_CTL, port);
        unsigned int enable = cfgTransportTlsEnable(cfg, CFG_LS);
        cfgTransportTlsEnableSet(cfg, CFG_CTL, enable);
        unsigned int validateserver = cfgTransportTlsValidateServer(cfg, CFG_LS);
        cfgTransportTlsValidateServerSet(cfg, CFG_CTL, validateserver);
        const char *cacertpath = cfgTransportTlsCACertPath(cfg, CFG_LS);
        cfgTransportTlsCACertPathSet(cfg, CFG_CTL, cacertpath);
    }

    cfg_transport_t type = cfgTransportType(cfg, CFG_LS);
    cfgTransportTypeSet(cfg, CFG_CTL, type);

    if (cfgMtcFormat(cfg) != TRUE) {
        scope_strncat(g_logmsg, "Metrics format, ", 20);
    }
    cfgMtcFormatSet(cfg, CFG_FMT_NDJSON);

    if (cfgLogLevel(cfg) > CFG_LOG_WARN ) {
        scope_strncat(g_logmsg, "Log level, ", 20);
        cfgLogLevelSet(cfg, CFG_LOG_WARN);
    }

    if (!cfgSendProcessStartMsg(cfg)) {
        scope_strncat(g_logmsg, "Send proc start msg, ", 25);
        cfgSendProcessStartMsgSet(cfg, TRUE);
    }

    return 0;
}

int
singleChannelSet(ctl_t *ctl, mtc_t *mtc)
{
    if (!ctl || !mtc) return -1;

    // if any logs created during cfg send now
    if (g_logmsg[0] != '\0') {
        scopeLogWarn("%s", g_logmsg);
    }

    transport_t *trans = ctlTransport(ctl, CFG_CTL);
    if (trans) {
        mtcTransportSet(mtc, trans);
        return 0;
    }

    return -1;
}

void
destroyProtEntry(void *data)
{
    if (!data) return;

    protocol_def_t *pre = data;
    if (pre->re) pcre2_code_free(pre->re);
    if (pre->regex) scope_free(pre->regex);
    if (pre->protname) scope_free(pre->protname);
    scope_free(pre);
}
