#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pwd.h>
#include <regex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cfgutils.h"
#include "dbg.h"
#include "format.h"
#include "scopetypes.h"

#ifndef NO_YAML
#include "yaml.h"
#endif

#define METRIC_NODE          "metric"
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
#define TRANSPORT_NODE           "transport"
#define LOG_NODE                 "log"
#define LEVEL_NODE                   "level"
#define TRANSPORT_NODE               "transport"
#define SUMMARYPERIOD_NODE       "summaryperiod"
#define COMMANDDIR_NODE          "commanddir"

#define EVENT_NODE           "event"
#define FORMAT_NODE              "format"
#define TYPE_NODE                    "type"
#define WATCH_NODE               "watch"
#define TYPE_NODE                    "type"
#define NAME_NODE                    "name"
#define FIELD_NODE                   "field"
#define VALUE_NODE                   "value"

typedef struct {
    const char* str;
    unsigned val;
} enum_map_t;

unsigned strToVal(enum_map_t map[], const char* str)
{
    enum_map_t* m;
    for (m=map; m->str; m++) {
        if (!strcmp(str, m->str)) return m->val;
    }
    return -1;
}

const char* valToStr(enum_map_t map[], unsigned val)
{
    enum_map_t* m;
    for (m=map; m->str; m++) {
        if (val == m->val) return m->str;
    }
    return NULL;
}

enum_map_t formatMap[] = {
    {"metricstatsd",          CFG_METRIC_STATSD},
    {"metricjson",            CFG_METRIC_JSON},
    {"ndjson",                CFG_EVENT_ND_JSON},
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
    {NULL,                    -1}
};

// forward declarations
void cfgOutFormatSetFromStr(config_t*, const char*);
void cfgOutStatsDPrefixSetFromStr(config_t*, const char*);
void cfgOutStatsDMaxLenSetFromStr(config_t*, const char*);
void cfgOutPeriodSetFromStr(config_t*, const char*);
void cfgCmdDirSetFromStr(config_t*, const char*);
void cfgEventFormatSetFromStr(config_t*, const char*);
void cfgEventValueFilterSetFromStr(config_t*, cfg_evt_t, const char*);
void cfgEventFieldFilterSetFromStr(config_t*, cfg_evt_t, const char*);
void cfgEventNameFilterSetFromStr(config_t*, cfg_evt_t, const char*);
void cfgEventSourceEnabledSetFromStr(config_t*, cfg_evt_t, const char*);
void cfgOutVerbositySetFromStr(config_t*, const char*);
void cfgTransportSetFromStr(config_t*, which_transport_t, const char*);
void cfgCustomTagAddFromStr(config_t*, const char*, const char*);
void cfgLogLevelSetFromStr(config_t*, const char*);

// These global variables limits us to only reading one config file at a time...
// which seems fine for now, I guess.
static which_transport_t transport_context;
static cfg_evt_t watch_context;

static regex_t* g_regex = NULL;

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

    while (re && !regexec(re, inptr, 1, &match, 0)) {

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
processEnvStyleInput(config_t* cfg, const char* env_line)
{

    if (!cfg || !env_line) return;

    char* env_ptr, *value;
    if (!(env_ptr = strchr(env_line, '='))) return;
    if (!(value = doEnvVariableSubstitution(&env_ptr[1]))) return;

    if (startsWith(env_line, "SCOPE_METRIC_FORMAT")) {
        cfgOutFormatSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_PREFIX")) {
        cfgOutStatsDPrefixSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_MAXLEN")) {
        cfgOutStatsDMaxLenSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_SUMMARY_PERIOD")) {
        cfgOutPeriodSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_CMD_DIR")) {
        cfgCmdDirSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_METRIC_VERBOSITY")) {
        cfgOutVerbositySetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_LOG_LEVEL")) {
        cfgLogLevelSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_METRIC_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_OUT, value);
    } else if (startsWith(env_line, "SCOPE_LOG_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_LOG, value);
    } else if (startsWith(env_line, "SCOPE_TAG_")) {
        processCustomTag(cfg, env_line, value);
    } else if (startsWith(env_line, "SCOPE_CMD_DBG_PATH")) {
        processCmdDebug(value);
    } else if (startsWith(env_line, "SCOPE_EVENT_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_CTL, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_FORMAT")) {
        cfgEventFormatSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_NAME")) {
        cfgEventNameFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_NAME")) {
        cfgEventNameFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_NAME")) {
        cfgEventNameFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_NAME")) {
        cfgEventNameFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_FIELD")) {
        cfgEventFieldFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_FIELD")) {
        cfgEventFieldFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_FIELD")) {
        cfgEventFieldFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_FIELD")) {
        cfgEventFieldFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE_VALUE")) {
        cfgEventValueFilterSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE_VALUE")) {
        cfgEventValueFilterSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG_VALUE")) {
        cfgEventValueFilterSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC_VALUE")) {
        cfgEventValueFilterSetFromStr(cfg, CFG_SRC_METRIC, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_LOGFILE")) {
        cfgEventSourceEnabledSetFromStr(cfg, CFG_SRC_FILE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_CONSOLE")) {
        cfgEventSourceEnabledSetFromStr(cfg, CFG_SRC_CONSOLE, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_SYSLOG")) {
        cfgEventSourceEnabledSetFromStr(cfg, CFG_SRC_SYSLOG, value);
    } else if (startsWith(env_line, "SCOPE_EVENT_METRIC")) {
        cfgEventSourceEnabledSetFromStr(cfg, CFG_SRC_METRIC, value);
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
cfgOutFormatSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgOutFormatSet(cfg, strToVal(formatMap, value));
}

void
cfgOutStatsDPrefixSetFromStr(config_t* cfg, const char* value)
{
    // A little silly to define passthrough function
    // but this keeps the interface consistent.
    cfgOutStatsDPrefixSet(cfg, value);
}

void
cfgOutStatsDMaxLenSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutStatsDMaxLenSet(cfg, x);
}

void
cfgOutPeriodSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutPeriodSet(cfg, x);
}

void
cfgCmdDirSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgCmdDirSet(cfg, value);
}

void
cfgEventFormatSetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    cfgEventFormatSet(cfg, strToVal(formatMap, value));
}

void
cfgEventValueFilterSetFromStr(config_t* cfg, cfg_evt_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEventValueFilterSet(cfg, src, value);
}

void
cfgEventFieldFilterSetFromStr(config_t* cfg, cfg_evt_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEventFieldFilterSet(cfg, src, value);
}

void
cfgEventNameFilterSetFromStr(config_t* cfg, cfg_evt_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEventNameFilterSet(cfg, src, value);
}

void
cfgEventSourceEnabledSetFromStr(config_t* cfg, cfg_evt_t src, const char* value)
{
    if (!cfg || !value) return;
    cfgEventSourceEnabledSet(cfg, src, !strcmp("true", value));
}

void
cfgOutVerbositySetFromStr(config_t* cfg, const char* value)
{
    if (!cfg || !value) return;
    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutVerbositySet(cfg, x);
}

void
cfgTransportSetFromStr(config_t* cfg, which_transport_t t, const char* value)
{
    if (!cfg || !value) return;

    // see if value starts with udp:// or file://
    if (value == strstr(value, "udp://")) {

        // copied to avoid directly modifing the process's env variable
        char value_cpy[1024];
        strncpy(value_cpy, value, sizeof(value_cpy));

        char* host = value_cpy + strlen("udp://");

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

        char* host = value_cpy + strlen("tcp://");

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
        const char* path = value + strlen("file://");
        cfgTransportTypeSet(cfg, t, CFG_FILE);
        cfgTransportPathSet(cfg, t, path);
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
    transport_context = CFG_OUT;
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
    if (node->type != YAML_SEQUENCE_NODE) return;

    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* i = yaml_document_get_node(doc, *item);
        if (i->type != YAML_MAPPING_NODE) continue;

        yaml_node_pair_t* pair = i->data.mapping.pairs.start;
        yaml_node_t* key = yaml_document_get_node(doc, pair->key);
        yaml_node_t* value = yaml_document_get_node(doc, pair->value);

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
    cfgOutFormatSetFromStr(config, value);
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
processStatsDPrefix(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgOutStatsDPrefixSetFromStr(config, value);
    if (value) free(value);
}

static void
processStatsDMaxLen(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgOutStatsDMaxLenSetFromStr(config, value);
    if (value) free(value);
}

static void
processVerbosity(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    char* value = stringVal(node);
    cfgOutVerbositySetFromStr(config, value);
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
        {YAML_SEQUENCE_NODE,  TAGS_NODE,            processTags},
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
    cfgOutPeriodSetFromStr(config, value);
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
processMetric(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
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
processEvtFormat(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,    TYPE_NODE,            processFormatTypeEvent},
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
    cfgEventSourceEnabledSet(config, watch_context, 1);
    if (value) free(value);
}

static void
processWatchName(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEventNameFilterSetFromStr(config, watch_context, value);
    if (value) free(value);
}

static void
processWatchField(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEventFieldFilterSetFromStr(config, watch_context, value);
    if (value) free(value);
}

static void
processWatchValue(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;

    char* value = stringVal(node);
    cfgEventValueFilterSetFromStr(config, watch_context, value);
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
    if (node->type != YAML_SEQUENCE_NODE) return;

    // absence of one of these values means to clear it.
    // clear them all, then set values for whatever we find.
    cfg_evt_t x;
    for (x = CFG_SRC_FILE; x<CFG_SRC_MAX; x++) {
        cfgEventSourceEnabledSet(config, x, 0);
    }

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
        {YAML_MAPPING_NODE,   FORMAT_NODE,          processEvtFormat},
        {YAML_SEQUENCE_NODE,  WATCH_NODE,           processWatch},
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
        {YAML_MAPPING_NODE,   TRANSPORT_NODE,       processTransportCtl},
        {YAML_MAPPING_NODE,   LOG_NODE,             processLogging},
        {YAML_SCALAR_NODE,    SUMMARYPERIOD_NODE,   processSummaryPeriod},
        {YAML_SCALAR_NODE,    COMMANDDIR_NODE,      processCommandDir},
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
        {YAML_MAPPING_NODE,   EVENT_NODE,           processEvent},
        {YAML_NO_NODE,        NULL,                 NULL}
    };

    yaml_node_pair_t* pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

config_t*
cfgRead(const char* path)
{
    FILE* f = NULL;
    config_t* config = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    yaml_parser_t parser;
    yaml_document_t doc;
    // ni for "not-interposed"... a direct glibc call without scope.
    FILE *(*ni_fopen)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
    int (*ni_fclose)(FILE*) = dlsym(RTLD_NEXT, "fclose");

    if (!ni_fopen || !ni_fclose) goto cleanup;

    config = cfgCreateDefault();
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
    return config;
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
    if (!(root = cJSON_CreateArray())) goto err;

    custom_tag_t **tags = cfgCustomTags(cfg);
    int i;
    if (tags) {
        for (i=0; tags[i]; i++) {
            cJSON* item;
            if (!(item = cJSON_CreateObject())) continue;
            if (!(cJSON_AddStringToObject(item, tags[i]->name, tags[i]->value))) {
                DBG(NULL);
                cJSON_Delete(item);
            } else {
                cJSON_AddItemToArray(root, item);
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
                     valToStr(formatMap, cfgOutFormat(cfg)))) goto err;
    if (!cJSON_AddStringToObjLN(root, STATSDPREFIX_NODE,
                                    cfgOutStatsDPrefix(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, STATSDMAXLEN_NODE,
                                    cfgOutStatsDMaxLen(cfg))) goto err;
    if (!cJSON_AddNumberToObjLN(root, VERBOSITY_NODE,
                                       cfgOutVerbosity(cfg))) goto err;

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

    if (!(transport = createTransportJson(cfg, CFG_OUT))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    if (!(format = createMetricFormatJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, FORMAT_NODE, format);

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createWatchObjectJson(config_t* cfg, cfg_evt_t evt)
{
    cJSON* root = NULL;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!cJSON_AddStringToObjLN(root, TYPE_NODE,
                                    valToStr(watchTypeMap, evt))) goto err;
    if (!cJSON_AddStringToObjLN(root, NAME_NODE,
                                   cfgEventNameFilter(cfg, evt))) goto err;
    if (!cJSON_AddStringToObjLN(root, FIELD_NODE,
                                  cfgEventFieldFilter(cfg, evt))) goto err;
    if (!cJSON_AddStringToObjLN(root, VALUE_NODE,
                                  cfgEventValueFilter(cfg, evt))) goto err;

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

    cfg_evt_t evt;
    for (evt = CFG_SRC_FILE; evt<CFG_SRC_MAX; evt++) {
        cJSON* item;
        if (!cfgEventSourceEnabled(cfg, evt)) continue;
        if (!(item = createWatchObjectJson(cfg, evt))) continue;
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

    return root;
err:
    if (root) cJSON_Delete(root);
    return NULL;
}

static cJSON*
createEventJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* format, *watch;

    if (!(root = cJSON_CreateObject())) goto err;

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
createLibscopeJson(config_t* cfg)
{
    cJSON* root = NULL;
    cJSON* transport, *log;

    if (!(root = cJSON_CreateObject())) goto err;

    if (!(transport = createTransportJson(cfg, CFG_CTL))) goto err;
    cJSON_AddItemToObjectCS(root, TRANSPORT_NODE, transport);

    if (!(log = createLogJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(root, LOG_NODE, log);

    if (!cJSON_AddNumberToObjLN(root, SUMMARYPERIOD_NODE,
                                      cfgOutPeriod(cfg))) goto err;

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
    cJSON* metric, *libscope, *event;

    if (!(json_root = cJSON_CreateObject())) goto err;

    if (!(metric = createMetricJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, METRIC_NODE, metric);

    if (!(libscope = createLibscopeJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, LIBSCOPE_NODE, libscope);

     if (!(event = createEventJson(cfg))) goto err;
    cJSON_AddItemToObjectCS(json_root, EVENT_NODE, event);

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

static format_t*
initFormat(config_t* cfg)
{
    format_t* fmt = fmtCreate(cfgOutFormat(cfg));
    if (!fmt) return NULL;

    fmtStatsDPrefixSet(fmt, cfgOutStatsDPrefix(cfg));
    fmtStatsDMaxLenSet(fmt, cfgOutStatsDMaxLen(cfg));
    fmtOutVerbositySet(fmt, cfgOutVerbosity(cfg));
    fmtCustomTagsSet(fmt, cfgCustomTags(cfg));
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

out_t*
initOut(config_t* cfg)
{
    out_t* out = outCreate();
    if (!out) return out;

    transport_t* t = initTransport(cfg, CFG_OUT);
    if (!t) {
        outDestroy(&out);
        return out;
    }
    outTransportSet(out, t);

    format_t* f = initFormat(cfg);
    if (!f) {
        outDestroy(&out);
        return out;
    }
    outFormatSet(out, f);

    return out;
}

evt_t *
initEvt(config_t *cfg)
{
    evt_t *evt = evtCreate();
    if (!evt) return evt;

    format_t *fmt = fmtCreate(cfgEventFormat(cfg));
    if (!fmt) {
        evtDestroy(&evt);
        return evt;
    }
    evtFormatSet(evt, fmt);

    cfg_evt_t src;
    for (src = CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        evtSourceEnabledSet(evt, src, cfgEventSourceEnabled(cfg, src));
        evtNameFilterSet(evt, src, cfgEventNameFilter(cfg, src));
        evtFieldFilterSet(evt, src, cfgEventFieldFilter(cfg, src));
        evtValueFilterSet(evt, src, cfgEventValueFilter(cfg, src));
    }

    return evt;
}

ctl_t*
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
    ctlTransportSet(ctl, trans);

    return ctl;
}
