#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "cfg.h"
#include "dbg.h"

typedef struct {
    cfg_transport_t type;
    struct {                             // For type = CFG_UDP
        char* host;
        char* port;
    } net;
    struct {
        char* path;                      // For type CFG_FILE
        cfg_buffer_t buf_policy;
    } file;
} transport_struct_t;

struct _config_t
{
    struct {
        cfg_out_format_t format;
        struct {
            char* prefix;
            unsigned maxlen;
        } statsd;
        unsigned period;
        unsigned verbosity;
        char* commanddir;
    } out;

    struct {
        cfg_out_format_t format;
        char* logfilefilter;
        unsigned src[CFG_SRC_MAX];
    } evt;

    struct {
        cfg_log_level_t level;
    } log;

    // CFG_OUT, CFG_EVT, or CFG_LOG
    transport_struct_t transport[CFG_WHICH_MAX]; 

    custom_tag_t** tags;
    unsigned max_tags;
};

#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_OUT_TYPE CFG_UDP
#define DEFAULT_OUT_HOST "127.0.0.1"
//#define DEFAULT_OUT_PORT DEFAULT_OUT_PORT (defined in scopetypes.h)
#define DEFAULT_OUT_PATH NULL
#define DEFAULT_OUT_BUF CFG_BUFFER_FULLY
#define DEFAULT_EVT_TYPE CFG_TCP
#define DEFAULT_EVT_HOST "127.0.0.1"
//#define DEFAULT_EVT_PORT DEFAULT_EVT_PORT (defined in scopetypes.h)
#define DEFAULT_EVT_PATH NULL
#define DEFAULT_EVT_BUF CFG_BUFFER_FULLY
#define DEFAULT_LOG_TYPE CFG_FILE
#define DEFAULT_LOG_HOST NULL
#define DEFAULT_LOG_PORT NULL
#define DEFAULT_LOG_PATH "/tmp/scope.log"
#define DEFAULT_LOG_BUF CFG_BUFFER_FULLY
#define DEFAULT_TAGS NULL
#define DEFAULT_NUM_TAGS 8
#define DEFAULT_COMMAND_DIR "/tmp"

    
///////////////////////////////////
// Constructors Destructors
///////////////////////////////////
config_t*
cfgCreateDefault()
{ 
    config_t* c = calloc(1, sizeof(config_t));
    if (!c) {
        DBG(NULL);
        return NULL;
    }
    c->out.format = DEFAULT_OUT_FORMAT;
    c->out.statsd.prefix = (DEFAULT_STATSD_PREFIX) ? strdup(DEFAULT_STATSD_PREFIX) : NULL;
    c->out.statsd.maxlen = DEFAULT_STATSD_MAX_LEN;
    c->out.period = DEFAULT_SUMMARY_PERIOD;
    c->out.verbosity = DEFAULT_OUT_VERBOSITY;
    c->out.commanddir = (DEFAULT_COMMAND_DIR) ? strdup(DEFAULT_COMMAND_DIR) : NULL;
    c->evt.format = DEFAULT_EVT_FORMAT;
    c->evt.logfilefilter = (DEFAULT_LOG_FILE_FILTER) ? strdup(DEFAULT_LOG_FILE_FILTER) : NULL;
    c->evt.src[CFG_SRC_LOGFILE] = DEFAULT_SRC_LOGFILE;
    c->evt.src[CFG_SRC_CONSOLE] = DEFAULT_SRC_CONSOLE;;
    c->evt.src[CFG_SRC_SYSLOG] = DEFAULT_SRC_SYSLOG;
    c->evt.src[CFG_SRC_METRIC] = DEFAULT_SRC_METRIC;
    c->transport[CFG_OUT].type = DEFAULT_OUT_TYPE;
    c->transport[CFG_OUT].net.host = (DEFAULT_OUT_HOST) ? strdup(DEFAULT_OUT_HOST) : NULL;
    c->transport[CFG_OUT].net.port = (DEFAULT_OUT_PORT) ? strdup(DEFAULT_OUT_PORT) : NULL;
    c->transport[CFG_OUT].file.path = (DEFAULT_OUT_PATH) ? strdup(DEFAULT_OUT_PATH) : NULL;
    c->transport[CFG_OUT].file.buf_policy = DEFAULT_OUT_BUF;
    c->transport[CFG_EVT].type = DEFAULT_EVT_TYPE;
    c->transport[CFG_EVT].net.host = (DEFAULT_EVT_HOST) ? strdup(DEFAULT_EVT_HOST) : NULL;
    c->transport[CFG_EVT].net.port = (DEFAULT_EVT_PORT) ? strdup(DEFAULT_EVT_PORT) : NULL;
    c->transport[CFG_EVT].file.path = (DEFAULT_EVT_PATH) ? strdup(DEFAULT_EVT_PATH) : NULL;
    c->transport[CFG_EVT].file.buf_policy = DEFAULT_EVT_BUF;
    c->transport[CFG_LOG].type = DEFAULT_LOG_TYPE;
    c->transport[CFG_LOG].net.host = (DEFAULT_LOG_HOST) ? strdup(DEFAULT_LOG_HOST) : NULL;
    c->transport[CFG_LOG].net.port = (DEFAULT_LOG_PORT) ? strdup(DEFAULT_LOG_PORT) : NULL;
    c->transport[CFG_LOG].file.path = (DEFAULT_LOG_PATH) ? strdup(DEFAULT_LOG_PATH) : NULL;
    c->transport[CFG_LOG].file.buf_policy = DEFAULT_LOG_BUF;
    c->tags = DEFAULT_TAGS;
    c->max_tags = DEFAULT_NUM_TAGS;
    c->log.level = DEFAULT_LOG_LEVEL;

    return c;
}

void
cfgDestroy(config_t** cfg)
{
    if (!cfg || !*cfg) return;
    config_t* c = *cfg;
    if (c->out.statsd.prefix) free(c->out.statsd.prefix);
    if (c->out.commanddir) free(c->out.commanddir);
    if (c->evt.logfilefilter) free(c->evt.logfilefilter);
    which_transport_t t;
    for (t=CFG_OUT; t<CFG_WHICH_MAX; t++) {
        if (c->transport[t].net.host) free(c->transport[t].net.host);
        if (c->transport[t].net.port) free(c->transport[t].net.port);
        if (c->transport[t].file.path) free(c->transport[t].file.path);
    }
    if (c->tags) {
        int i = 0;
        while (c->tags[i]) {
            free(c->tags[i]->name);
            free(c->tags[i]->value);
            free(c->tags[i]);
            i++;
        }
        free(c->tags);
    }
    free(c);
    *cfg = NULL;
}

///////////////////////////////////
// Accessors
///////////////////////////////////
cfg_out_format_t
cfgOutFormat(config_t* cfg)
{
    return (cfg) ? cfg->out.format : DEFAULT_OUT_FORMAT;
}

const char*
cfgOutStatsDPrefix(config_t* cfg)
{
    return (cfg && cfg->out.statsd.prefix) ? cfg->out.statsd.prefix : DEFAULT_STATSD_PREFIX;
}

unsigned
cfgOutStatsDMaxLen(config_t* cfg)
{
    return (cfg) ? cfg->out.statsd.maxlen : DEFAULT_STATSD_MAX_LEN;
}

unsigned
cfgOutPeriod(config_t* cfg)
{
    return (cfg) ? cfg->out.period : DEFAULT_SUMMARY_PERIOD;
}

const char *
cfgCmdDir(config_t* cfg)
{
    return (cfg) ? cfg->out.commanddir : DEFAULT_COMMAND_DIR;
}

cfg_out_format_t
cfgEventFormat(config_t* cfg)
{
    return (cfg) ? cfg->evt.format : DEFAULT_EVT_FORMAT;
}

const char*
cfgEventLogFileFilter(config_t* cfg)
{
    return (cfg) ? cfg->evt.logfilefilter : DEFAULT_LOG_FILE_FILTER;
}

unsigned
cfgEventSource(config_t* cfg, cfg_evt_t evt)
{
    if (cfg && evt < CFG_SRC_MAX) {
        return cfg->evt.src[evt];
    }

    switch (evt) {
        case CFG_SRC_LOGFILE:
            return DEFAULT_SRC_LOGFILE;
        case CFG_SRC_CONSOLE:
            return DEFAULT_SRC_CONSOLE;
        case CFG_SRC_SYSLOG:
            return DEFAULT_SRC_SYSLOG;
        case CFG_SRC_METRIC:
            return DEFAULT_SRC_METRIC;
        default:
            DBG(NULL);
            return DEFAULT_SRC_LOGFILE;
    }
}


unsigned
cfgOutVerbosity(config_t* cfg)
{
    return (cfg) ? cfg->out.verbosity : DEFAULT_OUT_VERBOSITY;
}

cfg_transport_t
cfgTransportType(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].type;
    }
 
    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_TYPE;
        case CFG_EVT:
            return DEFAULT_EVT_TYPE;
        case CFG_LOG:
            return DEFAULT_LOG_TYPE;
        default:
            DBG("%d", t);
            return DEFAULT_LOG_TYPE;
    }
}

const char*
cfgTransportHost(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].net.host;
    } 


    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_HOST;
        case CFG_EVT:
            return DEFAULT_EVT_HOST;
        case CFG_LOG:
            return DEFAULT_LOG_HOST;
        default:
            DBG("%d", t);
            return DEFAULT_LOG_HOST;
    }
}

const char*
cfgTransportPort(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].net.port;
    }

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_PORT;
        case CFG_EVT:
            return DEFAULT_EVT_PORT;
        case CFG_LOG:
            return DEFAULT_LOG_PORT;
        default:
            DBG("%d", t);
            return DEFAULT_LOG_PORT;
    }
}

const char*
cfgTransportPath(config_t* cfg, which_transport_t t)
{
    if (cfg && t  < CFG_WHICH_MAX) {
        return cfg->transport[t].file.path;
    }

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_PATH;
        case CFG_EVT:
            return DEFAULT_EVT_PATH;
        case CFG_LOG:
            return DEFAULT_LOG_PATH;
        default:
            DBG("%d", t);
            return DEFAULT_LOG_PATH;
    }
}

cfg_buffer_t
cfgTransportBuf(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].file.buf_policy;
    }

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_BUF;
        case CFG_EVT:
            return DEFAULT_EVT_BUF;
        case CFG_LOG:
            return DEFAULT_LOG_BUF;
        default:
            DBG("%d", t);
            return DEFAULT_LOG_BUF;
    }
}

custom_tag_t**
cfgCustomTags(config_t* cfg)
{
    return (cfg) ? cfg->tags : DEFAULT_TAGS;
}

static custom_tag_t*
cfgCustomTag(config_t* cfg, const char* tagName)
{
    if (!tagName) return NULL;
    if (!cfg || !cfg->tags) return NULL;

    int i = 0;
    while (cfg->tags[i]) {
        if (!strcmp(tagName, cfg->tags[i]->name)) {
            // tagName appears in the list of tags.
            return cfg->tags[i];
        }
        i++;
    }
    return NULL;
}

const char *
cfgCustomTagValue(config_t* cfg, const char* tagName)
{
    custom_tag_t* tag = cfgCustomTag(cfg, tagName);
    if (tag) return tag->value;

    return NULL;
}

cfg_log_level_t
cfgLogLevel(config_t* cfg)
{
    return (cfg) ? cfg->log.level : DEFAULT_LOG_LEVEL;
}

///////////////////////////////////
// Setters 
///////////////////////////////////
void
cfgOutFormatSet(config_t* cfg, cfg_out_format_t fmt)
{
    if (!cfg) return;
    cfg->out.format = fmt;
}

void
cfgOutStatsDPrefixSet(config_t* cfg, const char* prefix)
{
    if (!cfg) return;
    if (cfg->out.statsd.prefix) free(cfg->out.statsd.prefix);
    if (!prefix || prefix[0] == '\0') {
        cfg->out.statsd.prefix = strdup(DEFAULT_STATSD_PREFIX);
        return;
    }

    // Make sure that the prefix always ends in a '.'
    int n = strlen(prefix);
    if (prefix[n-1] != '.') {
        char* temp = malloc(n+2);
        if (temp) {
            strcpy(temp, prefix);
            temp[n] = '.';
            temp[n+1] = '\0';
        } else {
            DBG("%s", prefix);
        }
        cfg->out.statsd.prefix = temp;
    } else {
        cfg->out.statsd.prefix = strdup(prefix);
    }
}

void
cfgOutStatsDMaxLenSet(config_t* cfg, unsigned len)
{
    if (!cfg) return;
    cfg->out.statsd.maxlen = len;
}

void
cfgOutPeriodSet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    cfg->out.period = val;
}

void
cfgCmdDirSet(config_t* cfg, const char* path)
{
    if (!cfg) return;
    if (cfg->out.commanddir) free(cfg->out.commanddir);
    if (!path || (path[0] == '\0')) {
        cfg->out.commanddir = (DEFAULT_COMMAND_DIR) ? strdup(DEFAULT_COMMAND_DIR) : NULL;
        return;
    }

    cfg->out.commanddir = strdup(path);
}

void
cfgOutVerbositySet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    if (val > CFG_MAX_VERBOSITY) val = CFG_MAX_VERBOSITY;
    cfg->out.verbosity = val;
}

void
cfgEventFormatSet(config_t* cfg, cfg_out_format_t fmt)
{
    if (!cfg || fmt >= CFG_FORMAT_MAX) return;
    cfg->evt.format = fmt;
}

void
cfgEventLogFileFilterSet(config_t* cfg,  const char* filter)
{
    if (!cfg) return;
    if (cfg->evt.logfilefilter) free (cfg->evt.logfilefilter);
    if (!filter || (filter[0] == '\0')) {
        cfg->evt.logfilefilter = (DEFAULT_LOG_FILE_FILTER) ? strdup(DEFAULT_LOG_FILE_FILTER) : NULL;
        return;
    }
    cfg->evt.logfilefilter = strdup(filter);
}

void
cfgEventSourceSet(config_t* cfg, cfg_evt_t evt, unsigned val)
{
    if (!cfg || evt >= CFG_SRC_MAX) return;
    cfg->evt.src[evt] = val;
}

void
cfgTransportTypeSet(config_t* cfg, which_transport_t t, cfg_transport_t type)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;
    cfg->transport[t].type = type;
}

void
cfgTransportHostSet(config_t* cfg, which_transport_t t, const char* host)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;

    if (cfgTransportType(cfg, t) == CFG_UDP) {
        if (cfg->transport[t].net.host) free(cfg->transport[t].net.host);
        cfg->transport[t].net.host = (host) ? strdup(host) : NULL;
    }

    if (cfgTransportType(cfg, t) == CFG_TCP) {
        if (cfg->transport[t].net.host) free(cfg->transport[t].net.host);
        cfg->transport[t].net.host = (host) ? strdup(host) : NULL;
    }
}

void
cfgTransportPortSet(config_t* cfg, which_transport_t t, const char* port)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;

    if (cfgTransportType(cfg, t) == CFG_UDP) {
        if (cfg->transport[t].net.port) free(cfg->transport[t].net.port);
        cfg->transport[t].net.port = (port) ? strdup(port) : NULL;
    }

    if (cfgTransportType(cfg, t) == CFG_TCP) {
        if (cfg->transport[t].net.port) free(cfg->transport[t].net.port);
        cfg->transport[t].net.port = (port) ? strdup(port) : NULL;
    }
}

void
cfgTransportPathSet(config_t* cfg, which_transport_t t, const char* path)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].file.path) free(cfg->transport[t].file.path);
    cfg->transport[t].file.path = (path) ? strdup(path) : NULL;
}

void
cfgTransportBufSet(config_t* cfg, which_transport_t t, cfg_buffer_t buf_policy)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;
    cfg->transport[t].file.buf_policy = buf_policy;
}

void
cfgCustomTagAdd(config_t* c, const char* name, const char* value)
{
    if (!c || !name || !value) return;

    // If name is already there, replace value only.
    {
        custom_tag_t* t;
        if ((t=cfgCustomTag(c, name))) {
            char* newvalue = strdup(value);
            if (newvalue) {
                free(t->value);
                t->value = newvalue;
                return;
            }
        }
    }

    // Create space if it's the first add
    if (!c->tags) {
        c->tags = calloc(1, sizeof(custom_tag_t*) * c->max_tags);
        if (!c->tags) {
            DBG("%s %s", name, value);
            return;
        }
    }

    // find the first empty spot
    int i;
    for (i=0; i<c->max_tags && c->tags[i]; i++); 

    // If we're out of space, try to get more space
    if (i >= c->max_tags-1) {     // null delimiter is always required
        int tmp_max_tags = c->max_tags * 2;  // double each time
        custom_tag_t** temp = realloc(c->tags, sizeof(custom_tag_t*) * tmp_max_tags);
        if (!temp) {
            DBG("%s %s", name, value);
            return;
        }
        // Yeah!  We have more space!  init it, and set our state to remember it
        memset(&temp[c->max_tags], 0, sizeof(custom_tag_t*) * (tmp_max_tags - c->max_tags));
        c->tags = temp;
        c->max_tags = tmp_max_tags;
    }

    // save it
    {
        custom_tag_t* t = calloc(1,sizeof(custom_tag_t));
        char* n = strdup(name);
        char* v = strdup(value);
        if (!t || !n || !v) {
            if (t) free(t);
            if (n) free(n);
            if (v) free(v);
            DBG("t=%p n=%p v=%p", t, n, v);
            return;
        }
        c->tags[i] = t;
        t->name = n; 
        t->value = v;
    }
}

void
cfgLogLevelSet(config_t* cfg, cfg_log_level_t level)
{
    if (!cfg) return;
    cfg->log.level = level;
}
