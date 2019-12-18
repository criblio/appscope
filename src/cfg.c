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
        char* valuefilter[CFG_SRC_MAX];
        char* fieldfilter[CFG_SRC_MAX];
        char* namefilter[CFG_SRC_MAX];
        unsigned src[CFG_SRC_MAX];
    } evt;

    struct {
        cfg_log_level_t level;
    } log;

    // CFG_OUT, CFG_CTL, or CFG_LOG
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
#define DEFAULT_CTL_TYPE CFG_TCP
#define DEFAULT_CTL_HOST "127.0.0.1"
//#define DEFAULT_CTL_PORT DEFAULT_CTL_PORT (defined in scopetypes.h)
#define DEFAULT_CTL_PATH NULL
#define DEFAULT_CTL_BUF CFG_BUFFER_FULLY
#define DEFAULT_LOG_TYPE CFG_FILE
#define DEFAULT_LOG_HOST NULL
#define DEFAULT_LOG_PORT NULL
#define DEFAULT_LOG_PATH "/tmp/scope.log"
#define DEFAULT_LOG_BUF CFG_BUFFER_FULLY
#define DEFAULT_TAGS NULL
#define DEFAULT_NUM_TAGS 8
#define DEFAULT_COMMAND_DIR "/tmp"


static const char* valueFilterDefault[] = {
    DEFAULT_SRC_FILE_VALUE,
    DEFAULT_SRC_CONSOLE_VALUE,
    DEFAULT_SRC_SYSLOG_VALUE,
    DEFAULT_SRC_METRIC_VALUE,
};

static const char* fieldFilterDefault[] = {
    DEFAULT_SRC_FILE_FIELD,
    DEFAULT_SRC_CONSOLE_FIELD,
    DEFAULT_SRC_SYSLOG_FIELD,
    DEFAULT_SRC_METRIC_FIELD,
};

static const char* nameFilterDefault[] = {
    DEFAULT_SRC_FILE_NAME,
    DEFAULT_SRC_CONSOLE_NAME,
    DEFAULT_SRC_SYSLOG_NAME,
    DEFAULT_SRC_METRIC_NAME,
};

static unsigned srcEnabledDefault[] = {
    DEFAULT_SRC_FILE,
    DEFAULT_SRC_CONSOLE,
    DEFAULT_SRC_SYSLOG,
    DEFAULT_SRC_METRIC,
};

static cfg_transport_t typeDefault[] = {
    DEFAULT_OUT_TYPE,
    DEFAULT_CTL_TYPE,
    DEFAULT_LOG_TYPE,
};

static const char* hostDefault[] = {
    DEFAULT_OUT_HOST,
    DEFAULT_CTL_HOST,
    DEFAULT_LOG_HOST,
};

static const char* portDefault[] = {
    DEFAULT_OUT_PORT,
    DEFAULT_CTL_PORT,
    DEFAULT_LOG_PORT,
};

static const char* pathDefault[] = {
    DEFAULT_OUT_PATH,
    DEFAULT_CTL_PATH,
    DEFAULT_LOG_PATH,
};

static cfg_buffer_t bufDefault[] = {
    DEFAULT_OUT_BUF,
    DEFAULT_CTL_BUF,
    DEFAULT_LOG_BUF,
};

    
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
    c->evt.format = DEFAULT_CTL_FORMAT;

    cfg_evt_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        const char* val_def = valueFilterDefault[src];
        c->evt.valuefilter[src] = (val_def) ? strdup(val_def) : NULL;
        const char* field_def = fieldFilterDefault[src];
        c->evt.fieldfilter[src] = (field_def) ? strdup(field_def) : NULL;
        const char* name_def = nameFilterDefault[src];
        c->evt.namefilter[src] = (name_def) ? strdup(name_def) : NULL;
        c->evt.src[src] = srcEnabledDefault[src];
    }

    which_transport_t tp;
    for (tp=CFG_OUT; tp<CFG_WHICH_MAX; tp++) {
        c->transport[tp].type = typeDefault[tp];
        const char* host_def = hostDefault[tp];
        c->transport[tp].net.host = (host_def) ? strdup(host_def) : NULL;
        const char* port_def = portDefault[tp];
        c->transport[tp].net.port = (port_def) ? strdup(port_def) : NULL;
        const char* path_def = pathDefault[tp];
        c->transport[tp].file.path = (path_def) ? strdup(path_def) : NULL;
        c->transport[tp].file.buf_policy = bufDefault[tp];
    }

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

    cfg_evt_t src;
    for (src = CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (c->evt.valuefilter[src]) free (c->evt.valuefilter[src]);
        if (c->evt.fieldfilter[src]) free (c->evt.fieldfilter[src]);
        if (c->evt.namefilter[src]) free (c->evt.namefilter[src]);
    }

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
    return (cfg) ? cfg->evt.format : DEFAULT_CTL_FORMAT;
}

const char*
cfgEventValueFilter(config_t* cfg, cfg_evt_t evt)
{
    if (evt >= 0 && evt < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.valuefilter[evt];
        return valueFilterDefault[evt];
    }

    DBG("%d", evt);
    return valueFilterDefault[CFG_SRC_FILE];
}

const char*
cfgEventFieldFilter(config_t* cfg, cfg_evt_t evt)
{
    if (evt >= 0 && evt < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.fieldfilter[evt];
        return fieldFilterDefault[evt];
    }

    DBG("%d", evt);
    return fieldFilterDefault[CFG_SRC_FILE];
}

const char*
cfgEventNameFilter(config_t* cfg, cfg_evt_t evt)
{
    if (evt >= 0 && evt < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.namefilter[evt];
        return nameFilterDefault[evt];
    }

    DBG("%d", evt);
    return nameFilterDefault[CFG_SRC_FILE];
}

unsigned
cfgEventSourceEnabled(config_t* cfg, cfg_evt_t evt)
{
    if (evt >= 0 && evt < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.src[evt];
        return srcEnabledDefault[evt];
    }

    DBG("%d", evt);
    return srcEnabledDefault[CFG_SRC_FILE];
}


unsigned
cfgOutVerbosity(config_t* cfg)
{
    return (cfg) ? cfg->out.verbosity : DEFAULT_OUT_VERBOSITY;
}

cfg_transport_t
cfgTransportType(config_t* cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].type;
        return typeDefault[t];
    }

    DBG("%d", t);
    return typeDefault[CFG_LOG];
}

const char*
cfgTransportHost(config_t* cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].net.host;
        return hostDefault[t];
    } 

    DBG("%d", t);
    return hostDefault[CFG_LOG];
}

const char*
cfgTransportPort(config_t* cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].net.port;
        return portDefault[t];
    }

    DBG("%d", t);
    return portDefault[CFG_LOG];
}

const char*
cfgTransportPath(config_t* cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].file.path;
        return pathDefault[t];
    }

    DBG("%d", t);
    return pathDefault[CFG_LOG];
}

cfg_buffer_t
cfgTransportBuf(config_t* cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].file.buf_policy;
        return bufDefault[t];
    }

    DBG("%d", t);
    return bufDefault[CFG_LOG];
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
    if (!cfg || fmt < 0 || fmt >= CFG_FORMAT_MAX) return;
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
    if (!cfg || fmt < 0 || fmt >= CFG_FORMAT_MAX) return;
    cfg->evt.format = fmt;
}

void
cfgEventValueFilterSet(config_t* cfg, cfg_evt_t evt, const char* filter)
{
    if (!cfg || evt < 0 || evt >= CFG_SRC_MAX) return;
    if (cfg->evt.valuefilter[evt]) free (cfg->evt.valuefilter[evt]);
    if (!filter || (filter[0] == '\0')) {
        const char* vdefault = valueFilterDefault[evt];
        cfg->evt.valuefilter[evt] = (vdefault) ? strdup(vdefault) : NULL;
        return;
    }
    cfg->evt.valuefilter[evt] = strdup(filter);
}

void
cfgEventFieldFilterSet(config_t* cfg, cfg_evt_t evt, const char* filter)
{
    if (!cfg || evt < 0 || evt >= CFG_SRC_MAX) return;
    if (cfg->evt.fieldfilter[evt]) free (cfg->evt.fieldfilter[evt]);
    if (!filter || (filter[0] == '\0')) {
        const char* fdefault = fieldFilterDefault[evt];
        cfg->evt.fieldfilter[evt] = (fdefault) ? strdup(fdefault) : NULL;
        return;
    }
    cfg->evt.fieldfilter[evt] = strdup(filter);
}

void
cfgEventNameFilterSet(config_t* cfg, cfg_evt_t evt, const char* filter)
{
    if (!cfg || evt < 0 || evt >= CFG_SRC_MAX) return;
    if (cfg->evt.namefilter[evt]) free (cfg->evt.namefilter[evt]);
    if (!filter || (filter[0] == '\0')) {
        const char* ndefault = nameFilterDefault[evt];
        cfg->evt.namefilter[evt] = (ndefault) ? strdup(ndefault) : NULL;
        return;
    }
    cfg->evt.namefilter[evt] = strdup(filter);
}

void
cfgEventSourceEnabledSet(config_t* cfg, cfg_evt_t evt, unsigned val)
{
    if (!cfg || evt < 0 || evt >= CFG_SRC_MAX) return;
    cfg->evt.src[evt] = val;
}

void
cfgTransportTypeSet(config_t* cfg, which_transport_t t, cfg_transport_t type)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (type < 0 || type > CFG_TCP) return;
    cfg->transport[t].type = type;
}

void
cfgTransportHostSet(config_t* cfg, which_transport_t t, const char* host)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].net.host) free(cfg->transport[t].net.host);
    cfg->transport[t].net.host = (host) ? strdup(host) : NULL;

}

void
cfgTransportPortSet(config_t* cfg, which_transport_t t, const char* port)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].net.port) free(cfg->transport[t].net.port);
    cfg->transport[t].net.port = (port) ? strdup(port) : NULL;
}

void
cfgTransportPathSet(config_t* cfg, which_transport_t t, const char* path)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].file.path) free(cfg->transport[t].file.path);
    cfg->transport[t].file.path = (path) ? strdup(path) : NULL;
}

void
cfgTransportBufSet(config_t* cfg, which_transport_t t, cfg_buffer_t buf)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (buf < CFG_BUFFER_FULLY || buf > CFG_BUFFER_LINE) return;
    cfg->transport[t].file.buf_policy = buf;
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
    if (!cfg || level < 0 || level > CFG_LOG_NONE) return;
    cfg->log.level = level;
}
