#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "dbg.h"
#include "scopestdlib.h"

typedef struct {
    cfg_transport_t type;
    struct {                             // For type = CFG_UDP
        char* host;
        char* port;
        struct {
            unsigned enable;
            unsigned validateserver;
            char *cacertpath;
        } tls;
    } net;
    struct {
        char* path;                      // For type CFG_FILE
        cfg_buffer_t buf_policy;
    } file;
} transport_struct_t;

typedef struct {
    bool valid;
    regex_t re;
    char *filter;
} header_extract_t;

struct _config_t
{
    struct {
        unsigned enable;
        unsigned char categories;
        cfg_mtc_format_t format;
        struct {
            char* prefix;
            unsigned maxlen;
            unsigned enable;
        } statsd;
        unsigned period;
        unsigned verbosity;
    } mtc;

    struct {
        unsigned enable;
        cfg_mtc_format_t format;
        unsigned ratelimit;
        char *valuefilter[CFG_SRC_MAX];
        char *fieldfilter[CFG_SRC_MAX];
        char *namefilter[CFG_SRC_MAX];
        unsigned src[CFG_SRC_MAX];
        size_t numHeaders;
        header_extract_t **hextract;
        unsigned allowbinaryconsole;
    } evt;

    struct {
        cfg_log_level_t level;
    } log;

    struct {
        unsigned int enable;
        char *dir;
    } pay;

    struct {
        unsigned int enable;
        bool cloud;
    } logstream;

    // CFG_MTC, CFG_CTL, or CFG_LOG
    transport_struct_t transport[CFG_WHICH_MAX]; 

    custom_tag_t** tags;
    unsigned max_tags;

    char* commanddir;
    unsigned processstartmsg;
    unsigned enhancefs;
    char *authtoken;
};

static const char* valueFilterDefault[] = {
    DEFAULT_SRC_FILE_VALUE,
    DEFAULT_SRC_CONSOLE_VALUE,
    DEFAULT_SRC_SYSLOG_VALUE,
    DEFAULT_SRC_METRIC_VALUE,
    DEFAULT_SRC_HTTP_VALUE,
    DEFAULT_SRC_NET_VALUE,
    DEFAULT_SRC_FS_VALUE,
    DEFAULT_SRC_DNS_VALUE,
};

static const char* fieldFilterDefault[] = {
    DEFAULT_SRC_FILE_FIELD,
    DEFAULT_SRC_CONSOLE_FIELD,
    DEFAULT_SRC_SYSLOG_FIELD,
    DEFAULT_SRC_METRIC_FIELD,
    DEFAULT_SRC_HTTP_FIELD,
    DEFAULT_SRC_NET_FIELD,
    DEFAULT_SRC_FS_FIELD,
    DEFAULT_SRC_DNS_FIELD,
};

static const char* nameFilterDefault[] = {
    DEFAULT_SRC_FILE_NAME,
    DEFAULT_SRC_CONSOLE_NAME,
    DEFAULT_SRC_SYSLOG_NAME,
    DEFAULT_SRC_METRIC_NAME,
    DEFAULT_SRC_HTTP_NAME,
    DEFAULT_SRC_NET_NAME,
    DEFAULT_SRC_FS_NAME,
    DEFAULT_SRC_DNS_NAME,
};

static unsigned srcEnabledDefault[] = {
    DEFAULT_SRC_FILE,
    DEFAULT_SRC_CONSOLE,
    DEFAULT_SRC_SYSLOG,
    DEFAULT_SRC_METRIC,
    DEFAULT_SRC_HTTP,
    DEFAULT_SRC_NET,
    DEFAULT_SRC_FS,
    DEFAULT_SRC_DNS,
};

static cfg_transport_t typeDefault[] = {
    DEFAULT_MTC_TYPE,
    DEFAULT_CTL_TYPE,
    DEFAULT_LOG_TYPE,
    DEFAULT_LS_TYPE,
};

static const char* hostDefault[] = {
    DEFAULT_MTC_HOST,
    DEFAULT_CTL_HOST,
    DEFAULT_LOG_HOST,
    DEFAULT_LS_HOST,
};

static const char* portDefault[] = {
    DEFAULT_MTC_PORT,
    DEFAULT_CTL_PORT,
    DEFAULT_LOG_PORT,
    DEFAULT_LS_PORT,
};

static const char* pathDefault[] = {
    DEFAULT_MTC_PATH,
    DEFAULT_CTL_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_LS_PATH,
};

static cfg_buffer_t bufDefault[] = {
    DEFAULT_MTC_BUF,
    DEFAULT_CTL_BUF,
    DEFAULT_LOG_BUF,
    DEFAULT_LS_BUF,
};


///////////////////////////////////
// Constructors Destructors
///////////////////////////////////
config_t *
cfgCreateDefault()
{ 
    config_t *c = scope_calloc(1, sizeof(config_t));
    if (!c) {
        DBG(NULL);
        return NULL;
    }
    c->mtc.enable = DEFAULT_MTC_ENABLE;
    c->mtc.format = DEFAULT_MTC_FORMAT;
    c->mtc.statsd.prefix = (DEFAULT_STATSD_PREFIX) ? scope_strdup(DEFAULT_STATSD_PREFIX) : NULL;
    c->mtc.statsd.maxlen = DEFAULT_STATSD_MAX_LEN;
    c->mtc.statsd.enable = DEFAULT_MTC_STATSD_ENABLE;
    c->mtc.period = DEFAULT_SUMMARY_PERIOD;
    c->mtc.verbosity = DEFAULT_MTC_VERBOSITY;
    SCOPE_BIT_SET_VAR(c->mtc.categories, CFG_MTC_FS, DEFAULT_MTC_FS_ENABLE);
    SCOPE_BIT_SET_VAR(c->mtc.categories, CFG_MTC_NET, DEFAULT_MTC_NET_ENABLE);
    SCOPE_BIT_SET_VAR(c->mtc.categories, CFG_MTC_HTTP, DEFAULT_MTC_HTTP_ENABLE);
    SCOPE_BIT_SET_VAR(c->mtc.categories, CFG_MTC_DNS, DEFAULT_MTC_DNS_ENABLE);
    SCOPE_BIT_SET_VAR(c->mtc.categories, CFG_MTC_PROC, DEFAULT_MTC_PROC_ENABLE);
    c->evt.enable = DEFAULT_EVT_ENABLE;
    c->evt.format = DEFAULT_CTL_FORMAT;
    c->evt.ratelimit = DEFAULT_MAXEVENTSPERSEC;

    watch_t src;
    for (src=CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        const char *val_def = valueFilterDefault[src];
        c->evt.valuefilter[src] = (val_def) ? scope_strdup(val_def) : NULL;
        const char *field_def = fieldFilterDefault[src];
        c->evt.fieldfilter[src] = (field_def) ? scope_strdup(field_def) : NULL;
        const char *name_def = nameFilterDefault[src];
        c->evt.namefilter[src] = (name_def) ? scope_strdup(name_def) : NULL;
        c->evt.src[src] = srcEnabledDefault[src];
    }

    c->evt.hextract = DEFAULT_SRC_HTTP_HEADER;
    c->evt.numHeaders = 0;
    c->evt.allowbinaryconsole = DEFAULT_ALLOW_BINARY_CONSOLE;

    which_transport_t tp;
    for (tp=CFG_MTC; tp<CFG_WHICH_MAX; tp++) {
        c->transport[tp].type = typeDefault[tp];
        const char* host_def = hostDefault[tp];
        c->transport[tp].net.host = (host_def) ? scope_strdup(host_def) : NULL;
        const char* port_def = portDefault[tp];
        c->transport[tp].net.port = (port_def) ? scope_strdup(port_def) : NULL;
        const char* path_def = pathDefault[tp];
        c->transport[tp].file.path = (path_def) ? scope_strdup(path_def) : NULL;
        c->transport[tp].file.buf_policy = bufDefault[tp];
        c->transport[tp].net.tls.enable = DEFAULT_TLS_ENABLE;
        c->transport[tp].net.tls.validateserver = DEFAULT_TLS_VALIDATE_SERVER;
        c->transport[tp].net.tls.cacertpath = (DEFAULT_TLS_CA_CERT) ? scope_strdup(DEFAULT_TLS_CA_CERT) : NULL;
    }

    c->log.level = DEFAULT_LOG_LEVEL;

    c->pay.enable = DEFAULT_PAYLOAD_ENABLE;
    c->pay.dir = (DEFAULT_PAYLOAD_DIR) ? scope_strdup(DEFAULT_PAYLOAD_DIR) : NULL;

    c->tags = DEFAULT_CUSTOM_TAGS;
    c->max_tags = DEFAULT_NUM_TAGS;

    c->commanddir = (DEFAULT_COMMAND_DIR) ? scope_strdup(DEFAULT_COMMAND_DIR) : NULL;
    c->processstartmsg = DEFAULT_PROCESS_START_MSG;
    c->enhancefs = DEFAULT_ENHANCE_FS;

    c->logstream.enable = DEFAULT_LOGSTREAM_ENABLE;
    c->logstream.cloud = DEFAULT_LOGSTREAM_CLOUD;

    return c;
}

void
cfgDestroy(config_t **cfg)
{
    if (!cfg || !*cfg) return;
    config_t *c = *cfg;
    if (c->mtc.statsd.prefix) scope_free(c->mtc.statsd.prefix);
    if (c->commanddir) scope_free(c->commanddir);

    watch_t src;
    for (src = CFG_SRC_FILE; src<CFG_SRC_MAX; src++) {
        if (c->evt.valuefilter[src]) scope_free(c->evt.valuefilter[src]);
        if (c->evt.fieldfilter[src]) scope_free(c->evt.fieldfilter[src]);
        if (c->evt.namefilter[src]) scope_free(c->evt.namefilter[src]);
    }

    int i;
    for (i = 0; i < c->evt.numHeaders; i++) {
        if (c->evt.hextract && c->evt.hextract[i]) {
            c->evt.hextract[i]->valid = FALSE;
            if (c->evt.hextract[i]->filter) scope_free(c->evt.hextract[i]->filter);
            regfree(&c->evt.hextract[i]->re);
            scope_free(c->evt.hextract[i]);
        }
    }

    if (c->evt.hextract) scope_free(c->evt.hextract);

    which_transport_t t;
    for (t=CFG_MTC; t<CFG_WHICH_MAX; t++) {
        if (c->transport[t].net.host) scope_free(c->transport[t].net.host);
        if (c->transport[t].net.port) scope_free(c->transport[t].net.port);
        if (c->transport[t].file.path) scope_free(c->transport[t].file.path);
    }

    if (c->pay.dir) scope_free(c->pay.dir);

    if (c->tags) {
        int i = 0;
        while (c->tags[i]) {
            scope_free(c->tags[i]->name);
            scope_free(c->tags[i]->value);
            scope_free(c->tags[i]);
            i++;
        }
        scope_free(c->tags);
    }
    scope_free(c);
    *cfg = NULL;
}

///////////////////////////////////
// Accessors
///////////////////////////////////
unsigned
cfgMtcEnable(config_t* cfg)
{
    return (cfg) ? cfg->mtc.enable : DEFAULT_MTC_ENABLE;
}

unsigned
cfgMtcWatchEnable(config_t* cfg, metric_watch_t category) {
    switch(category){
        case CFG_MTC_FS:
            return (cfg) ? SCOPE_BIT_CHECK(cfg->mtc.categories, CFG_MTC_FS) : DEFAULT_MTC_FS_ENABLE;
        case CFG_MTC_NET:
            return (cfg) ? SCOPE_BIT_CHECK(cfg->mtc.categories, CFG_MTC_NET) : DEFAULT_MTC_NET_ENABLE;
        case CFG_MTC_HTTP:
            return (cfg) ? SCOPE_BIT_CHECK(cfg->mtc.categories, CFG_MTC_HTTP) : DEFAULT_MTC_HTTP_ENABLE;
        case CFG_MTC_DNS:
            return (cfg) ? SCOPE_BIT_CHECK(cfg->mtc.categories, CFG_MTC_DNS) : DEFAULT_MTC_DNS_ENABLE;
        case CFG_MTC_PROC:
            return (cfg) ? SCOPE_BIT_CHECK(cfg->mtc.categories, CFG_MTC_PROC) : DEFAULT_MTC_PROC_ENABLE;
        case CFG_MTC_STATSD:
            return (cfg) ? cfg->mtc.statsd.enable : DEFAULT_MTC_STATSD_ENABLE;
        default:
            DBG(NULL);
            return 0;
    }
}

cfg_mtc_format_t
cfgMtcFormat(config_t* cfg)
{
    return (cfg) ? cfg->mtc.format : DEFAULT_MTC_FORMAT;
}

const char*
cfgMtcStatsDPrefix(config_t* cfg)
{
    return (cfg && cfg->mtc.statsd.prefix) ? cfg->mtc.statsd.prefix : DEFAULT_STATSD_PREFIX;
}

unsigned
cfgMtcStatsDMaxLen(config_t* cfg)
{
    return (cfg) ? cfg->mtc.statsd.maxlen : DEFAULT_STATSD_MAX_LEN;
}

unsigned
cfgMtcPeriod(config_t* cfg)
{
    return (cfg) ? cfg->mtc.period : DEFAULT_SUMMARY_PERIOD;
}


const char *
cfgCmdDir(config_t* cfg)
{
    return (cfg) ? cfg->commanddir : DEFAULT_COMMAND_DIR;
}

unsigned
cfgSendProcessStartMsg(config_t* cfg)
{
    return (cfg) ? cfg->processstartmsg : DEFAULT_PROCESS_START_MSG;
}

unsigned
cfgEvtEnable(config_t* cfg)
{
    return (cfg) ? cfg->evt.enable : DEFAULT_EVT_ENABLE;
}

cfg_mtc_format_t
cfgEventFormat(config_t* cfg)
{
    return (cfg) ? cfg->evt.format : DEFAULT_CTL_FORMAT;
}

unsigned
cfgEvtRateLimit(config_t* cfg)
{
    return (cfg) ? cfg->evt.ratelimit : DEFAULT_MAXEVENTSPERSEC;
}

unsigned
cfgEnhanceFs(config_t* cfg)
{
    return (cfg) ? cfg->enhancefs : DEFAULT_ENHANCE_FS;
}

const char*
cfgEvtFormatValueFilter(config_t* cfg, watch_t src)
{
    if (src >= 0 && src < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.valuefilter[src];
        return valueFilterDefault[src];
    }

    DBG("%d", src);
    return valueFilterDefault[CFG_SRC_FILE];
}

const char*
cfgEvtFormatFieldFilter(config_t* cfg, watch_t src)
{
    if (src >= 0 && src < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.fieldfilter[src];
        return fieldFilterDefault[src];
    }

    DBG("%d", src);
    return fieldFilterDefault[CFG_SRC_FILE];
}

const char*
cfgEvtFormatNameFilter(config_t* cfg, watch_t src)
{
    if (src >= 0 && src < CFG_SRC_MAX) {
        if (cfg) return cfg->evt.namefilter[src];
        return nameFilterDefault[src];
    }

    DBG("%d", src);
    return nameFilterDefault[CFG_SRC_FILE];
}

size_t
cfgEvtFormatNumHeaders(config_t *cfg)
{
    if (cfg) return cfg->evt.numHeaders;

    return 0;
}

const char *
cfgEvtFormatHeader(config_t *cfg, int num)
{
    if (cfg && (num < cfg->evt.numHeaders)) {
        if (cfg->evt.hextract && cfg->evt.hextract[num] &&
            (cfg->evt.hextract[num]->valid == TRUE)) {
            return cfg->evt.hextract[num]->filter;
        }
    }

    return DEFAULT_SRC_HTTP_HEADER;
}

regex_t *
cfgEvtFormatHeaderRe(config_t *cfg, int num)
{
    if (cfg && (num < cfg->evt.numHeaders)) {
        if (cfg->evt.hextract && cfg->evt.hextract[num] &&
            (cfg->evt.hextract[num]->valid == TRUE)) {
            return &cfg->evt.hextract[num]->re;
        }
    }

    return NULL;
}

unsigned
cfgEvtFormatSourceEnabled(config_t *cfg, watch_t src)
{
    if ((src >= 0) && (src < CFG_SRC_MAX)) {
        if (cfg) return cfg->evt.src[src];
        return srcEnabledDefault[src];
    }

    DBG("%d", src);
    return srcEnabledDefault[CFG_SRC_FILE];
}

unsigned
cfgEvtAllowBinaryConsole(config_t *cfg)
{
    if (cfg) return cfg->evt.allowbinaryconsole;

    return DEFAULT_ALLOW_BINARY_CONSOLE;
}

unsigned
cfgMtcVerbosity(config_t* cfg)
{
    return (cfg) ? cfg->mtc.verbosity : DEFAULT_MTC_VERBOSITY;
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

unsigned
cfgTransportTlsEnable(config_t *cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].net.tls.enable;
        return DEFAULT_TLS_ENABLE;
    }
    DBG("%d", t);
    return DEFAULT_TLS_ENABLE;
}

unsigned
cfgTransportTlsValidateServer(config_t *cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].net.tls.validateserver;
        return DEFAULT_TLS_VALIDATE_SERVER;
    }
    DBG("%d", t);
    return DEFAULT_TLS_VALIDATE_SERVER;
}

const char *
cfgTransportTlsCACertPath(config_t *cfg, which_transport_t t)
{
    if (t >= 0 && t < CFG_WHICH_MAX) {
        if (cfg) return cfg->transport[t].net.tls.cacertpath;
        return DEFAULT_TLS_CA_CERT;
    }
    DBG("%d", t);
    return DEFAULT_TLS_CA_CERT;
}

custom_tag_t**
cfgCustomTags(config_t* cfg)
{
    return (cfg) ? cfg->tags : DEFAULT_CUSTOM_TAGS;
}

static custom_tag_t*
cfgCustomTag(config_t* cfg, const char* tagName)
{
    if (!tagName) return NULL;
    if (!cfg || !cfg->tags) return NULL;

    int i = 0;
    while (cfg->tags[i]) {
        if (!scope_strcmp(tagName, cfg->tags[i]->name)) {
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

unsigned int
cfgPayEnable(config_t *cfg)
{
    return (cfg) ? cfg->pay.enable : DEFAULT_PAYLOAD_ENABLE;
}

const char *
cfgPayDir(config_t *cfg)
{
    return (cfg) ? cfg->pay.dir : DEFAULT_PAYLOAD_DIR;
}

unsigned
cfgLogStreamEnable(config_t *cfg)
{
    return (cfg) ? cfg->logstream.enable : DEFAULT_LOGSTREAM_ENABLE;
}

unsigned
cfgLogStreamCloud(config_t *cfg)
{
    return (cfg) ? cfg->logstream.cloud : DEFAULT_LOGSTREAM_CLOUD;
}

const char *
cfgAuthToken(config_t * cfg)
{
    return (cfg) ? cfg->authtoken : NULL;
}

///////////////////////////////////
// Setters 
///////////////////////////////////
void
cfgMtcEnableSet(config_t* cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->mtc.enable = val;
}

void
cfgMtcFormatSet(config_t* cfg, cfg_mtc_format_t fmt)
{
    if (!cfg || fmt < 0 || fmt >= CFG_FORMAT_MAX) return;
    cfg->mtc.format = fmt;
}

void
cfgMtcStatsDPrefixSet(config_t* cfg, const char* prefix)
{
    if (!cfg) return;
    if (cfg->mtc.statsd.prefix) scope_free(cfg->mtc.statsd.prefix);
    if (!prefix || (prefix[0] == '\0')) {
        cfg->mtc.statsd.prefix = scope_strdup(DEFAULT_STATSD_PREFIX);
        return;
    }

    // Make sure that the prefix always ends in a '.'
    int n = scope_strlen(prefix);
    if (prefix[n-1] != '.') {
        char* temp = scope_malloc(n+2);
        if (temp) {
            scope_strcpy(temp, prefix);
            temp[n] = '.';
            temp[n+1] = '\0';
        } else {
            DBG("%s", prefix);
        }
        cfg->mtc.statsd.prefix = temp;
    } else {
        cfg->mtc.statsd.prefix = scope_strdup(prefix);
    }
}

void
cfgMtcStatsDMaxLenSet(config_t* cfg, unsigned len)
{
    if (!cfg) return;
    cfg->mtc.statsd.maxlen = len;
}

void
cfgMtcPeriodSet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    cfg->mtc.period = val;
}

void
cfgMtcWatchEnableSet(config_t *cfg, unsigned val, metric_watch_t type)
{
    if (!cfg || val > 1) return;

    switch (type) {
        case CFG_MTC_FS:
        case CFG_MTC_NET:
        case CFG_MTC_HTTP:
        case CFG_MTC_DNS:
        case CFG_MTC_PROC:
            SCOPE_BIT_SET_VAR(cfg->mtc.categories, type, val);
            break;
        case CFG_MTC_STATSD:
            cfg->mtc.statsd.enable = val;
            break;
        default:
            DBG(NULL);
            break;
    }
}

void
cfgCmdDirSet(config_t* cfg, const char* path)
{
    if (!cfg) return;
    if (cfg->commanddir) scope_free(cfg->commanddir);
    if (!path || (path[0] == '\0')) {
        cfg->commanddir = (DEFAULT_COMMAND_DIR) ? scope_strdup(DEFAULT_COMMAND_DIR) : NULL;
        return;
    }

    cfg->commanddir = scope_strdup(path);
}

void
cfgSendProcessStartMsgSet(config_t* cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->processstartmsg = val;
}

void
cfgMtcVerbositySet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    if (val > CFG_MAX_VERBOSITY) val = CFG_MAX_VERBOSITY;
    cfg->mtc.verbosity = val;
}

void
cfgEvtEnableSet(config_t* cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->evt.enable = val;
}

void
cfgEventFormatSet(config_t* cfg, cfg_mtc_format_t fmt)
{
    if (!cfg || fmt < 0 || fmt >= CFG_FORMAT_MAX) return;
    cfg->evt.format = fmt;
}

void
cfgEvtRateLimitSet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    cfg->evt.ratelimit = val;
}

void
cfgEnhanceFsSet(config_t* cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->enhancefs = val;
}

void
cfgEvtFormatValueFilterSet(config_t* cfg, watch_t src, const char* filter)
{
    if (!cfg || src < 0 || src >= CFG_SRC_MAX) return;
    if (cfg->evt.valuefilter[src]) scope_free (cfg->evt.valuefilter[src]);
    if (!filter || (filter[0] == '\0')) {
        const char* vdefault = valueFilterDefault[src];
        cfg->evt.valuefilter[src] = (vdefault) ? scope_strdup(vdefault) : NULL;
        return;
    }
    cfg->evt.valuefilter[src] = scope_strdup(filter);
}

void
cfgEvtFormatFieldFilterSet(config_t* cfg, watch_t src, const char* filter)
{
    if (!cfg || src < 0 || src >= CFG_SRC_MAX) return;
    if (cfg->evt.fieldfilter[src]) scope_free (cfg->evt.fieldfilter[src]);
    if (!filter || (filter[0] == '\0')) {
        const char* fdefault = fieldFilterDefault[src];
        cfg->evt.fieldfilter[src] = (fdefault) ? scope_strdup(fdefault) : NULL;
        return;
    }
    cfg->evt.fieldfilter[src] = scope_strdup(filter);
}

void
cfgEvtFormatNameFilterSet(config_t* cfg, watch_t src, const char* filter)
{
    if (!cfg || src < 0 || src >= CFG_SRC_MAX) return;
    if (cfg->evt.namefilter[src]) scope_free (cfg->evt.namefilter[src]);
    if (!filter || (filter[0] == '\0')) {
        const char* ndefault = nameFilterDefault[src];
        cfg->evt.namefilter[src] = (ndefault) ? scope_strdup(ndefault) : NULL;
        return;
    }
    cfg->evt.namefilter[src] = scope_strdup(filter);
}

void
cfgEvtFormatHeaderSet(config_t *cfg, const char *filter)
{
    if (!cfg || !filter) return;

    size_t headnum = cfg->evt.numHeaders + 1;

    header_extract_t **tempex = scope_realloc(cfg->evt.hextract, headnum * sizeof(header_extract_t *));
    if (!tempex) {
        DBG(NULL);
        return;
    }

    cfg->evt.hextract = tempex;

    header_extract_t *hextract = scope_calloc(1, sizeof(header_extract_t));
    if (hextract) {
        if (!regcomp(&hextract->re, filter, REG_EXTENDED | REG_NOSUB)) {
            hextract->filter = scope_strdup(filter);
            hextract->valid = TRUE;
        } else {
            hextract->valid = FALSE;
            hextract->filter = NULL;
        }
    }

    cfg->evt.hextract[cfg->evt.numHeaders] = hextract;
    cfg->evt.numHeaders = headnum;
}

void
cfgEvtFormatSourceEnabledSet(config_t* cfg, watch_t src, unsigned val)
{
    if (!cfg || src < 0 || src >= CFG_SRC_MAX || val > 1) return;
    cfg->evt.src[src] = val;
}

void
cfgEvtAllowBinaryConsoleSet(config_t *cfg, unsigned val)
{
    if (!cfg || val < 0 || val > 1) return;
    cfg->evt.allowbinaryconsole = val;
}

void
cfgTransportTypeSet(config_t* cfg, which_transport_t t, cfg_transport_t type)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (type < 0 || type > CFG_EDGE) return;
    cfg->transport[t].type = type;
}

void
cfgTransportHostSet(config_t* cfg, which_transport_t t, const char* host)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].net.host) scope_free(cfg->transport[t].net.host);
    cfg->transport[t].net.host = (host) ? scope_strdup(host) : NULL;

}

void
cfgTransportPortSet(config_t* cfg, which_transport_t t, const char* port)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].net.port) scope_free(cfg->transport[t].net.port);
    cfg->transport[t].net.port = (port) ? scope_strdup(port) : NULL;
}

void
cfgTransportPathSet(config_t* cfg, which_transport_t t, const char* path)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].file.path) scope_free(cfg->transport[t].file.path);
    cfg->transport[t].file.path = (path) ? scope_strdup(path) : NULL;
}

void
cfgTransportBufSet(config_t* cfg, which_transport_t t, cfg_buffer_t buf)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (buf < CFG_BUFFER_FULLY || buf > CFG_BUFFER_LINE) return;
    cfg->transport[t].file.buf_policy = buf;
}

void
cfgTransportTlsEnableSet(config_t *cfg, which_transport_t t, unsigned val)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX || val > 1) return;
    cfg->transport[t].net.tls.enable = val;
}

void
cfgTransportTlsValidateServerSet(config_t *cfg, which_transport_t t, unsigned val)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX || val > 1) return;
    cfg->transport[t].net.tls.validateserver = val;
}

void
cfgTransportTlsCACertPathSet(config_t *cfg, which_transport_t t, const char *path)
{
    if (!cfg || t < 0 || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].net.tls.cacertpath) scope_free(cfg->transport[t].net.tls.cacertpath);
    if (!path || (path[0] == '\0')) {
        cfg->transport[t].net.tls.cacertpath = NULL;
        return;
    }
    cfg->transport[t].net.tls.cacertpath = scope_strdup(path);
}

void
cfgCustomTagAdd(config_t* c, const char* name, const char* value)
{
    if (!c || !name || !value) return;

    // If name is already there, replace value only.
    {
        custom_tag_t* t;
        if ((t=cfgCustomTag(c, name))) {
            char* newvalue = scope_strdup(value);
            if (newvalue) {
                scope_free(t->value);
                t->value = newvalue;
                return;
            }
        }
    }

    // Create space if it's the first add
    if (!c->tags) {
        c->tags = scope_calloc(1, sizeof(custom_tag_t*) * c->max_tags);
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
        custom_tag_t** temp = scope_realloc(c->tags, sizeof(custom_tag_t*) * tmp_max_tags);
        if (!temp) {
            DBG("%s %s", name, value);
            return;
        }
        // Yeah!  We have more space!  init it, and set our state to remember it
        scope_memset(&temp[c->max_tags], 0, sizeof(custom_tag_t*) * (tmp_max_tags - c->max_tags));
        c->tags = temp;
        c->max_tags = tmp_max_tags;
    }

    // save it
    {
        custom_tag_t* t = scope_calloc(1,sizeof(custom_tag_t));
        char* n = scope_strdup(name);
        char* v = scope_strdup(value);
        if (!t || !n || !v) {
            DBG("t=%p n=%p v=%p", t, n, v);
            if (t) scope_free(t);
            if (n) scope_free(n);
            if (v) scope_free(v);
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

void
cfgPayEnableSet(config_t *cfg, unsigned int val)
{
    if (!cfg || val > 1) return;
    cfg->pay.enable = val;
}

void
cfgPayDirSet(config_t *cfg, const char *dir)
{
    if (!cfg) return;
    if (cfg->pay.dir) scope_free(cfg->pay.dir);
    if (!dir || (dir[0] == '\0')) {
        cfg->pay.dir = (DEFAULT_PAYLOAD_DIR) ? scope_strdup(DEFAULT_PAYLOAD_DIR) : NULL;
        return;
    }

    cfg->pay.dir = scope_strdup(dir);
}

void
cfgLogStreamEnableSet(config_t *cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->logstream.enable = val;
}

void
cfgLogStreamCloudSet(config_t *cfg, unsigned val)
{
    if (!cfg || val > 1) return;
    cfg->logstream.cloud = val;
}

void
cfgAuthTokenSet(config_t * cfg, const char * authtoken)
{
    if (!cfg) return;
    if (cfg->authtoken) scope_free(cfg->authtoken);
    if (!authtoken || (authtoken[0] == '\0')) {
        cfg->authtoken = NULL;
        return;
    }

    cfg->authtoken = scope_strdup(authtoken);
}
