#include "cfg.h"
#define _GNU_SOURCE
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#ifndef NO_YAML
#include "yaml.h"
#endif

typedef struct {
    cfg_transport_t type;
    struct {              // For type = CFG_UDP
        char* host;
        char* port;
    } udp;
    char* path;           // For type = CFG_UNIX or CFG_FILE
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
    } out;

    // CFG_OUT or CFG_LOG
    transport_struct_t transport[CFG_WHICH_MAX]; 
    which_transport_t transport_context; // only used during cfgRead

    custom_tag_t** tags;
    unsigned max_tags;
    cfg_log_level_t level;
};

#define DEFAULT_SUMMARY_PERIOD 10
#define DEFAULT_OUT_TYPE CFG_UDP
#define DEFAULT_OUT_HOST "127.0.0.1"
#define DEFAULT_OUT_PORT "8125"
#define DEFAULT_OUT_PATH NULL
#define DEFAULT_LOG_TYPE CFG_FILE
#define DEFAULT_LOG_HOST NULL
#define DEFAULT_LOG_PORT NULL
#define DEFAULT_LOG_PATH "/tmp/scope.log"
#define DEFAULT_TAGS NULL
#define DEFAULT_NUM_TAGS 8

    
///////////////////////////////////
// Constructors Destructors
///////////////////////////////////
config_t*
cfgCreateDefault()
{ 
    config_t* c = calloc(1, sizeof(config_t));
    if (!c) return NULL;
    c->out.format = DEFAULT_OUT_FORMAT;
    c->out.statsd.prefix = (DEFAULT_STATSD_PREFIX) ? strdup(DEFAULT_STATSD_PREFIX) : NULL;
    c->out.statsd.maxlen = DEFAULT_STATSD_MAX_LEN;
    c->out.period = DEFAULT_SUMMARY_PERIOD;
    c->out.verbosity = DEFAULT_OUT_VERBOSITY;
    c->transport[CFG_OUT].type = DEFAULT_OUT_TYPE;
    c->transport[CFG_OUT].udp.host = (DEFAULT_OUT_HOST) ? strdup(DEFAULT_OUT_HOST) : NULL;
    c->transport[CFG_OUT].udp.port = (DEFAULT_OUT_PORT) ? strdup(DEFAULT_OUT_PORT) : NULL;
    c->transport[CFG_OUT].path = (DEFAULT_OUT_PATH) ? strdup(DEFAULT_OUT_PATH) : NULL;
    c->transport[CFG_LOG].type = DEFAULT_LOG_TYPE;
    c->transport[CFG_LOG].udp.host = (DEFAULT_LOG_HOST) ? strdup(DEFAULT_LOG_HOST) : NULL;
    c->transport[CFG_LOG].udp.port = (DEFAULT_LOG_PORT) ? strdup(DEFAULT_LOG_PORT) : NULL;
    c->transport[CFG_LOG].path = (DEFAULT_LOG_PATH) ? strdup(DEFAULT_LOG_PATH) : NULL;
    c->tags = DEFAULT_TAGS;
    c->max_tags = DEFAULT_NUM_TAGS;
    c->level = DEFAULT_LOG_LEVEL;

    return c;
}

#ifndef NO_YAML

#define foreach(pair, pairs) \
    for (pair = pairs.start; pair != pairs.top; pair++)

typedef void (*node_fn)(config_t*, yaml_document_t*, yaml_node_t*);

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
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    if (!strcmp(v_str, "debug")) {
        cfgLogLevelSet(config, CFG_LOG_DEBUG);
    } else if (!strcmp(v_str, "info")) {
        cfgLogLevelSet(config, CFG_LOG_INFO);
    } else if (!strcmp(v_str, "warning")) {
        cfgLogLevelSet(config, CFG_LOG_WARN);
    } else if (!strcmp(v_str, "error")) {
        cfgLogLevelSet(config, CFG_LOG_ERROR);
    } else if (!strcmp(v_str, "none")) {
        cfgLogLevelSet(config, CFG_LOG_NONE);
    } else if (!strcmp(v_str, "trace")) {
        cfgLogLevelSet(config, CFG_LOG_TRACE);
    }
}

static void
processTransportType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    which_transport_t c = config->transport_context;
    if (!strcmp(v_str, "udp")) {
        cfgTransportTypeSet(config, c, CFG_UDP);
    } else if (!strcmp(v_str, "unix")) {
        cfgTransportTypeSet(config, c, CFG_UNIX);
    } else if (!strcmp(v_str, "file")) {
        cfgTransportTypeSet(config, c, CFG_FILE);
    } else if (!strcmp(v_str, "syslog")) {
        cfgTransportTypeSet(config, c, CFG_SYSLOG);
    } else if (!strcmp(v_str, "shm")) {
        cfgTransportTypeSet(config, c, CFG_SHM);
    }
}

static void
processHost(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    which_transport_t c = config->transport_context;
    cfgTransportHostSet(config, c, v_str);
}

static void
processPort(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    which_transport_t c = config->transport_context;
    cfgTransportPortSet(config, c, v_str);
}

static void
processPath(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    which_transport_t c = config->transport_context;
    cfgTransportPathSet(config, c, v_str);
}

static void
processTransport(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "type",       processTransportType},
        {YAML_SCALAR_NODE,  "host",       processHost},
        {YAML_SCALAR_NODE,  "port",       processPort},
        {YAML_SCALAR_NODE,  "path",       processPath},
        {YAML_NO_NODE, NULL, NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

static void
processLogging(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "level",      processLevel},
        {YAML_MAPPING_NODE, "transport",  processTransport},
        {YAML_NO_NODE, NULL, NULL}
    };

    // Remember that we're currently processing logging
    config->transport_context = CFG_LOG;

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
        if (key->type != YAML_SCALAR_NODE) return;
        if (value->type != YAML_SCALAR_NODE) return;

        const char* key_str = (const char*)key->data.scalar.value;
        const char* value_str = (const char*)value->data.scalar.value;

        cfgCustomTagAdd(config, key_str, value_str);
    }
}

static void
processFormatType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    if (!strcmp(v_str, "expandedstatsd")) {
        cfgOutFormatSet(config, CFG_EXPANDED_STATSD);
    } else if (!strcmp(v_str, "newlinedelimited")) {
        cfgOutFormatSet(config, CFG_NEWLINE_DELIMITED);
    }
}

static void
processStatsDPrefix(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    cfgOutStatsDPrefixSet(config, v_str);
}

static void
processStatsDMaxLen(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(v_str, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutStatsDMaxLenSet(config, x);
}

static void
processVerbosity(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(v_str, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutVerbositySet(config, x);
}

static void
processFormat(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "type",            processFormatType},
        {YAML_SCALAR_NODE,  "statsdprefix",    processStatsDPrefix},
        {YAML_SCALAR_NODE,  "statsdmaxlen",    processStatsDMaxLen},
        {YAML_SCALAR_NODE,  "verbosity",       processVerbosity},
        {YAML_NO_NODE, NULL, NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }

}

static void
processSummaryPeriod(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(v_str, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutPeriodSet(config, x);
}

static void
processOutput(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_MAPPING_NODE, "format",          processFormat},
        {YAML_MAPPING_NODE, "transport",       processTransport},
        {YAML_SCALAR_NODE,  "summaryperiod",   processSummaryPeriod},
        {YAML_SEQUENCE_NODE, "tags",           processTags},
        {YAML_NO_NODE, NULL, NULL}
    };

    // Remember that we're currently processing output
    config->transport_context = CFG_OUT;

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
        {YAML_MAPPING_NODE,  "output",             processOutput},
        {YAML_MAPPING_NODE,  "logging",            processLogging},
        {YAML_NO_NODE, NULL, NULL}
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
    FILE *(*fnopen)(const char *, const char *);

    fnopen = dlsym(RTLD_NEXT, "fopen");
    if (!fnopen) goto cleanup;
    
    config = cfgCreateDefault();
    if (!config) goto cleanup;

    f = fnopen(path, "rb");
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
    if (f) fclose(f);
    return config;
}
#else
config_t*
cfgRead(const char* path)
{
    return cfgCreateDefault();
}
#endif

void
cfgDestroy(config_t** cfg)
{
    if (!cfg || !*cfg) return;
    config_t* c = *cfg;
    if (c->out.statsd.prefix) free(c->out.statsd.prefix);
    which_transport_t t;
    for (t=CFG_OUT; t<CFG_WHICH_MAX; t++) {
        if (c->transport[t].udp.host) free(c->transport[t].udp.host);
        if (c->transport[t].udp.port) free(c->transport[t].udp.port);
        if (c->transport[t].path) free(c->transport[t].path);
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
        case CFG_LOG:
        default:
            return DEFAULT_LOG_TYPE;
    }
}

const char*
cfgTransportHost(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].udp.host;
    } 

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_HOST;
        case CFG_LOG:
        default:
            return DEFAULT_LOG_HOST;
    }
}

const char*
cfgTransportPort(config_t* cfg, which_transport_t t)
{
    if (cfg && t < CFG_WHICH_MAX) {
        return cfg->transport[t].udp.port;
    }

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_PORT;
        case CFG_LOG:
        default:
            return DEFAULT_LOG_PORT;
    }
}

const char*
cfgTransportPath(config_t* cfg, which_transport_t t)
{
    if (cfg && t  < CFG_WHICH_MAX) {
        return cfg->transport[t].path;
    }

    switch (t) {
        case CFG_OUT:
            return DEFAULT_OUT_PATH;
        case CFG_LOG:
        default:
            return DEFAULT_LOG_PATH;
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
    return (cfg) ? cfg->level : DEFAULT_LOG_LEVEL;
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
cfgOutVerbositySet(config_t* cfg, unsigned val)
{
    if (!cfg) return;
    if (val > CFG_MAX_VERBOSITY) val = CFG_MAX_VERBOSITY;
    cfg->out.verbosity = val;
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
    if (cfg->transport[t].udp.host) free(cfg->transport[t].udp.host);
    cfg->transport[t].udp.host = (host) ? strdup(host) : NULL;
}

void
cfgTransportPortSet(config_t* cfg, which_transport_t t, const char* port)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].udp.port) free(cfg->transport[t].udp.port);
    cfg->transport[t].udp.port = (port) ? strdup(port) : NULL;
}

void
cfgTransportPathSet(config_t* cfg, which_transport_t t, const char* path)
{
    if (!cfg || t >= CFG_WHICH_MAX) return;
    if (cfg->transport[t].path) free(cfg->transport[t].path);
    cfg->transport[t].path = (path) ? strdup(path) : NULL;
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
        if (!c->tags) return;
    }

    // find the first empty spot
    int i;
    for (i=0; i<c->max_tags && c->tags[i]; i++); 

    // If we're out of space, try to get more space
    if (i >= c->max_tags-1) {     // null delimiter is always required
        int tmp_max_tags = c->max_tags * 2;  // double each time
        custom_tag_t** temp = realloc(c->tags, sizeof(custom_tag_t*) * tmp_max_tags);
        if (!temp) return;
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
    cfg->level = level;
}
