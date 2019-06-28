#include "scopecfg.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef NO_YAML
#include "yaml.h"
#endif

typedef char** filters_t;

struct _config_t
{
    struct {
        cfg_out_format_t format;
        char* statsd_prefix;  // For format = CFG_EXPANDED_STATSD
        cfg_out_tx_t type;
        struct {              // For type = CFG_UDP
            char* host;
            int port;
        } udp;
        char* path;           // For type = CFG_UNIX or CFG_FILE
    } out;

    filters_t filters;
    unsigned max_filters;
    int logging;
    cfg_log_level_t level;
};

#define DEFAULT_FORMAT CFG_EXPANDED_STATSD
#define DEFAULT_STATSD_PREFIX NULL
#define DEFAULT_TYPE CFG_UDP
#define DEFAULT_HOST "localhost"
#define DEFAULT_PORT 8125
#define DEFAULT_PATH NULL
#define DEFAULT_FILTERS NULL
#define DEFAULT_NUM_FILTERS 8
#define DEFAULT_LOGGING 0
#define DEFAULT_LEVEL CFG_LOG_NONE

    
///////////////////////////////////
// Constructors Destructors
///////////////////////////////////
config_t*
cfgCreateDefault()
{ 
    config_t* c = calloc(1, sizeof(config_t));
    if (!c) return NULL;
    c->out.format = DEFAULT_FORMAT;
    c->out.statsd_prefix = DEFAULT_STATSD_PREFIX;
    c->out.type = DEFAULT_TYPE;
    c->out.udp.host = strdup(DEFAULT_HOST);
    c->out.udp.port = DEFAULT_PORT;
    c->out.path = DEFAULT_PATH;
    c->filters = DEFAULT_FILTERS;
    c->max_filters = DEFAULT_NUM_FILTERS;
    c->logging = DEFAULT_LOGGING;
    c->level = DEFAULT_LEVEL;

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
    }
        
}

static void
processTransports(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SEQUENCE_NODE) return;

    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* i = yaml_document_get_node(doc, *item);
        if (i->type != YAML_SCALAR_NODE) continue;

        char* str = (char*)i->data.scalar.value;
        if (!strcmp(str, "udp")) {
            cfgLogTransportEnabledSet(config, CFG_LOG_UDP, 1);
        } else if (!strcmp(str, "syslog")) {
            cfgLogTransportEnabledSet(config, CFG_LOG_SYSLOG, 1);
        } else if (!strcmp(str, "shm")) {
            cfgLogTransportEnabledSet(config, CFG_LOG_SHM, 1);
        }
    }
}

static void
processLogging(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "level",      processLevel},
        {YAML_SEQUENCE_NODE,"transports", processTransports},
        {YAML_NO_NODE, NULL, NULL}
    };

    yaml_node_pair_t* pair;
    foreach(pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}


static void
processFilters(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SEQUENCE_NODE) return;

    yaml_node_item_t* item;
    foreach(item, node->data.sequence.items) {
        yaml_node_t* i = yaml_document_get_node(doc, *item);
        if (i->type != YAML_SCALAR_NODE) continue;

        cfgFuncFiltersAdd(config, (char*)i->data.scalar.value);
    }
}

static void
processFormat(config_t* config, yaml_document_t* doc, yaml_node_t* node)
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
processStatsdPrefix(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    cfgOutStatsDPrefixSet(config, v_str);
}

static void
processType(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    if (!strcmp(v_str, "udp")) {
        cfgOutTransportTypeSet(config, CFG_UDP);
    } else if (!strcmp(v_str, "unix")) {
        cfgOutTransportTypeSet(config, CFG_UNIX);
    } else if (!strcmp(v_str, "file")) {
        cfgOutTransportTypeSet(config, CFG_FILE);
    } else if (!strcmp(v_str, "syslog")) {
        cfgOutTransportTypeSet(config, CFG_SYSLOG);
    }
}

static void
processHost(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    cfgOutTransportHostSet(config, v_str);
}

static void
processPort(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    
    errno = 0;
    long x = strtol(v_str, NULL, 10);
    if (!errno) {
        cfgOutTransportPortSet(config, x);
    }
}

static void
processPath(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_SCALAR_NODE) return;
    char* v_str = (char *)node->data.scalar.value;
    cfgOutTransportPathSet(config, v_str);
}

static void
processTransport(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "type",       processType},
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
processOutput(config_t* config, yaml_document_t* doc, yaml_node_t* node)
{
    if (node->type != YAML_MAPPING_NODE) return;

    parse_table_t t[] = {
        {YAML_SCALAR_NODE,  "format",          processFormat},
        {YAML_SCALAR_NODE,  "statsdprefix",    processStatsdPrefix},
        {YAML_MAPPING_NODE, "transport",       processTransport},
        {YAML_NO_NODE, NULL, NULL}
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
        {YAML_MAPPING_NODE, "output", processOutput},
        {YAML_SEQUENCE_NODE, "filteredFunctions", processFilters},
        {YAML_MAPPING_NODE, "logging", processLogging},
        {YAML_NO_NODE, NULL, NULL}
    };

    yaml_node_pair_t* pair;
    foreach (pair, node->data.mapping.pairs) {
        processKeyValuePair(t, pair, config, doc);
    }
}

config_t*
cfgRead(char* path)
{
    FILE* f = NULL;
    config_t* config = NULL;
    int parser_successful = 0;
    int doc_successful = 0;
    yaml_parser_t parser;
    yaml_document_t doc;

    config = cfgCreateDefault();
    if (!config) goto cleanup;

    f = fopen(path, "rb");
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
cfgRead(char* path)
{
    return cfgCreateDefault();
}
#endif

void
cfgDestroy(config_t** cfg)
{
    if (!cfg || !*cfg) return;
    config_t* c = *cfg;
    if (c->out.statsd_prefix) free(c->out.statsd_prefix);
    if (c->out.udp.host) free(c->out.udp.host);
    if (c->out.path) free(c->out.path);
    if (c->filters) {
        int i = 0;
        while (c->filters[i]) free(c->filters[i++]);
        free(c->filters);
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
    return (cfg) ? cfg->out.format : DEFAULT_FORMAT;
}

char*
cfgOutStatsDPrefix(config_t* cfg)
{
    return (cfg) ? cfg->out.statsd_prefix : DEFAULT_STATSD_PREFIX;
}

cfg_out_tx_t
cfgOutTransportType(config_t* cfg)
{
    return (cfg) ? cfg->out.type : DEFAULT_TYPE;
}

char*
cfgOutTransportHost(config_t* cfg)
{
    return (cfg) ? cfg->out.udp.host : DEFAULT_HOST;
}

int
cfgOutTransportPort(config_t* cfg)
{
    return (cfg) ? cfg->out.udp.port : DEFAULT_PORT;
}

char*
cfgOutTransportPath(config_t* cfg)
{
    return (cfg) ? cfg->out.path : DEFAULT_PATH;
}

char**
cfgFuncFilters(config_t* cfg)
{
    return (cfg) ? cfg->filters : DEFAULT_FILTERS;
}

int
cfgFuncIsFiltered(config_t* cfg, char* funcName)
{
    if (!funcName) return 0;              // I guess?
    if (!cfg || !cfg->filters) return 0;

    int i = 0;
    while (cfg->filters[i]) {
        if (!strcmp(funcName, cfg->filters[i])) { 
            // funcName appears in the list of filters.
            return 1;
        }
        i++;
    }
    return 0;
}

int
cfgLoggingEnabled(config_t* cfg)
{
    // Yes iff there are one or more transports and the level is not none
    if (cfg) {
        return cfg->logging && (cfg->level != CFG_LOG_NONE);
    }
    return DEFAULT_LOGGING;
}

int
cfgLogTransportEnabled(config_t* cfg, cfg_log_tx_t type)
{
    return ((cfg) ? cfg->logging : DEFAULT_LOGGING) >> type & 1;
}

cfg_log_level_t
cfgLogLevel(config_t* cfg)
{
    return (cfg) ? cfg->level : DEFAULT_LEVEL;
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
cfgOutStatsDPrefixSet(config_t* cfg, char* prefix)
{
    if (!cfg) return;
    if (cfg->out.statsd_prefix) free(cfg->out.statsd_prefix);
    if (!prefix) {
        cfg->out.statsd_prefix = NULL;
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
        cfg->out.statsd_prefix = temp;
    } else {
        cfg->out.statsd_prefix = strdup(prefix);
    }
}

void
cfgOutTransportTypeSet(config_t* cfg, cfg_out_tx_t type)
{
    if (!cfg) return;
    cfg->out.type = type;
}

void
cfgOutTransportHostSet(config_t* cfg, char* host)
{
    if (!cfg) return;
    if (cfg->out.udp.host) free(cfg->out.udp.host);
    cfg->out.udp.host = (host) ? strdup(host) : NULL;
}

void
cfgOutTransportPortSet(config_t* cfg, int port)
{
    if (!cfg) return;
    cfg->out.udp.port = port;
}

void
cfgOutTransportPathSet(config_t* cfg, char* path)
{
    if (!cfg) return;
    if (cfg->out.path) free(cfg->out.path);
    cfg->out.path = (path) ? strdup(path) : NULL;
}

void
cfgFuncFiltersAdd(config_t* c, char* funcname)
{
    if (!c || !funcname) return;

    if (cfgFuncIsFiltered(c, funcname)) return; // Already there.

    // Create space if it's the first add
    if (!c->filters) {
        c->filters = calloc(1, sizeof(char*) * c->max_filters);
        if (!c->filters) return;
    }

    // find the first empty spot
    int i;
    for (i=0; i<c->max_filters && c->filters[i]; i++); 

    // If we're out of space, try to get more space
    if (i >= c->max_filters-1) {     // null delimiter is always required
        int tmp_max_filters = c->max_filters * 2;  // double each time
        filters_t temp = realloc(c->filters, sizeof(char*) * tmp_max_filters);
        if (!temp) return;
        // Yeah!  We have more space!  init it, and set our state to remember it
        memset(&temp[c->max_filters], 0, sizeof(char*) * (tmp_max_filters - c->max_filters));
        c->filters = temp;
        c->max_filters = tmp_max_filters;
    }

    // save it
    c->filters[i] = strdup(funcname);
}

void
cfgLogTransportEnabledSet(config_t* cfg, cfg_log_tx_t type, int value)
{
    if (!cfg) return;
    if (value) {
        cfg->logging |= (1 << type);
    } else {
        cfg->logging &= ~(1 << type);
    }
}

void
cfgLogLevelSet(config_t* cfg, cfg_log_level_t level)
{
    if (!cfg) return;
    cfg->level = level;
}
