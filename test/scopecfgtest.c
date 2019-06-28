#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "scopecfg.h"

// gcc -g scopecfgtest.c scopecfg.c ./libyaml/src/.libs/libyaml.a -o scopecfgtest && ./scopecfgtest
// or to remove one dependency (and the ability to read yaml files)
// gcc -g scopecfgtest.c scopecfg.c -DNO_YAML -o scopecfgtest && ./scopecfgtest

void
testDefaults(config_t* config)
{
    assert(cfgOutFormat(config) == CFG_EXPANDED_STATSD);
    assert(!cfgOutStatsDPrefix(config));
    assert(cfgOutTransportType(config) == CFG_UDP);
    assert(!strcmp(cfgOutTransportHost(config), "localhost"));
    assert(cfgOutTransportPort(config) == 8125);
    assert(!cfgOutTransportPath(config)); // Path should be NULL
    assert(!cfgFuncFilters(config));
    assert(!cfgFuncIsFiltered(config, "read"));
    assert(!cfgLoggingEnabled(config));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_SHM));
    assert(cfgLogLevel(config) == CFG_LOG_NONE);
}

int
writeFile(char* path, char* text)
{
    FILE* f = fopen(path, "w");
    if (!f) return 1;

    if (!fwrite(text, strlen(text), 1, f)) return 1;

    if (!fclose(f)) return 1;
    return 0;
}

int
deleteFile(char* path)
{
    return unlink(path);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    // Test default config and accessors
    config_t* config = cfgCreateDefault();
    assert(config);
    testDefaults(config);
    cfgDestroy(&config);
    assert(!config);

    // Test accessors return defaults if config is null
    testDefaults(NULL);

    // Test setters - special attention to cfgFuncFiltersAdd and cfgOutStatsDPrefixSet
    config = cfgCreateDefault();
    cfgOutFormatSet(config, CFG_NEWLINE_DELIMITED);
    assert(cfgOutFormat(config) == CFG_NEWLINE_DELIMITED);

    cfgOutStatsDPrefixSet(config, "heywithdot.");
    assert(!strcmp(cfgOutStatsDPrefix(config), "heywithdot."));
    cfgOutStatsDPrefixSet(config, "heywithoutdot");
    assert(!strcmp(cfgOutStatsDPrefix(config), "heywithoutdot."));
    cfgOutStatsDPrefixSet(config, NULL);
    assert(!cfgOutStatsDPrefix(config));

    cfgOutTransportTypeSet(config, CFG_SYSLOG);
    assert(cfgOutTransportType(config) == CFG_SYSLOG);

    cfgOutTransportHostSet(config, "larrysComputer");
    assert(!strcmp(cfgOutTransportHost(config), "larrysComputer"));
    cfgOutTransportHostSet(config, NULL);
    assert(!cfgOutTransportHost(config));

    cfgOutTransportPortSet(config, 54321);
    assert(cfgOutTransportPort(config) == 54321);

    cfgOutTransportPathSet(config, "/tmp/mysock");
    assert(!strcmp(cfgOutTransportPath(config), "/tmp/mysock"));
    cfgOutTransportPathSet(config, NULL);
    assert(!cfgOutTransportPath(config));

    config = cfgCreateDefault();
    assert(!cfgFuncFilters(config));
    int i;
    for (i=0; i<10; i++) {
        char funcName[64];
        snprintf(funcName, sizeof(funcName), "myfavoritefunc%d", i);

        assert(!cfgFuncIsFiltered(config, funcName));
        cfgFuncFiltersAdd(config, funcName);
        assert(cfgFuncIsFiltered(config, funcName));
        assert(!strcmp(cfgFuncFilters(config)[i], funcName));
        assert(!cfgFuncFilters(config)[i+1]);
    }

    assert(!cfgLoggingEnabled(config));
    cfgLogTransportEnabledSet(config, CFG_LOG_UDP, 1);
    assert(!cfgLoggingEnabled(config));
    cfgLogLevelSet(config, CFG_LOG_DEBUG);
    assert(cfgLogLevel(config) == CFG_LOG_DEBUG);
    assert(cfgLoggingEnabled(config));
    assert(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_SHM));
    cfgLogTransportEnabledSet(config, CFG_LOG_UDP, 0);
    assert(!cfgLoggingEnabled(config));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_UDP));
    cfgLogTransportEnabledSet(config, CFG_LOG_SYSLOG, 1);
    cfgLogTransportEnabledSet(config, CFG_LOG_SHM, 1);
    assert(cfgLoggingEnabled(config));
    assert(!cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert(cfgLogTransportEnabled(config, CFG_LOG_SHM));
   
    cfgDestroy(&config);

    // Test file config (yaml)
    char* yamlText = 
        "---\n" 
        "output:\n" 
        "  format: newlinedelimited          # expandedstatsd, newlinedelimited\n" 
        "  statsdprefix : 'cribl.scope'      # prepends each statsd metric\n" 
        "  transport:                        # defines how scope output is sent\n" 
        "    type: file                      # udp, unix, file, syslog\n" 
        "    path: '/var/log/scope.log'\n" 
        "filteredFunctions:                  # keep Scope from processing these\n" 
        "  - read                            # specific intercepted functions\n" 
        "  - write\n" 
        "  - accept\n" 
        "logging:\n"
        "  level: debug                      # debug, info, warning, error, none\n"
        "  transports:\n"
        "    - udp\n" 
        "    - syslog\n" 
        "    - shm\n" 
        "...\n";
    char* path = "./scope.cfg";
    writeFile(path, yamlText);
    config = cfgRead(path);
    assert(config);
    assert(cfgOutFormat(config) == CFG_NEWLINE_DELIMITED);
    assert(!strcmp(cfgOutStatsDPrefix(config), "cribl.scope."));
    assert(cfgOutTransportType(config) == CFG_FILE);
    assert(!strcmp(cfgOutTransportHost(config), "localhost"));
    assert(cfgOutTransportPort(config) == 8125);
    assert(!strcmp(cfgOutTransportPath(config), "/var/log/scope.log")); // Path should be NULL
    assert(cfgFuncFilters(config));
    assert(cfgFuncIsFiltered(config, "read"));
    assert(cfgFuncIsFiltered(config, "write"));
    assert(cfgFuncIsFiltered(config, "accept"));
    assert(cfgLoggingEnabled(config));
    assert(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert(cfgLogTransportEnabled(config, CFG_LOG_SHM));
    cfgDestroy(&config);
    deleteFile(path);

    // Test file config (json)

    // Test that malloc failures aren't fatal  (redundant with tolerating null cfg?)

    // Test unreachable file results in default

    // Test unparseable yaml/json results in default

    // Test extra fields in yaml are harmless

    // Test that order of peer fields doesn't matter

    // Test that size of compiled binary isn't totally absurd

    return 0;
}


