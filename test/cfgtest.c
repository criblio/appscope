#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "cfg.h"

#include "test.h"

// gcc -g cfgtest.c cfg.c ./libyaml/src/.libs/libyaml.a -o cfgtest && ./cfgtest
// or to remove one dependency (and the ability to read yaml files)
// gcc -g cfgtest.c cfg.c -DNO_YAML -o cfgtest && ./cfgtest

// with cmocka now:
// gcc -Wall -g -o test/macOS/cfgtest -I./src -I./contrib/libyaml/include -Icontrib/cmocka/include test/cfgtest.c src/cfg.c contrib/libyaml/src/.libs/libyaml.a -lcmocka -L contrib/cmocka/build/src/
// DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/macOS/cfgtest
//
// Coverage:
// brew install lcov
// <build with gcc options "-O0 -coverage">
// lcov --capture --directory . --output-file coverage.info
// genhtml coverage.info --output-directory out
// file:///Users/cribl/scope/out/index.html

static void
verifyDefaults(config_t* config)
{
    assert_int_equal       (cfgOutFormat(config), CFG_EXPANDED_STATSD);
    assert_null            (cfgOutStatsDPrefix(config));
    assert_int_equal       (cfgOutTransportType(config), CFG_UDP);
    assert_string_equal    (cfgOutTransportHost(config), "localhost");
    assert_int_equal       (cfgOutTransportPort(config), 8125);
    assert_null            (cfgOutTransportPath(config));
    assert_null            (cfgFuncFilters(config));
    assert_false           (cfgFuncIsFiltered(config, "read"));
    assert_false           (cfgLoggingEnabled(config));
    assert_false           (cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert_false           (cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert_false           (cfgLogTransportEnabled(config, CFG_LOG_SHM));
    assert_int_equal       (cfgLogLevel(config), CFG_LOG_NONE);
}

int
writeFile(char* path, char* text)
{
    FILE* f = fopen(path, "w");
    if (!f)
        fail_msg("Couldn't open file");

    if (!fwrite(text, strlen(text), 1, f))
        fail_msg("Couldn't write file");

    if (fclose(f))
        fail_msg("Couldn't close file");

    return 0;
}

int
deleteFile(char* path)
{
    return unlink(path);
}

static void
cfgCreateDefaultReturnsValidPtr(void** state)
{
    // construction returns non-null ptr
    config_t* config = cfgCreateDefault();
    assert_non_null (config);

    // cleanup
    cfgDestroy(&config);
    assert_null(config);
}

static void
accessorValuesForDefaultConfigAreAsExpected(void** state)
{
    // defaults via accessors of the config object
    config_t* config = cfgCreateDefault();
    verifyDefaults(config);

    // cleanup
    cfgDestroy(&config);
    assert_null(config);
}

static void
accessorsReturnDefaultsWhenConfigIsNull(void** state)
{
    // Implicitly this verifies no crashes when trying to access a null object
    verifyDefaults(NULL);
}

static void
cfgOutFormatSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutFormatSet(config, CFG_NEWLINE_DELIMITED);
    assert_int_equal(cfgOutFormat(config), CFG_NEWLINE_DELIMITED);
    cfgOutFormatSet(config, CFG_EXPANDED_STATSD);
    assert_int_equal(cfgOutFormat(config), CFG_EXPANDED_STATSD);
    cfgDestroy(&config);
}

static void
cfgOutStatsDPrefixSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutStatsDPrefixSet(config, "heywithdot.");
    assert_string_equal(cfgOutStatsDPrefix(config), "heywithdot.");
    cfgOutStatsDPrefixSet(config, "heywithoutdot");
    assert_string_equal(cfgOutStatsDPrefix(config), "heywithoutdot.");
    cfgOutStatsDPrefixSet(config, NULL);
    assert_null(cfgOutStatsDPrefix(config));
    cfgDestroy(&config);
}

static void
cfgOutTransportTypeSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutTransportTypeSet(config, CFG_SYSLOG);
    assert_int_equal(cfgOutTransportType(config), CFG_SYSLOG);
    cfgOutTransportTypeSet(config, CFG_FILE);
    assert_int_equal(cfgOutTransportType(config), CFG_FILE);
    cfgOutTransportTypeSet(config, CFG_UNIX);
    assert_int_equal(cfgOutTransportType(config), CFG_UNIX);
    cfgOutTransportTypeSet(config, CFG_UDP);
    assert_int_equal(cfgOutTransportType(config), CFG_UDP);
    cfgDestroy(&config);
}

static void
cfgOutTransportHostSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutTransportHostSet(config, "larrysComputer");
    assert_string_equal(cfgOutTransportHost(config), "larrysComputer");
    cfgOutTransportHostSet(config, "bobsComputer");
    assert_string_equal(cfgOutTransportHost(config), "bobsComputer");
    cfgOutTransportHostSet(config, NULL);
    assert_null(cfgOutTransportHost(config));
    cfgDestroy(&config);
}

static void
cfgOutTransportPortSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutTransportPortSet(config, 54321);
    assert_int_equal(cfgOutTransportPort(config), 54321);
    cfgOutTransportPortSet(config, 12345);
    assert_int_equal(cfgOutTransportPort(config), 12345);
    cfgDestroy(&config);
}

static void
cfgOutTransportPathSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutTransportPathSet(config, "/tmp/mysock");
    assert_string_equal(cfgOutTransportPath(config), "/tmp/mysock");
    cfgOutTransportPathSet(config, NULL);
    assert_null(cfgOutTransportPath(config));
    cfgDestroy(&config);
}

static void
cfgFuncFiltersSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    assert_null(cfgFuncFilters(config));
    int i;
    for (i=0; i<10; i++) {
        char funcName[64];
        snprintf(funcName, sizeof(funcName), "myfavoritefunc%d", i);

        assert_false(cfgFuncIsFiltered(config, funcName));
        cfgFuncFiltersAdd(config, funcName);
        assert_true(cfgFuncIsFiltered(config, funcName));
        assert_string_equal(cfgFuncFilters(config)[i], funcName);
        assert_null(cfgFuncFilters(config)[i+1]);
    }
    cfgDestroy(&config);
}

static void
cfgLoggingSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    assert_false(cfgLoggingEnabled(config));
    cfgLogTransportEnabledSet(config, CFG_LOG_UDP, 1);
    assert_false(cfgLoggingEnabled(config));
    cfgLogLevelSet(config, CFG_LOG_DEBUG);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    assert_true(cfgLoggingEnabled(config));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert_false(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert_false(cfgLogTransportEnabled(config, CFG_LOG_SHM));

    cfgLogTransportEnabledSet(config, CFG_LOG_UDP, 0);
    assert_false(cfgLoggingEnabled(config));
    assert_false(cfgLogTransportEnabled(config, CFG_LOG_UDP));

    cfgLogTransportEnabledSet(config, CFG_LOG_SYSLOG, 1);
    cfgLogTransportEnabledSet(config, CFG_LOG_SHM, 1);
    assert_true(cfgLoggingEnabled(config));
    assert_false(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SHM));
    cfgDestroy(&config);
}

static void
cfgLogLevelSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgLogLevelSet(config, CFG_LOG_DEBUG);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    cfgLogLevelSet(config, CFG_LOG_INFO);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);
    cfgLogLevelSet(config, CFG_LOG_WARN);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_WARN);
    cfgLogLevelSet(config, CFG_LOG_ERROR);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_ERROR);
    cfgLogLevelSet(config, CFG_LOG_NONE);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_NONE);
    cfgDestroy(&config);
}

static void
cfgReadGoodYaml(void** state)
{
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
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_NEWLINE_DELIMITED);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutTransportType(config), CFG_FILE);
    assert_string_equal(cfgOutTransportHost(config), "localhost");
    assert_int_equal(cfgOutTransportPort(config), 8125);
    assert_string_equal(cfgOutTransportPath(config), "/var/log/scope.log");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_true(cfgFuncIsFiltered(config, "write"));
    assert_true(cfgFuncIsFiltered(config, "accept"));
    assert_true(cfgLoggingEnabled(config));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SHM));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    cfgDestroy(&config);
    deleteFile(path);
}

static void
writeFileWithSubstitution(char* path, char* base, char* variable)
{
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), base, variable);
    if (n < 0)
        fail_msg("Unable to create yaml buffer");
    if (n >= sizeof(buf))
        fail_msg("Inadequate yaml buffer size");

    writeFile(path, buf);
}

static void
cfgReadEveryTransportType(void** state)
{
    char* yamlText =
        "---\n"
        "output:\n"
        "  transport:\n"
        "%s"
        "...\n";

    char* udp_str =
        "    type: udp\n"
        "    host: 'labmachine8235'\n"
        "    port: 235\n";
    char* unix_str =
        "    type: unix\n"
        "    path: '/var/run/scope.sock'\n";
    char* file_str =
        "    type: file\n"
        "    path: '/var/log/scope.log'\n";
    char* syslog_str =
        "    type: syslog\n";
    char* transport_lines[] = {udp_str, unix_str, file_str, syslog_str};

    char* path = "./scope.cfg";

    int i;
    for (i = 0; i<sizeof(transport_lines) / sizeof(transport_lines[0]); i++) {

        writeFileWithSubstitution(path, yamlText, transport_lines[i]);
        config_t* config = cfgRead(path);

        if (transport_lines[i] == udp_str) {
                assert_int_equal(cfgOutTransportType(config), CFG_UDP);
                assert_string_equal(cfgOutTransportHost(config), "labmachine8235");
                assert_int_equal(cfgOutTransportPort(config), 235);
        } else if (transport_lines[i] == unix_str) {
                assert_int_equal(cfgOutTransportType(config), CFG_UNIX);
                assert_string_equal(cfgOutTransportPath(config), "/var/run/scope.sock");
        } else if (transport_lines[i] == file_str) {
                assert_int_equal(cfgOutTransportType(config), CFG_FILE);
                assert_string_equal(cfgOutTransportPath(config), "/var/log/scope.log");
        } else if (transport_lines[i] == syslog_str) {
                assert_int_equal(cfgOutTransportType(config), CFG_SYSLOG);
        }

        deleteFile(path);
        cfgDestroy(&config);
    }

}

static void
cfgReadEveryProcessLevel(void** state)
{
    char* yamlText =
        "---\n"
        "logging:\n"
        "  level: %s\n"
        "...\n";

    char* path = "./scope.cfg";
    char* level[] = {"debug", "info", "warning", "error", "none"};
    cfg_log_level_t value[] = {CFG_LOG_DEBUG, CFG_LOG_INFO, CFG_LOG_WARN, CFG_LOG_ERROR, CFG_LOG_NONE};
    int i;
    for (i = 0; i< sizeof(level)/sizeof(level[0]); i++) {
        writeFileWithSubstitution(path, yamlText, level[i]);
        config_t* config = cfgRead(path);
        assert_int_equal(cfgLogLevel(config), value[i]);
        deleteFile(path);
        cfgDestroy(&config);
    }
}


static void
cfgReadGoodJson(void** state)
{
    // Test file config (json)
    char* jsonText =
        "{\n"
        "  'output': {\n"
        "    'format': 'newlinedelimited',\n"
        "    'statsdprefix': 'cribl.scope',\n"
        "    'transport': {\n"
        "      'type': 'file',\n"
        "      'path': '/var/log/scope.log'\n"
        "    }\n"
        "  },\n"
        "  'filteredFunctions': [\n"
        "    'read',\n"
        "    'write',\n"
        "    'accept'\n"
        "  ],\n"
        "  'logging': {\n"
        "    'level': 'debug',\n"
        "    'transports': [\n"
        "      'udp',\n"
        "      'syslog',\n"
        "      'shm'\n"
        "    ]\n"
        "  }\n"
        "}\n";
    char* path = "./scope.cfg";
    writeFile(path, jsonText);
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_NEWLINE_DELIMITED);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutTransportType(config), CFG_FILE);
    assert_string_equal(cfgOutTransportHost(config), "localhost");
    assert_int_equal(cfgOutTransportPort(config), 8125);
    assert_string_equal(cfgOutTransportPath(config), "/var/log/scope.log");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_true(cfgFuncIsFiltered(config, "write"));
    assert_true(cfgFuncIsFiltered(config, "accept"));
    assert_true(cfgLoggingEnabled(config));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_UDP));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SYSLOG));
    assert_true(cfgLogTransportEnabled(config, CFG_LOG_SHM));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadNonExistentFileReturnsDefaults(void** state)
{
    config_t* config = cfgRead("../thisFileNameWontBeFoundAnywhere.txt");
    verifyDefaults(config);
    cfgDestroy(&config);
}

static void
cfgReadBadYamlReturnsDefaults(void** state)
{
    char* yamlText =
        "---\n"
        "output:\n"
        "  format: newlinedelimited\n"
        "  statsdprefix : 'cribl.scope'\n"
        "  transport:\n"
        "    type: file\n"
        "    path: '/var/log/scope.log'\n"
        "filteredFunctions:\n"
        "  - read\n"
        "logging:\n"
        "      level: debug                  # <--- Extra indention!  bad!\n"
        "  transports:\n"
        "    - syslog\n"
        "...\n";
    char* path = "./scope.cfg";
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    verifyDefaults(config);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadExtraFieldsAreHarmless(void** state)
{
    char* yamlText =
        "---\n"
        "momsApplePieRecipe:                # has possibilities...\n"
        "  [apples,sugar,flour,dirt]        # dirt mom?  Really?\n"
        "output:\n"
        "  format: expandedstatsd\n"
        "  request: 'make it snappy'        # Extra.\n"
        "  transport:\n"
        "    type: unix\n"
        "    path: '/var/run/scope.sock'\n"
        "    color: 'puce'                  # Extra.\n"
        "filteredFunctions:\n"
        "  - read\n"
        "logging:\n"
        "  level: info\n"
        "...\n";
    char* path = "./scope.cfg";
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_EXPANDED_STATSD);
    assert_null(cfgOutStatsDPrefix(config));
    assert_int_equal(cfgOutTransportType(config), CFG_UNIX);
    assert_string_equal(cfgOutTransportPath(config), "/var/run/scope.sock");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_false(cfgLoggingEnabled(config));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadYamlOrderWithinStructureDoesntMatter(void** state)
{
    char* yamlText =
        "---\n"
        "logging:\n"
        "  level: info\n"
        "filteredFunctions:\n"
        "  - read\n"
        "output:\n"
        "  transport:\n"
        "    path: '/var/run/scope.sock'\n"
        "    type: unix\n"
        "  format: expandedstatsd\n"
        "...\n";
    char* path = "./scope.cfg";
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_EXPANDED_STATSD);
    assert_null(cfgOutStatsDPrefix(config));
    assert_int_equal(cfgOutTransportType(config), CFG_UNIX);
    assert_string_equal(cfgOutTransportPath(config), "/var/run/scope.sock");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_false(cfgLoggingEnabled(config));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgCreateDefaultReturnsValidPtr),
        cmocka_unit_test(accessorValuesForDefaultConfigAreAsExpected),
        cmocka_unit_test(accessorsReturnDefaultsWhenConfigIsNull),
        cmocka_unit_test(cfgOutFormatSetAndGet),
        cmocka_unit_test(cfgOutStatsDPrefixSetAndGet),
        cmocka_unit_test(cfgOutTransportTypeSetAndGet),
        cmocka_unit_test(cfgOutTransportHostSetAndGet),
        cmocka_unit_test(cfgOutTransportPortSetAndGet),
        cmocka_unit_test(cfgOutTransportPathSetAndGet),
        cmocka_unit_test(cfgFuncFiltersSetAndGet),
        cmocka_unit_test(cfgLoggingSetAndGet),
        cmocka_unit_test(cfgLogLevelSetAndGet),
        cmocka_unit_test(cfgReadGoodYaml),
        cmocka_unit_test(cfgReadEveryTransportType),
        cmocka_unit_test(cfgReadEveryProcessLevel),
        cmocka_unit_test(cfgReadGoodJson),
        cmocka_unit_test(cfgReadNonExistentFileReturnsDefaults),
        cmocka_unit_test(cfgReadBadYamlReturnsDefaults),
        cmocka_unit_test(cfgReadExtraFieldsAreHarmless),
        cmocka_unit_test(cfgReadYamlOrderWithinStructureDoesntMatter),
    };
    cmocka_run_group_tests(tests, NULL, NULL);

    return 0;
}


