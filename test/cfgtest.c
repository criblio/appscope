#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "cfg.h"

#include "test.h"

static void
verifyDefaults(config_t* config)
{
    assert_int_equal       (cfgOutFormat(config), DEFAULT_OUT_FORMAT);
    assert_string_equal    (cfgOutStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal       (cfgOutStatsDMaxLen(config), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal       (cfgOutVerbosity(config), DEFAULT_OUT_VERBOSITY);
    assert_int_equal       (cfgOutPeriod(config), DEFAULT_SUMMARY_PERIOD);
    assert_int_equal       (cfgTransportType(config, CFG_OUT), CFG_UDP);
    assert_string_equal    (cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_int_equal       (cfgTransportPort(config, CFG_OUT), 8125);
    assert_null            (cfgTransportPath(config, CFG_OUT));
    assert_int_equal       (cfgTransportType(config, CFG_LOG), CFG_FILE);
    assert_string_equal    (cfgTransportHost(config, CFG_LOG), "127.0.0.1");
    assert_int_equal       (cfgTransportPort(config, CFG_LOG), 8125);
    assert_string_equal    (cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_null            (cfgFuncFilters(config));
    assert_false           (cfgFuncIsFiltered(config, "read"));
    assert_int_equal       (cfgLogLevel(config), DEFAULT_LOG_LEVEL);
}

int
writeFile(const char* path, const char* text)
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
deleteFile(const char* path)
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
    assert_string_equal(cfgOutStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    cfgOutStatsDPrefixSet(config, "");
    assert_string_equal(cfgOutStatsDPrefix(config), "");
    cfgDestroy(&config);
}

static void
cfgOutStatsDMaxLenSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutStatsDMaxLenSet(config, 0);
    assert_int_equal(cfgOutStatsDMaxLen(config), 0);
    cfgOutStatsDMaxLenSet(config, UINT_MAX);
    assert_int_equal(cfgOutStatsDMaxLen(config), UINT_MAX);
    cfgDestroy(&config);
}

static void
cfgOutVerbositySetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    int i;
    for (i=0; i<=CFG_MAX_VERBOSITY+1; i++) {
        cfgOutVerbositySet(config, i);
        if (i > CFG_MAX_VERBOSITY)
            assert_int_equal(cfgOutVerbosity(config), CFG_MAX_VERBOSITY);
        else
            assert_int_equal(cfgOutVerbosity(config), i);
    }
    cfgOutVerbositySet(config, UINT_MAX);
    assert_int_equal(cfgOutVerbosity(config), CFG_MAX_VERBOSITY);
    cfgDestroy(&config);
}

static void
cfgOutPeriodSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgOutPeriodSet(config, 0);
    assert_int_equal(cfgOutPeriod(config), 0);
    cfgOutPeriodSet(config, UINT_MAX);
    assert_int_equal(cfgOutPeriod(config), UINT_MAX);
    cfgDestroy(&config);
}

static void
cfgTransportTypeSetAndGet(void** state)
{
    which_transport_t t = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();
    cfgTransportTypeSet(config, t, CFG_UDP);
    assert_int_equal(cfgTransportType(config, t), CFG_UDP);
    cfgTransportTypeSet(config, t, CFG_UNIX);
    assert_int_equal(cfgTransportType(config, t), CFG_UNIX);
    cfgTransportTypeSet(config, t, CFG_FILE);
    assert_int_equal(cfgTransportType(config, t), CFG_FILE);
    cfgTransportTypeSet(config, t, CFG_SYSLOG);
    assert_int_equal(cfgTransportType(config, t), CFG_SYSLOG);
    cfgTransportTypeSet(config, t, CFG_SHM);
    assert_int_equal(cfgTransportType(config, t), CFG_SHM);
    cfgDestroy(&config);
}

static void
cfgTransportHostSetAndGet(void** state)
{
    which_transport_t t = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();
    cfgTransportHostSet(config, t, "larrysComputer");
    assert_string_equal(cfgTransportHost(config, t), "larrysComputer");
    cfgTransportHostSet(config, t, "bobsComputer");
    assert_string_equal(cfgTransportHost(config, t), "bobsComputer");
    cfgTransportHostSet(config, t, NULL);
    assert_null(cfgTransportHost(config, t));
    cfgDestroy(&config);
}

static void
cfgTransportPortSetAndGet(void** state)
{
    which_transport_t t = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();
    cfgTransportPortSet(config, t, 54321);
    assert_int_equal(cfgTransportPort(config, t), 54321);
    cfgTransportPortSet(config, t, 12345);
    assert_int_equal(cfgTransportPort(config, t), 12345);
    cfgDestroy(&config);
}

static void
cfgTransportPathSetAndGet(void** state)
{
    which_transport_t t = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();
    cfgTransportPathSet(config, t, "/tmp/mysock");
    assert_string_equal(cfgTransportPath(config, t), "/tmp/mysock");
    cfgTransportPathSet(config, t, NULL);
    assert_null(cfgTransportPath(config, t));
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
    cfgLogLevelSet(config, CFG_LOG_DEBUG);
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
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
    const char* yamlText =
        "---\n" 
        "output:\n" 
        "  format:\n"
        "    type: newlinedelimited          # expandedstatsd, newlinedelimited\n"
        "    statsdprefix : 'cribl.scope'    # prepends each statsd metric\n"
        "    statsdmaxlen : 1024             # max size of a formatted statsd string\n"
        "    verbosity: 3                    # 0-9 (0 is least verbose, 9 is most)\n"
        "  transport:                        # defines how scope output is sent\n" 
        "    type: file                      # udp, unix, file, syslog\n" 
        "    path: '/var/log/scope.log'\n"
        "  summaryperiod: 11                 # in seconds\n"
        "filteredFunctions:                  # keep Scope from processing these\n" 
        "  - read                            # specific intercepted functions\n" 
        "  - write\n" 
        "  - accept\n" 
        "logging:\n"
        "  level: debug                      # debug, info, warning, error, none\n"
        "  transport:\n"
        "    type: syslog\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_NEWLINE_DELIMITED);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 1024);
    assert_int_equal(cfgOutVerbosity(config), 3);
    assert_int_equal(cfgOutPeriod(config), 11);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_int_equal(cfgTransportPort(config, CFG_OUT), 8125);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/log/scope.log");
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_SYSLOG);
    assert_string_equal(cfgTransportHost(config, CFG_LOG), "127.0.0.1");
    assert_int_equal(cfgTransportPort(config, CFG_LOG), 8125);
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_true(cfgFuncIsFiltered(config, "write"));
    assert_true(cfgFuncIsFiltered(config, "accept"));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    cfgDestroy(&config);
    deleteFile(path);
}

static void
writeFileWithSubstitution(const char* path, const char* base, const char* variable)
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
    const char* yamlText =
        "---\n"
        "output:\n"
        "  transport:\n"
        "%s"
        "...\n";

    const char* udp_str =
        "    type: udp\n"
        "    host: 'labmachine8235'\n"
        "    port: 235\n";
    const char* unix_str =
        "    type: unix\n"
        "    path: '/var/run/scope.sock'\n";
    const char* file_str =
        "    type: file\n"
        "    path: '/var/log/scope.log'\n";
    const char* syslog_str =
        "    type: syslog\n";
    const char* shm_str =
        "    type: shm\n";
    const char* transport_lines[] = {udp_str, unix_str, file_str, syslog_str, shm_str};

    const char* path = CFG_FILE_NAME;

    int i;
    for (i = 0; i<sizeof(transport_lines) / sizeof(transport_lines[0]); i++) {

        writeFileWithSubstitution(path, yamlText, transport_lines[i]);
        config_t* config = cfgRead(path);

        if (transport_lines[i] == udp_str) {
                assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UDP);
                assert_string_equal(cfgTransportHost(config, CFG_OUT), "labmachine8235");
                assert_int_equal(cfgTransportPort(config, CFG_OUT), 235);
        } else if (transport_lines[i] == unix_str) {
                assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UNIX);
                assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/run/scope.sock");
        } else if (transport_lines[i] == file_str) {
                assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_FILE);
                assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/log/scope.log");
        } else if (transport_lines[i] == syslog_str) {
                assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_SYSLOG);
        } else if (transport_lines[i] == shm_str) {
                assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_SHM);
         }

        deleteFile(path);
        cfgDestroy(&config);
    }

}

static void
cfgReadEveryProcessLevel(void** state)
{
    const char* yamlText =
        "---\n"
        "logging:\n"
        "  level: %s\n"
        "...\n";

    const char* path = CFG_FILE_NAME;
    const char* level[] = {"debug", "info", "warning", "error", "none", "dpath"};
    cfg_log_level_t value[] = {CFG_LOG_DPATH, CFG_LOG_DEBUG, CFG_LOG_INFO, CFG_LOG_WARN, CFG_LOG_ERROR, CFG_LOG_NONE};
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
    const char* jsonText =
        "{\n"
        "  'output': {\n"
        "    'format': {\n"
        "      'type': 'newlinedelimited',\n"
        "      'statsdprefix': 'cribl.scope',\n"
        "      'statsdmaxlen': '42',\n"
        "      'verbosity': '0'\n"
        "    },\n"
        "    'transport': {\n"
        "      'type': 'file',\n"
        "      'path': '/var/log/scope.log'\n"
        "    },\n"
        "    'summaryperiod': '13'\n"
        "  },\n"
        "  'filteredFunctions': [\n"
        "    'read',\n"
        "    'write',\n"
        "    'accept'\n"
        "  ],\n"
        "  'logging': {\n"
        "    'level': 'debug',\n"
        "    'transport': {\n"
        "      'type': 'shm'\n"
        "    }\n"
        "  }\n"
        "}\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, jsonText);
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_NEWLINE_DELIMITED);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 42);
    assert_int_equal(cfgOutVerbosity(config), 0);
    assert_int_equal(cfgOutPeriod(config), 13);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_int_equal(cfgTransportPort(config, CFG_OUT), 8125);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/log/scope.log");
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_SHM);
    assert_string_equal(cfgTransportHost(config, CFG_LOG), "127.0.0.1");
    assert_int_equal(cfgTransportPort(config, CFG_LOG), 8125);
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_true(cfgFuncIsFiltered(config, "write"));
    assert_true(cfgFuncIsFiltered(config, "accept"));
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
    const char* yamlText =
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
        "  transport:\n"
        "    type: syslog\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    verifyDefaults(config);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadExtraFieldsAreHarmless(void** state)
{
    const char* yamlText =
        "---\n"
        "momsApplePieRecipe:                # has possibilities...\n"
        "  [apples,sugar,flour,dirt]        # dirt mom?  Really?\n"
        "output:\n"
        "  format:\n"
        "    type: expandedstatsd\n"
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
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_EXPANDED_STATSD);
    assert_string_equal(cfgOutStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/run/scope.sock");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadYamlOrderWithinStructureDoesntMatter(void** state)
{
    const char* yamlText =
        "---\n"
        "logging:\n"
        "  level: info\n"
        "filteredFunctions:\n"
        "  - read\n"
        "output:\n"
        "  summaryperiod: 42\n"
        "  transport:\n"
        "    path: '/var/run/scope.sock'\n"
        "    type: unix\n"
        "  format:\n"
        "    verbosity: 4294967295\n"
        "    statsdmaxlen: 4294967295\n"
        "    statsdprefix: 'cribl.scope'\n"
        "    type:  expandedstatsd\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_EXPANDED_STATSD);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 4294967295);
    assert_int_equal(cfgOutVerbosity(config), CFG_MAX_VERBOSITY);
    assert_int_equal(cfgOutPeriod(config), 42);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/run/scope.sock");
    assert_non_null(cfgFuncFilters(config));
    assert_true(cfgFuncIsFiltered(config, "read"));
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    void* out_state[] = {(void*)CFG_OUT, NULL};
    void* log_state[] = {(void*)CFG_LOG, NULL};

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgCreateDefaultReturnsValidPtr),
        cmocka_unit_test(accessorValuesForDefaultConfigAreAsExpected),
        cmocka_unit_test(accessorsReturnDefaultsWhenConfigIsNull),
        cmocka_unit_test(cfgOutFormatSetAndGet),
        cmocka_unit_test(cfgOutStatsDPrefixSetAndGet),
        cmocka_unit_test(cfgOutStatsDMaxLenSetAndGet),
        cmocka_unit_test(cfgOutVerbositySetAndGet),
        cmocka_unit_test(cfgOutPeriodSetAndGet),

        cmocka_unit_test_prestate(cfgTransportTypeSetAndGet, out_state),
        cmocka_unit_test_prestate(cfgTransportHostSetAndGet, out_state),
        cmocka_unit_test_prestate(cfgTransportPortSetAndGet, out_state),
        cmocka_unit_test_prestate(cfgTransportPathSetAndGet, out_state),

        cmocka_unit_test_prestate(cfgTransportTypeSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportHostSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportPortSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportPathSetAndGet, log_state),

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


