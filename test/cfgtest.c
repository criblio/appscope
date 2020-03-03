#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "cfg.h"
#include "dbg.h"

#include "test.h"


static void
verifyDefaults(config_t* config)
{
    assert_int_equal       (cfgMtcFormat(config), DEFAULT_MTC_FORMAT);
    assert_string_equal    (cfgMtcStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal       (cfgMtcStatsDMaxLen(config), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal       (cfgMtcVerbosity(config), DEFAULT_MTC_VERBOSITY);
    assert_int_equal       (cfgMtcPeriod(config), DEFAULT_SUMMARY_PERIOD);
    assert_string_equal    (cfgCmdDir(config), DEFAULT_COMMAND_DIR);
    assert_int_equal       (cfgEventFormat(config), DEFAULT_CTL_FORMAT);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_VALUE);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_FIELD);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_NAME);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_FILE), DEFAULT_SRC_FILE);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC);
    assert_int_equal       (cfgTransportType(config, CFG_MTC), CFG_UDP);
    assert_string_equal    (cfgTransportHost(config, CFG_MTC), "127.0.0.1");
    assert_string_equal    (cfgTransportPort(config, CFG_MTC), DEFAULT_MTC_PORT);
    assert_null            (cfgTransportPath(config, CFG_MTC));
    assert_int_equal       (cfgTransportBuf(config, CFG_MTC), CFG_BUFFER_LINE);
    assert_int_equal       (cfgTransportType(config, CFG_CTL), CFG_TCP);
    assert_string_equal    (cfgTransportHost(config, CFG_CTL), "127.0.0.1");
    assert_string_equal    (cfgTransportPort(config, CFG_CTL), DEFAULT_CTL_PORT);
    assert_null            (cfgTransportPath(config, CFG_CTL));
    assert_int_equal       (cfgTransportBuf(config, CFG_CTL), CFG_BUFFER_FULLY);
    assert_int_equal       (cfgTransportType(config, CFG_LOG), CFG_FILE);
    assert_null            (cfgTransportHost(config, CFG_LOG));
    assert_null            (cfgTransportPort(config, CFG_LOG));
    assert_string_equal    (cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_int_equal       (cfgTransportBuf(config, CFG_MTC), CFG_BUFFER_LINE);
    assert_null            (cfgCustomTags(config));
    assert_null            (cfgCustomTagValue(config, "tagname"));
    assert_int_equal       (cfgLogLevel(config), DEFAULT_LOG_LEVEL);
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
cfgMtcFormatSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgMtcFormatSet(config, CFG_METRIC_JSON);
    assert_int_equal(cfgMtcFormat(config), CFG_METRIC_JSON);
    cfgMtcFormatSet(config, CFG_METRIC_STATSD);
    assert_int_equal(cfgMtcFormat(config), CFG_METRIC_STATSD);
    cfgMtcFormatSet(config, CFG_EVENT_ND_JSON);
    assert_int_equal(cfgMtcFormat(config), CFG_EVENT_ND_JSON);
    cfgDestroy(&config);
}

static void
cfgMtcStatsDPrefixSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgMtcStatsDPrefixSet(config, "heywithdot.");
    assert_string_equal(cfgMtcStatsDPrefix(config), "heywithdot.");
    cfgMtcStatsDPrefixSet(config, "heywithoutdot");
    assert_string_equal(cfgMtcStatsDPrefix(config), "heywithoutdot.");
    cfgMtcStatsDPrefixSet(config, NULL);
    assert_string_equal(cfgMtcStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    cfgMtcStatsDPrefixSet(config, "");
    assert_string_equal(cfgMtcStatsDPrefix(config), "");
    cfgDestroy(&config);
}

static void
cfgMtcStatsDMaxLenSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgMtcStatsDMaxLenSet(config, 0);
    assert_int_equal(cfgMtcStatsDMaxLen(config), 0);
    cfgMtcStatsDMaxLenSet(config, UINT_MAX);
    assert_int_equal(cfgMtcStatsDMaxLen(config), UINT_MAX);
    cfgDestroy(&config);
}

static void
cfgMtcVerbositySetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    int i;
    for (i=0; i<=CFG_MAX_VERBOSITY+1; i++) {
        cfgMtcVerbositySet(config, i);
        if (i > CFG_MAX_VERBOSITY)
            assert_int_equal(cfgMtcVerbosity(config), CFG_MAX_VERBOSITY);
        else
            assert_int_equal(cfgMtcVerbosity(config), i);
    }
    cfgMtcVerbositySet(config, UINT_MAX);
    assert_int_equal(cfgMtcVerbosity(config), CFG_MAX_VERBOSITY);
    cfgDestroy(&config);
}

static void
cfgMtcPeriodSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgMtcPeriodSet(config, 0);
    assert_int_equal(cfgMtcPeriod(config), 0);
    cfgMtcPeriodSet(config, UINT_MAX);
    assert_int_equal(cfgMtcPeriod(config), UINT_MAX);
    cfgDestroy(&config);
}

static void
cfgCmdDirSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgCmdDirSet(config, "/some/path");
    assert_string_equal(cfgCmdDir(config), "/some/path");
    cfgCmdDirSet(config, NULL);
    assert_string_equal(cfgCmdDir(config), DEFAULT_COMMAND_DIR);
    cfgDestroy(&config);
}

static void
cfgEventFormatSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    cfgEventFormatSet(config, CFG_METRIC_STATSD);
    assert_int_equal(cfgEventFormat(config), CFG_METRIC_STATSD);
    cfgEventFormatSet(config, CFG_METRIC_JSON);
    assert_int_equal(cfgEventFormat(config), CFG_METRIC_JSON);
    cfgEventFormatSet(config, CFG_EVENT_ND_JSON);
    assert_int_equal(cfgEventFormat(config), CFG_EVENT_ND_JSON);
    cfgDestroy(&config);
}

typedef struct
{
    watch_t   src;
    const char* default_value;
    const char* default_field;
    const char* default_name;
} source_state_t;

static void
cfgEvtFormatValueFilterSetAndGet(void** state)
{
    source_state_t* data = (source_state_t*)state[0];

    config_t* config = cfgCreateDefault();
    cfgEvtFormatValueFilterSet(config, data->src, ".*\\.log$");
    assert_string_equal(cfgEvtFormatValueFilter(config, data->src), ".*\\.log$");
    cfgEvtFormatValueFilterSet(config, data->src, "^/var/log/.*");
    assert_string_equal(cfgEvtFormatValueFilter(config, data->src), "^/var/log/.*");
    cfgEvtFormatValueFilterSet(config, data->src, NULL);
    assert_string_equal(cfgEvtFormatValueFilter(config, data->src), data->default_value);
    cfgDestroy(&config);
}

static void
cfgEvtFormatFieldFilterSetAndGet(void** state)
{
    source_state_t* data = (source_state_t*)state[0];

    config_t* config = cfgCreateDefault();
    cfgEvtFormatFieldFilterSet(config, data->src, ".*\\.log$");
    assert_string_equal(cfgEvtFormatFieldFilter(config, data->src), ".*\\.log$");
    cfgEvtFormatFieldFilterSet(config, data->src, "^/var/log/.*");
    assert_string_equal(cfgEvtFormatFieldFilter(config, data->src), "^/var/log/.*");
    cfgEvtFormatFieldFilterSet(config, data->src, NULL);
    assert_string_equal(cfgEvtFormatFieldFilter(config, data->src), data->default_field);
    cfgDestroy(&config);
}

static void
cfgEvtFormatNameFilterSetAndGet(void** state)
{
    source_state_t* data = (source_state_t*)state[0];

    config_t* config = cfgCreateDefault();
    cfgEvtFormatNameFilterSet(config, data->src, ".*\\.log$");
    assert_string_equal(cfgEvtFormatNameFilter(config, data->src), ".*\\.log$");
    cfgEvtFormatNameFilterSet(config, data->src, "^/var/log/.*");
    assert_string_equal(cfgEvtFormatNameFilter(config, data->src), "^/var/log/.*");
    cfgEvtFormatNameFilterSet(config, data->src, NULL);
    assert_string_equal(cfgEvtFormatNameFilter(config, data->src), data->default_name);
    cfgDestroy(&config);
}

static void
cfgEvtFormatSourceEnabledSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();

    // Set everything to 1
    int i, j;
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX+1; i++) {
        cfgEvtFormatSourceEnabledSet(config, i, 1);
        if (i >= CFG_SRC_MAX) {
             assert_int_equal(cfgEvtFormatSourceEnabled(config, i), DEFAULT_SRC_FILE);
             assert_int_equal(dbgCountMatchingLines("src/cfg.c"), 1);
             dbgInit(); // reset dbg for the rest of the tests
        } else {
             assert_int_equal(dbgCountMatchingLines("src/cfg.c"), 0);
             assert_int_equal(cfgEvtFormatSourceEnabled(config, i), 1);
        }
    }

    // Clear one at a time to see there aren't side effects
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX; i++) {
        cfgEvtFormatSourceEnabledSet(config, i, 0); // Clear it
        for (j=CFG_SRC_FILE; j<CFG_SRC_MAX; j++) {
            if (i==j)
                 assert_int_equal(cfgEvtFormatSourceEnabled(config, j), 0);
            else
                 assert_int_equal(cfgEvtFormatSourceEnabled(config, j), 1);
        }
        cfgEvtFormatSourceEnabledSet(config, i, 1); // Set it back
    }

    cfgDestroy(&config);

    // Test get with NULL config
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX; i++) {
        unsigned expected;
        switch (i) {
            case CFG_SRC_FILE:
                expected = DEFAULT_SRC_FILE;
                break;
            case CFG_SRC_CONSOLE:
                expected = DEFAULT_SRC_CONSOLE;
                break;
            case CFG_SRC_SYSLOG:
                expected = DEFAULT_SRC_SYSLOG;
                break;
            case CFG_SRC_METRIC:
                expected = DEFAULT_SRC_METRIC;
                break;
        }
        assert_int_equal(cfgEvtFormatSourceEnabled(config, i), expected);
    }
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
    which_transport_t typeMtcCtlLog = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();

    // You can no longer hardcode host/port
    cfgTransportTypeSet(config, typeMtcCtlLog, CFG_UDP);

    cfgTransportHostSet(config, typeMtcCtlLog, "larrysComputer");
    assert_string_equal(cfgTransportHost(config, typeMtcCtlLog), "larrysComputer");
    cfgTransportHostSet(config, typeMtcCtlLog, "bobsComputer");
    assert_string_equal(cfgTransportHost(config, typeMtcCtlLog), "bobsComputer");
    cfgTransportHostSet(config, typeMtcCtlLog, NULL);
    assert_null(cfgTransportHost(config, typeMtcCtlLog));
    cfgDestroy(&config);
}

static void
cfgTransportPortSetAndGet(void** state)
{
    which_transport_t typeMtcCtlLog = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();

    // You can no longer hardcode host/port
    cfgTransportTypeSet(config, typeMtcCtlLog, CFG_UDP);

    cfgTransportPortSet(config, typeMtcCtlLog, "54321");
    assert_string_equal(cfgTransportPort(config, typeMtcCtlLog), "54321");
    cfgTransportPortSet(config, typeMtcCtlLog, "12345");
    assert_string_equal(cfgTransportPort(config, typeMtcCtlLog), "12345");
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
cfgTransportBufSetAndGet(void** state)
{
    which_transport_t t = *(which_transport_t*)state[0];
    config_t* config = cfgCreateDefault();
    cfgTransportBufSet(config, t, CFG_BUFFER_LINE);
    assert_int_equal(cfgTransportBuf(config, t), CFG_BUFFER_LINE);
    cfgTransportBufSet(config, t, CFG_BUFFER_FULLY);
    assert_int_equal(cfgTransportBuf(config, t), CFG_BUFFER_FULLY);

    // Don't crash
    cfgTransportBufSet(NULL, t, CFG_BUFFER_FULLY);
    cfgTransportBuf(NULL, t);
    cfgTransportBufSet(config, t, CFG_BUFFER_LINE+1);

    cfgDestroy(&config);
}


static void
cfgCustomTagsSetAndGet(void** state)
{
    config_t* config = cfgCreateDefault();
    assert_null(cfgCustomTags(config));
    int i;
    for (i=0; i<10; i++) {
        char name[64];
        char value[64];
        snprintf(name, sizeof(name), "name%d", i);
        snprintf(value, sizeof(value), "value%d", i);

        assert_false(cfgCustomTagValue(config, name));
        cfgCustomTagAdd(config, name, value);
        assert_string_equal(cfgCustomTagValue(config, name), value);
        custom_tag_t** tags = cfgCustomTags(config);
        assert_non_null(tags[i]);
        assert_string_equal(tags[i]->name, name);
        assert_string_equal(tags[i]->value, value);
        assert_null(tags[i+1]);
    }

    // test that a tag can be overridden by a later tag
    cfgCustomTagAdd(config, "name0", "some other value");
    assert_string_equal(cfgCustomTagValue(config, "name0"), "some other value");

    // test that invalid values don't crash
    cfgCustomTagAdd(config, NULL, "something");
    cfgCustomTagAdd(config, "something", NULL);
    cfgCustomTagAdd(config, NULL, NULL);
    cfgCustomTagAdd(NULL, "something", "something else");

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


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    void* mtc_state[] = {(void*)CFG_MTC, NULL};
    void* evt_state[] = {(void*)CFG_CTL, NULL};
    void* log_state[] = {(void*)CFG_LOG, NULL};

    source_state_t log = {CFG_SRC_FILE, DEFAULT_SRC_FILE_VALUE, DEFAULT_SRC_FILE_FIELD, DEFAULT_SRC_FILE_NAME};
    source_state_t con = {CFG_SRC_CONSOLE, DEFAULT_SRC_CONSOLE_VALUE, DEFAULT_SRC_CONSOLE_FIELD, DEFAULT_SRC_CONSOLE_NAME};
    source_state_t sys = {CFG_SRC_SYSLOG, DEFAULT_SRC_SYSLOG_VALUE, DEFAULT_SRC_SYSLOG_FIELD, DEFAULT_SRC_SYSLOG_NAME};
    source_state_t met = {CFG_SRC_METRIC, DEFAULT_SRC_METRIC_VALUE, DEFAULT_SRC_METRIC_FIELD, DEFAULT_SRC_METRIC_NAME};

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgCreateDefaultReturnsValidPtr),
        cmocka_unit_test(accessorValuesForDefaultConfigAreAsExpected),
        cmocka_unit_test(accessorsReturnDefaultsWhenConfigIsNull),
        cmocka_unit_test(cfgMtcFormatSetAndGet),
        cmocka_unit_test(cfgMtcStatsDPrefixSetAndGet),
        cmocka_unit_test(cfgMtcStatsDMaxLenSetAndGet),
        cmocka_unit_test(cfgMtcVerbositySetAndGet),
        cmocka_unit_test(cfgMtcPeriodSetAndGet),
        cmocka_unit_test(cfgCmdDirSetAndGet),
        cmocka_unit_test(cfgEventFormatSetAndGet),

        cmocka_unit_test_prestate(cfgEvtFormatValueFilterSetAndGet, &log),
        cmocka_unit_test_prestate(cfgEvtFormatValueFilterSetAndGet, &con),
        cmocka_unit_test_prestate(cfgEvtFormatValueFilterSetAndGet, &sys),
        cmocka_unit_test_prestate(cfgEvtFormatValueFilterSetAndGet, &met),

        cmocka_unit_test_prestate(cfgEvtFormatFieldFilterSetAndGet, &log),
        cmocka_unit_test_prestate(cfgEvtFormatFieldFilterSetAndGet, &con),
        cmocka_unit_test_prestate(cfgEvtFormatFieldFilterSetAndGet, &sys),
        cmocka_unit_test_prestate(cfgEvtFormatFieldFilterSetAndGet, &met),

        cmocka_unit_test_prestate(cfgEvtFormatNameFilterSetAndGet, &log),
        cmocka_unit_test_prestate(cfgEvtFormatNameFilterSetAndGet, &con),
        cmocka_unit_test_prestate(cfgEvtFormatNameFilterSetAndGet, &sys),
        cmocka_unit_test_prestate(cfgEvtFormatNameFilterSetAndGet, &met),

        cmocka_unit_test(cfgEvtFormatSourceEnabledSetAndGet),

        cmocka_unit_test_prestate(cfgTransportTypeSetAndGet, mtc_state),
        cmocka_unit_test_prestate(cfgTransportHostSetAndGet, mtc_state),
        cmocka_unit_test_prestate(cfgTransportPortSetAndGet, mtc_state),
        cmocka_unit_test_prestate(cfgTransportPathSetAndGet, mtc_state),
        cmocka_unit_test_prestate(cfgTransportBufSetAndGet,  mtc_state),

        cmocka_unit_test_prestate(cfgTransportTypeSetAndGet, evt_state),
        cmocka_unit_test_prestate(cfgTransportHostSetAndGet, evt_state),
        cmocka_unit_test_prestate(cfgTransportPortSetAndGet, evt_state),
        cmocka_unit_test_prestate(cfgTransportPathSetAndGet, evt_state),
        cmocka_unit_test_prestate(cfgTransportBufSetAndGet,  evt_state),

        cmocka_unit_test_prestate(cfgTransportTypeSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportHostSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportPortSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportPathSetAndGet, log_state),
        cmocka_unit_test_prestate(cfgTransportBufSetAndGet,  log_state),

        cmocka_unit_test(cfgCustomTagsSetAndGet),
        cmocka_unit_test(cfgLoggingSetAndGet),
        cmocka_unit_test(cfgLogLevelSetAndGet),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}


