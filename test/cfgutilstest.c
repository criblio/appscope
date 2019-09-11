#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cfgutils.h"

#include "test.h"

#define MAX_PATH 1024

void
cfgPathHonorsPriorityOrder(void** state)
{
    // Get HOME env variable
    const char* home = getenv("HOME");
    assert_non_null(home);
    char homeConfDir[MAX_PATH];
    int rv = snprintf(homeConfDir, sizeof(homeConfDir), "%s/conf", home);
    assert_int_not_equal(rv, -1);

    // Create temp directories
    const char* newdir[] = {"testtempdir1", "testtempdir2", 
                      "testtempdir1/conf", "testtempdir2/conf", 
                      homeConfDir};
    int i;
    for (i=0; i<sizeof(newdir)/sizeof(newdir[0]); i++) {
        assert_int_equal(mkdir(newdir[i], 0777), 0);
    }

    // get the basedir, set cwd and scopeHome from it
    char basedir[MAX_PATH];
    char cwd[MAX_PATH];
    char scopeHome[MAX_PATH];
    assert_non_null(getcwd(basedir, sizeof(basedir)));
    snprintf(cwd, sizeof(cwd), "%s/%s", basedir, newdir[0]);
    snprintf(scopeHome, sizeof(scopeHome), "%s/%s", basedir, newdir[1]);

    // Change to cwd
    assert_int_equal(chdir(cwd), 0);

    // Set SCOPE_HOME to the other
    assert_int_equal(setenv("SCOPE_HOME", scopeHome, 1), 0);

    // Create the paths we want to test
    const char file[] = CFG_FILE_NAME ".test"; // scope.cfg.test
    char path[6][MAX_PATH];
    // Lowest priority first
    snprintf(path[0], sizeof(path[0]), "%s/%s", cwd, file);
    snprintf(path[1], sizeof(path[1]), "%s/conf/%s", cwd, file);
    snprintf(path[2], sizeof(path[2]), "%s/%s", home, file);
    snprintf(path[3], sizeof(path[3]), "%s/conf/%s", home, file);
    // Skip for now...  we may not have permissions to write to /etc/scope
    //snprintf(path[4], sizeof(path[4]), "/etc/scope/%s", file);
    snprintf(path[4], sizeof(path[4]), "%s/%s", scopeHome, file);
    snprintf(path[5], sizeof(path[5]), "%s/conf/%s", scopeHome, file);

    // Test that none of them exist before we start
    const int count = sizeof(path)/sizeof(path[0]);
    for (i=0; i<count; i++) {
        struct stat s;
        if (!stat(path[i], &s) && S_ISDIR(s.st_mode)) {
            fail_msg("Found unexpected path that will interfere with test: %s", path[i]);
        }
    }

    // Create them in priority order, and test that they are returned
    for (i=0; i<count; i++) {
        int fd = creat(path[i], 0777);
        assert_int_not_equal(fd, -1);
        assert_int_equal(close(fd), 0);
        char* result = cfgPath(file);
        assert_non_null(result);
	if (strcmp(result, path[i])) {
            fail_msg("Expected %s but found %s for i=%d", path[i], result, i);
	}
        if (result) free(result);
    }

    // Delete the files we just created
    for (i=0; i<count; i++) {
        assert_int_equal(unlink(path[i]), 0);
    }

    // change back to basedir
    assert_int_equal(chdir(basedir), 0);

    // Delete the directories we just created
    for (i=(sizeof(newdir)/sizeof(newdir[0]))-1; i>=0; i--) {
        assert_int_equal(rmdir(newdir[i]), 0);
    }
}

void
cfgProcessEnvironmentOutFormat(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutFormatSet(cfg, CFG_NEWLINE_DELIMITED);
    assert_int_equal(cfgOutFormat(cfg), CFG_NEWLINE_DELIMITED);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_OUT_FORMAT", "expandedstatsd", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_EXPANDED_STATSD);

    assert_int_equal(setenv("SCOPE_OUT_FORMAT", "newlinedelimited", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_NEWLINE_DELIMITED);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_OUT_FORMAT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_NEWLINE_DELIMITED);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_OUT_FORMAT", "bson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_NEWLINE_DELIMITED);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentStatsDPrefix(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutStatsDPrefixSet(cfg, "something");
    assert_string_equal(cfgOutStatsDPrefix(cfg), "something.");

    // should override current cfg
    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgOutStatsDPrefix(cfg), "blah.");

    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "hey", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgOutStatsDPrefix(cfg), "hey.");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_STATSD_PREFIX"), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgOutStatsDPrefix(cfg), "hey.");

    // empty string
    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgOutStatsDPrefix(cfg), "");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentStatsDMaxLen(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutStatsDMaxLenSet(cfg, 0);
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 3);

    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_STATSD_MAXLEN"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentOutPeriod(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutPeriodSet(cfg, 0);
    assert_int_equal(cfgOutPeriod(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_OUT_SUM_PERIOD", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 3);

    assert_int_equal(setenv("SCOPE_OUT_SUM_PERIOD", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_OUT_SUM_PERIOD"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_OUT_SUM_PERIOD", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentOutVerbosity(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutVerbositySet(cfg, 0);
    assert_int_equal(cfgOutVerbosity(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_OUT_VERBOSITY", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 3);

    assert_int_equal(setenv("SCOPE_OUT_VERBOSITY", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_OUT_VERBOSITY"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_OUT_VERBOSITY", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentStatsdTags(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgCustomTagAdd(cfg, "NAME1", "val1");
    assert_non_null(cfgCustomTags(cfg));
    assert_string_equal(cfgCustomTagValue(cfg, "NAME1"), "val1");
    assert_string_equal(cfgCustomTags(cfg)[0]->name, "NAME1");
    assert_string_equal(cfgCustomTags(cfg)[0]->value, "val1");
    assert_null(cfgCustomTags(cfg)[1]);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_TAG_NAME1", "newvalue", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCustomTagValue(cfg, "NAME1"), "newvalue");
    assert_string_equal(cfgCustomTags(cfg)[0]->name, "NAME1");
    assert_string_equal(cfgCustomTags(cfg)[0]->value, "newvalue");
    assert_null(cfgCustomTags(cfg)[1]);

    // should extend current cfg
    assert_int_equal(setenv("SCOPE_TAG_NAME2", "val2", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCustomTagValue(cfg, "NAME2"), "val2");
    assert_string_equal(cfgCustomTags(cfg)[1]->name, "NAME2");
    assert_string_equal(cfgCustomTags(cfg)[1]->value, "val2");
    assert_null(cfgCustomTags(cfg)[2]);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_TAG_NAME1"), 0);
    assert_int_equal(unsetenv("SCOPE_TAG_NAME2"), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCustomTagValue(cfg, "NAME1"), "newvalue");
    assert_string_equal(cfgCustomTagValue(cfg, "NAME2"), "val2");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentOutTransport(void** state)
{
    config_t* cfg = cfgCreateDefault();

    // should override current cfg
    assert_int_equal(setenv("SCOPE_OUT_DEST", "udp://host:234", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_OUT), CFG_UDP);
    assert_string_equal(cfgTransportHost(cfg, CFG_OUT), "host");
    assert_string_equal(cfgTransportPort(cfg, CFG_OUT), "234");

    // test that our code doesn't modify the env variable directly
    assert_string_equal(getenv("SCOPE_OUT_DEST"), "udp://host:234");

    assert_int_equal(setenv("SCOPE_OUT_DEST", "file:///some/path/somewhere", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/some/path/somewhere");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_OUT_DEST"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/some/path/somewhere");

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_OUT_DEST", "somewhere else", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/some/path/somewhere");

    // port is required, if not there cfg should not be modified
    assert_int_equal(setenv("SCOPE_OUT_DEST", "udp://host", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/some/path/somewhere");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentLogTransport(void** state)
{
    config_t* cfg = cfgCreateDefault();

    // should override current cfg
    assert_int_equal(setenv("SCOPE_LOG_DEST", "udp://host:234", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_LOG), CFG_UDP);
    assert_string_equal(cfgTransportHost(cfg, CFG_LOG), "host");
    assert_string_equal(cfgTransportPort(cfg, CFG_LOG), "234");

    // test that our code doesn't modify the env variable directly
    assert_string_equal(getenv("SCOPE_LOG_DEST"), "udp://host:234");

    assert_int_equal(setenv("SCOPE_LOG_DEST", "file:///some/path/somewhere", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_LOG), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/some/path/somewhere");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_LOG_DEST"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_LOG), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/some/path/somewhere");

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_LOG_DEST", "somewhere else", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_LOG), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/some/path/somewhere");

    // port is required, if not there cfg should not be modified
    assert_int_equal(setenv("SCOPE_LOG_DEST", "udp://host", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, CFG_LOG), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/some/path/somewhere");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
cfgProcessEnvironmentLogLevel(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgLogLevelSet(cfg, CFG_LOG_DEBUG);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_DEBUG);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_LOG_LEVEL", "trace", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);

    assert_int_equal(setenv("SCOPE_LOG_LEVEL", "debug", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_DEBUG);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_LOG_LEVEL"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_DEBUG);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_LOG_LEVEL", "everythingandmore", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_DEBUG);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

void
initLogReturnsPtr(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_SHM; t++) {
	    switch (t) {
            case CFG_UDP:
                cfgTransportHostSet(cfg, CFG_LOG, "localhost");
                cfgTransportPortSet(cfg, CFG_LOG, "4444");
                break;
            case CFG_UNIX:
				cfgTransportPathSet(cfg, CFG_LOG, "/var/run/scope.sock");
                break;
            case CFG_FILE:
				cfgTransportPathSet(cfg, CFG_LOG, "/tmp/scope.log");
                break;
            case CFG_SYSLOG:
            case CFG_SHM:
                break;
	    }
        cfgTransportTypeSet(cfg, CFG_LOG, t);
        log_t* log = initLog(cfg);
        assert_non_null(log);
        logDestroy(&log);
    }
    cfgDestroy(&cfg);
}

void
initOutReturnsPtrWithNullLogReference(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_SHM; t++) {
        cfgTransportTypeSet(cfg, CFG_OUT, t);
        if (t==CFG_UNIX || t==CFG_FILE) {
            cfgTransportPathSet(cfg, CFG_OUT, "/tmp/scope.log");
        }
        out_t* out = initOut(cfg, NULL);
        assert_non_null(out);
        outDestroy(&out);
    }
    cfgDestroy(&cfg);
}

void
initOutReturnsPtrWithLogReference(void** state)
{
    // Create cfg
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);
    cfgTransportTypeSet(cfg, CFG_OUT, CFG_FILE);
    cfgTransportPathSet(cfg, CFG_OUT, "/tmp/scope.log");

    // Create log
    log_t* log = logCreate();

    // Run the test
    out_t* out = initOut(cfg, log);
    assert_non_null(out);

    // Cleanup
    logDestroy(&log);
    outDestroy(&out);
    cfgDestroy(&cfg);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgPathHonorsPriorityOrder),
        cmocka_unit_test(cfgProcessEnvironmentOutFormat),
        cmocka_unit_test(cfgProcessEnvironmentStatsDPrefix),
        cmocka_unit_test(cfgProcessEnvironmentStatsDMaxLen),
        cmocka_unit_test(cfgProcessEnvironmentOutPeriod),
        cmocka_unit_test(cfgProcessEnvironmentOutVerbosity),
        cmocka_unit_test(cfgProcessEnvironmentStatsdTags),
        cmocka_unit_test(cfgProcessEnvironmentOutTransport),
        cmocka_unit_test(cfgProcessEnvironmentLogTransport),
        cmocka_unit_test(cfgProcessEnvironmentLogLevel),
        cmocka_unit_test(initLogReturnsPtr),
        cmocka_unit_test(initOutReturnsPtrWithNullLogReference),
        cmocka_unit_test(initOutReturnsPtrWithLogReference),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}


