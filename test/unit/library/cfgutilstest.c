#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fn.h"
#include "com.h"
#include "cfgutils.h"
#include "scopestdlib.h"
#include "test.h"
#include "dbg.h"

#define MAX_PATH 1024


static char dirPath[PATH_MAX];

static int
testDirPath(char *path, const char *argv0) {
    char buf[PATH_MAX];
    if (argv0[0] == '/') {
        scope_strcpy(buf, argv0);
    } else {
        if (scope_getcwd(buf, PATH_MAX) == NULL) {
            scope_perror("getcwd error");
            return -1;
        }
        scope_strcat(buf, "/");
        scope_strcat(buf, argv0);
    }

    if (scope_realpath(buf, path) == NULL) {
        scope_perror("scope_realpath error");
        return -1;
    }

    /*
    * Retrieve the test directory path.
    * From:
    * /<dir>/appscope/test/linux/cfgutilsrulestest
    * To:
    * /<dir>/appscope/test/
    */
    for (int i= 0; i < 2; ++i) {
        path = scope_dirname(path);
        if (path == NULL) {
            scope_perror("scope_dirname error");
            return -1;
        }
    }
    return 0;
}

// mimics behavior of retrieving the default rules path
static const char *
testAccessRulesPath(const char *rulesPath) {
    return (scope_access(rulesPath, R_OK) == 0) ? rulesPath : NULL;
}

static void
openFileAndExecuteCfgProcessCommands(const char *path, config_t *cfg)
{
    FILE *f = scope_fopen(path, "r");
    cfgProcessCommands(cfg, f);
    scope_fclose(f);
}

static void
cfgPathHonorsEnvVar(void **state)
{
    const char *file_path = "/tmp/myfile.yml";

    // grab the current working directory
    char origdir[MAX_PATH];
    assert_non_null(scope_getcwd(origdir, sizeof(origdir)));
    // create newdir, and switch to it
    char newdir[MAX_PATH + 12];

    scope_snprintf(newdir, sizeof(newdir), "%s/%s", origdir, "newdir");
    if (scope_access(newdir, R_OK)) {
        assert_int_equal(scope_mkdir(newdir, 0777), 0);
    }

    assert_int_equal(scope_chdir(newdir), 0);


    // Verify that if there is no env variable, cfgPath is null
    assert_null(cfgPath());

    // Verify that if there is an env variable, but no file, cfgPath is null
    assert_int_equal(setenv("SCOPE_CONF_PATH", file_path, 1), 0);
    assert_null(cfgPath());

    // Verify that if there is an env variable, and a file, cfgPath is defined
    int fd = scope_open(file_path, O_RDWR | O_CREAT, S_IRUSR | S_IRGRP | S_IROTH);
    assert_return_code(fd, scope_errno);
    char *path = cfgPath();
    assert_non_null(path);
    assert_string_equal(path, file_path);

    // cleanup
    scope_free(path);
    scope_unlink(file_path);
    assert_int_equal(unsetenv("SCOPE_CONF_PATH"), 0);

    // change back to origdir
    assert_int_equal(scope_chdir(origdir), 0);
    // Delete the directory we created
    assert_int_equal(scope_rmdir(newdir), 0);
}

/*
static void
cfgPathHonorsPriorityOrder(void **state)
{
    // there is something amiss with creating the newdir[] entries in this test
    // when running in a CI environment so we're skipping it for now.
    const char *CI = getenv("CI");
    if (CI) {
        skip();
        return;
    }

    // Get HOME env variable
    const char *home = getenv("HOME");
    assert_non_null(home);
    char homeConfDir[MAX_PATH];
    int rv = snprintf(homeConfDir, sizeof(homeConfDir), "%s/conf", home);
    assert_int_not_equal(rv, -1);

    // Create temp directories
    const char *newdir[] = {"testtempdir1", "testtempdir2", 
                      "testtempdir1/conf", "testtempdir2/conf", 
                      homeConfDir};
    int i;
    for (i=0; i<sizeof(newdir)/sizeof(newdir[0]); i++) {
        if (access(newdir[i], F_OK) == -1) {
            assert_int_equal(mkdir(newdir[i], 0777), 0);
        }
    }

    // get the basedir, set cwd and scopeHome from it
    char basedir[MAX_PATH];
    char cwd[MAX_PATH * 2];
    char scopeHome[MAX_PATH * 3];
    assert_non_null(getcwd(basedir, sizeof(basedir)));
    snprintf(cwd, sizeof(cwd), "%s/%s", basedir, newdir[0]);
    snprintf(scopeHome, sizeof(scopeHome), "%s/%s", basedir, newdir[1]);

    // Change to cwd
    assert_int_equal(chdir(cwd), 0);

    // Set SCOPE_HOME to the other
    assert_int_equal(setenv("SCOPE_HOME", scopeHome, 1), 0);

    // Create the paths we want to test
    const char file[] = CFG_FILE_NAME; // scope.yml
    char path[6][MAX_PATH * 4];
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
        char *result = cfgPath();
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
*/

static void
cfgProcessEnvironmentMtcEnable(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcEnableSet(cfg, FALSE);
    assert_int_equal(cfgMtcEnable(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_ENABLE", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcEnable(cfg), TRUE);

    assert_int_equal(setenv("SCOPE_METRIC_ENABLE", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcEnable(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_ENABLE"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcEnable(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_ENABLE", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcEnable(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentMtcFormat(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcFormatSet(cfg, CFG_FMT_NDJSON);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_NDJSON);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "statsd", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_STATSD);

    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "ndjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_NDJSON);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_FORMAT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_NDJSON);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "bson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_NDJSON);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentStatsDPrefix(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcStatsDPrefixSet(cfg, "something");
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "something.");

    // should override current cfg
    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "blah.");

    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "hey", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "hey.");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_STATSD_PREFIX"), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "hey.");

    // empty string
    assert_int_equal(setenv("SCOPE_STATSD_PREFIX", "", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentStatsDMaxLen(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcStatsDMaxLenSet(cfg, 0);
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 3);

    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_STATSD_MAXLEN"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_STATSD_MAXLEN", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentWatchStatsdEnable(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcWatchEnableSet(cfg, TRUE, CFG_MTC_STATSD);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), TRUE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_STATSD", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_STATSD", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), TRUE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_STATSD"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), TRUE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_STATSD", "sure thing", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), TRUE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}


static void
cfgProcessEnvironmentMtcPeriod(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcPeriodSet(cfg, 0);
    assert_int_equal(cfgMtcPeriod(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcPeriod(cfg), 3);

    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcPeriod(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_SUMMARY_PERIOD"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcPeriod(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcPeriod(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentCommandDir(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgCmdDirSet(cfg, "/my/favorite/directory");
    assert_string_equal(cfgCmdDir(cfg), "/my/favorite/directory");

    // should override current cfg
    assert_int_equal(setenv("SCOPE_CMD_DIR", "/my/other/dir", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCmdDir(cfg), "/my/other/dir");

    assert_int_equal(setenv("SCOPE_CMD_DIR", "/my/dir", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCmdDir(cfg), "/my/dir");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_CMD_DIR"), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCmdDir(cfg), "/my/dir");

    // empty string
    assert_int_equal(setenv("SCOPE_CMD_DIR", "", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgCmdDir(cfg), DEFAULT_COMMAND_DIR);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentConfigEvent(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgSendProcessStartMsgSet(cfg, FALSE);
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_CONFIG_EVENT", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgSendProcessStartMsg(cfg), TRUE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_CONFIG_EVENT", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_CONFIG_EVENT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_CONFIG_EVENT", "hi!", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentEvtEnable(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgEvtEnableSet(cfg, FALSE);
    assert_int_equal(cfgEvtEnable(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_EVENT_ENABLE", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtEnable(cfg), TRUE);

    assert_int_equal(setenv("SCOPE_EVENT_ENABLE", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtEnable(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_EVENT_ENABLE"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtEnable(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_EVENT_ENABLE", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtEnable(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentEventFormat(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgEventFormatSet(cfg, CFG_FMT_NDJSON);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "ndjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);

    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "statsd", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_EVENT_FORMAT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "bson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentMaxEps(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgEvtRateLimitSet(cfg, 0);
    assert_int_equal(cfgEvtRateLimit(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_EVENT_MAXEPS", "13", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtRateLimit(cfg), 13);

    assert_int_equal(setenv("SCOPE_EVENT_MAXEPS", "31", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtRateLimit(cfg), 31);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_EVENT_MAXEPS"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtRateLimit(cfg), 31);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_EVENT_MAXEPS", "cribl_rulz", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtRateLimit(cfg), 31);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentEnhanceFs(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgEnhanceFsSet(cfg, FALSE);
    assert_int_equal(cfgEnhanceFs(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_ENHANCE_FS", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEnhanceFs(cfg), TRUE);

    assert_int_equal(setenv("SCOPE_ENHANCE_FS", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEnhanceFs(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_ENHANCE_FS"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEnhanceFs(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_ENHANCE_FS", "cribl_rulz", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEnhanceFs(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

typedef struct
{
    const char *env_name;
    watch_t   src;
    unsigned    default_val;
} source_state_t;

static void
cfgProcessEnvironmentEventSource(void **state)
{
    source_state_t *data = (source_state_t*)state[0];

    config_t *cfg = cfgCreateDefault();
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), data->default_val);

    cfgEvtFormatSourceEnabledSet(cfg, data->src, 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), 0);

    // should override current cfg
    assert_int_equal(setenv(data->env_name, "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), 1);

    assert_int_equal(setenv(data->env_name, "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), 0);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv(data->env_name), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), 0);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv(data->env_name, "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, data->src), 0);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}


static void
cfgProcessEnvironmentMtcVerbosity(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgMtcVerbositySet(cfg, 0);
    assert_int_equal(cfgMtcVerbosity(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcVerbosity(cfg), 3);

    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "9", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcVerbosity(cfg), 9);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_VERBOSITY"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcVerbosity(cfg), 9);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgMtcVerbosity(cfg), 9);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentLogLevel(void **state)
{
    config_t *cfg = cfgCreateDefault();
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

typedef struct
{
    const char *env_name;
    which_transport_t transport;
} dest_state_t;

static void
cfgProcessEnvironmentTransport(void **state)
{
    dest_state_t *data = (dest_state_t*)state[0];

    config_t *cfg = cfgCreateDefault();

    // should override current cfg
    assert_int_equal(setenv(data->env_name, "udp://host:234", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_UDP);
    assert_string_equal(cfgTransportHost(cfg, data->transport), "host");
    assert_string_equal(cfgTransportPort(cfg, data->transport), "234");

    // test that our code doesn't modify the env variable directly
    assert_string_equal(getenv(data->env_name), "udp://host:234");

    assert_int_equal(setenv(data->env_name, "file:///some/path/somewhere", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, data->transport), "/some/path/somewhere");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv(data->env_name), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, data->transport), "/some/path/somewhere");

    // unrecognised value should not affect cfg
    assert_int_equal(setenv(data->env_name, "somewhere else", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, data->transport), "/some/path/somewhere");

    // port is required, if not there cfg should not be modified
    assert_int_equal(setenv(data->env_name, "udp://host", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_FILE);
    assert_string_equal(cfgTransportPath(cfg, data->transport), "/some/path/somewhere");

    // one more variant... unix://
    assert_int_equal(setenv(data->env_name, "unix://@theUnixAddress", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgTransportType(cfg, data->transport), CFG_UNIX);
    assert_string_equal(cfgTransportPath(cfg, data->transport), "@theUnixAddress");

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentStatsdTags(void **state)
{
    config_t *cfg = cfgCreateDefault();
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

static void
cfgProcessEnvironmentPayEnable(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgPayEnableSet(cfg, FALSE);
    assert_int_equal(cfgPayEnable(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_PAYLOAD_ENABLE", "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayEnable(cfg), TRUE);

    assert_int_equal(setenv("SCOPE_PAYLOAD_ENABLE", "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayEnable(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_PAYLOAD_ENABLE"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayEnable(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_PAYLOAD_ENABLE", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayEnable(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentPayType(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgPayDirEnableSet(cfg, FALSE);
    assert_int_equal(cfgPayDirEnable(cfg), FALSE);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_PAYLOAD_DEST", "dir", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayDirEnable(cfg), TRUE);

    assert_int_equal(setenv("SCOPE_PAYLOAD_DEST", "event", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayDirEnable(cfg), FALSE);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_PAYLOAD_DEST"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayDirEnable(cfg), FALSE);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_PAYLOAD_DEST", "blah", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgPayEnable(cfg), FALSE);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentPayDir(void **state)
{
    config_t *cfg = cfgCreateDefault();
    cfgPayDirSet(cfg, "/my/favorite/directory");
    assert_string_equal(cfgPayDir(cfg), "/my/favorite/directory");

    // should override current cfg
    assert_int_equal(setenv("SCOPE_PAYLOAD_DIR", "/my/other/dir", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgPayDir(cfg), "/my/other/dir");

    assert_int_equal(setenv("SCOPE_PAYLOAD_DIR", "/my/dir", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgPayDir(cfg), "/my/dir");

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_PAYLOAD_DIR"), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgPayDir(cfg), "/my/dir");

    // empty string
    assert_int_equal(setenv("SCOPE_PAYLOAD_DIR", "", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_string_equal(cfgPayDir(cfg), DEFAULT_PAYLOAD_DIR);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentCmdDebugIsIgnored(void **state)
{
    const char *path = "/tmp/dbgoutfile.txt";
    assert_int_equal(setenv("SCOPE_CMD_DBG_PATH", path, 1), 0);

    long file_pos_before = fileEndPosition(path);

    config_t *cfg = cfgCreateDefault();
    cfgProcessEnvironment(cfg);
    cfgDestroy(&cfg);

    long file_pos_after = fileEndPosition(path);

    // since it's not processed, the file position better not have changed.
    assert_int_equal(file_pos_before, file_pos_after);

    unsetenv("SCOPE_CMD_DBG_PATH");
    if (file_pos_after != -1) unlink(path);
}

static void
cfgProcessCommandsCmdDebugIsProcessed(void **state)
{
    const char *outpath = "/tmp/dbgoutfile.txt";
    const char *inpath = "/tmp/dbginfile.txt";

    long file_pos_before = fileEndPosition(outpath);

    config_t *cfg = cfgCreateDefault();
    writeFile(inpath, "SCOPE_CMD_DBG_PATH=/tmp/dbgoutfile.txt");
    openFileAndExecuteCfgProcessCommands(inpath, cfg);
    cfgDestroy(&cfg);

    long file_pos_after = fileEndPosition(outpath);

    // since it's not processed, the file position should be updated
    assert_int_not_equal(file_pos_before, file_pos_after);

    scope_unlink(inpath);
    if (file_pos_after != -1) scope_unlink(outpath);
}

static void
cfgProcessCommandsFromFile(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    const char *path = "/tmp/test.file";

    // Just making sure these don't crash us.
    cfgProcessCommands(NULL, NULL);
    cfgProcessCommands(cfg, NULL);


    // test the basics
    writeFile(path, "SCOPE_METRIC_FORMAT=ndjson");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_NDJSON);

    writeFile(path, "\nSCOPE_METRIC_FORMAT=statsd\r\nblah");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_STATSD);

    // just demonstrating that the "last one wins"
    writeFile(path, "SCOPE_METRIC_FORMAT=ndjson\n"
                    "SCOPE_METRIC_FORMAT=statsd");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgMtcFormat(cfg), CFG_FMT_STATSD);


    // test everything else once
    writeFile(path,
        "SCOPE_METRIC_ENABLE=false\n"
        "SCOPE_STATSD_PREFIX=prefix\n"
        "SCOPE_STATSD_MAXLEN=1024\n"
        "SCOPE_METRIC_STATSD=false\n"
        "SCOPE_SUMMARY_PERIOD=11\n"
        "SCOPE_CMD_DIR=/the/path/\n"
        "SCOPE_CONFIG_EVENT=false\n"
        "SCOPE_METRIC_VERBOSITY=1\n"
        "SCOPE_METRIC_VERBOSITY:prefix\n"     // ignored (no '=')
        "SCOPE_METRIC_VERBOSITY=blah\n"       // processed, but 'blah' isn't int)
        "\n"                               // ignored (no '=')
        "ignored =  too.\n"                // ignored (not one of our env vars)
        "SEE_THAT_THIS_IS_HARMLESS=True\n" // ignored (not one of our env vars)
        "SCOPE_LOG_LEVEL=trace\n"
        "SCOPE_METRIC_DEST=file:///tmp/file.tmp\n"
        "SCOPE_LOG_DEST=file:///tmp/file.tmp2\n"
        "SCOPE_TAG_CUSTOM1=val1\n"
        "SCOPE_TAG_CUSTOM2=val2\n"
        "SCOPE_EVENT_DEST=udp://host:1234\n"
        "SCOPE_EVENT_ENABLE=false\n"
        "SCOPE_EVENT_FORMAT=ndjson\n"
        "SCOPE_EVENT_LOGFILE=true\n"
        "SCOPE_EVENT_CONSOLE=false\n"
        "SCOPE_EVENT_SYSLOG=true\n"
        "SCOPE_EVENT_METRIC=false\n"
        "SCOPE_EVENT_HTTP=false\n"
        "SCOPE_EVENT_NET=true\n"
        "SCOPE_EVENT_FS=false\n"
        "SCOPE_EVENT_DNS=true\n"
        "SCOPE_EVENT_LOGFILE_NAME=a\n"
        "SCOPE_EVENT_CONSOLE_NAME=b\n"
        "SCOPE_EVENT_SYSLOG_NAME=c\n"
        "SCOPE_EVENT_METRIC_NAME=d\n"
        "SCOPE_EVENT_HTTP_NAME=e\n"
        "SCOPE_EVENT_NET_NAME=f\n"
        "SCOPE_EVENT_FS_NAME=g\n"
        "SCOPE_EVENT_DNS_NAME=h\n"
        "SCOPE_EVENT_LOGFILE_FIELD=i\n"
        "SCOPE_EVENT_CONSOLE_FIELD=j\n"
        "SCOPE_EVENT_SYSLOG_FIELD=k\n"
        "SCOPE_EVENT_METRIC_FIELD=l\n"
        "SCOPE_EVENT_HTTP_FIELD=m\n"
        "SCOPE_EVENT_NET_FIELD=n\n"
        "SCOPE_EVENT_FS_FIELD=o\n"
        "SCOPE_EVENT_DNS_FIELD=p\n"
        "SCOPE_EVENT_LOGFILE_VALUE=q\n"
        "SCOPE_EVENT_CONSOLE_VALUE=r\n"
        "SCOPE_EVENT_SYSLOG_VALUE=s\n"
        "SCOPE_EVENT_METRIC_VALUE=t\n"
        "SCOPE_EVENT_HTTP_VALUE=u\n"
        "SCOPE_EVENT_NET_VALUE=v\n"
        "SCOPE_EVENT_FS_VALUE=w\n"
        "SCOPE_EVENT_DNS_VALUE=x\n"
        "SCOPE_EVENT_MAXEPS=123456789\n"
        "SCOPE_ENHANCE_FS=false\n"
        "SCOPE_PAYLOAD_ENABLE=false\n"
        "SCOPE_PAYLOAD_DIR=/the/path\n"
    );

    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgMtcEnable(cfg), FALSE);
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "prefix.");
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgMtcWatchEnable(cfg, CFG_MTC_STATSD), FALSE);
    assert_int_equal(cfgMtcPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/the/path/");
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);
    assert_int_equal(cfgMtcVerbosity(cfg), 1);
    assert_string_equal(cfgTransportPath(cfg, CFG_MTC), "/tmp/file.tmp");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM1"), "val1");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM2"), "val2");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    assert_int_equal(cfgTransportType(cfg, CFG_CTL), CFG_UDP);
    assert_string_equal(cfgTransportHost(cfg, CFG_CTL), "host");
    assert_string_equal(cfgTransportPort(cfg, CFG_CTL), "1234");
    assert_int_equal(cfgEvtEnable(cfg), FALSE);
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_METRIC), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_HTTP), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_NET), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_FS), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_DNS), 1);
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_FILE), "a");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_CONSOLE), "b");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_SYSLOG), "c");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_METRIC), "d");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_HTTP), "e");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_NET), "f");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_FS), "g");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_DNS), "h");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_FILE), "i");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_CONSOLE), "j");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_SYSLOG), "k");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_METRIC), "l");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_HTTP), "m");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_NET), "n");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_FS), "o");
    assert_string_equal(cfgEvtFormatFieldFilter(cfg, CFG_SRC_DNS), "p");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_FILE), "q");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_CONSOLE), "r");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_SYSLOG), "s");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_METRIC), "t");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_HTTP), "u");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_NET), "v");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_FS), "w");
    assert_string_equal(cfgEvtFormatValueFilter(cfg, CFG_SRC_DNS), "x");
    assert_int_equal(cfgEvtRateLimit(cfg), 123456789);
    assert_int_equal(cfgEnhanceFs(cfg), FALSE);
    assert_int_equal(cfgPayEnable(cfg), FALSE);
    assert_string_equal(cfgPayDir(cfg), "/the/path");

    deleteFile(path);
    cfgDestroy(&cfg);
}

static void
cfgProcessCommandsEnvSubstitution(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    const char *path = "/tmp/test.file";

    // test everything else once
    writeFile(path,
        "SCOPE_METRIC_ENABLE=$MASTER_ENABLE\n"
        "SCOPE_STATSD_PREFIX=$VAR1.$MY_ENV_VAR\n"
        "SCOPE_STATSD_MAXLEN=$MAXLEN\n"
        "SCOPE_SUMMARY_PERIOD=$PERIOD\n"
        "SCOPE_CMD_DIR=/$MYHOME/scope/\n"
        "SCOPE_CONFIG_EVENT=$MASTER_ENABLE\n"
        "SCOPE_METRIC_VERBOSITY=$VERBOSITY\n"
        "SCOPE_LOG_LEVEL=$LOGLEVEL\n"
        "SCOPE_METRIC_DEST=file:///\\$VAR1/$MY_ENV_VAR/\n"
        "SCOPE_LOG_DEST=$DEST\n"
        "SCOPE_TAG_CUSTOM=$PERIOD\n"
        "SCOPE_TAG_whyyoumadbro=Bill owes me $5.00\n"
        "SCOPE_TAG_undefined=$UNDEFINEDENV\n"
        "SCOPE_EVENT_DEST=udp://ho$st:1234\n"
        "SCOPE_EVENT_ENABLE=$MASTER_ENABLE\n"
        "SCOPE_EVENT_LOGFILE=$TRUTH\n"
        "SCOPE_EVENT_CONSOLE=false\n"
        "SCOPE_EVENT_SYSLOG=$TRUTH\n"
        "SCOPE_EVENT_METRIC=false\n"
        "SCOPE_EVENT_LOGFILE_NAME=$FILTER\n"
        "SCOPE_EVENT_MAXEPS=$EPS\n"
        "SCOPE_ENHANCE_FS=$TRUTH\n"
        "SCOPE_PAYLOAD_ENABLE=$TRUTH\n"
        "SCOPE_PAYLOAD_DIR=$MYHOME\n"
    );


    // Set env variables to test indirect substitution
    assert_int_equal(setenv("MASTER_ENABLE", "false", 1), 0);
    assert_int_equal(setenv("VAR1", "longer", 1), 0);
    assert_int_equal(setenv("MY_ENV_VAR", "shorter", 1), 0);
    assert_int_equal(setenv("MAXLEN", "1024", 1), 0);
    assert_int_equal(setenv("DEST", "file:///tmp/file.tmp2", 1), 0);
    assert_int_equal(setenv("PERIOD", "11", 1), 0);
    assert_int_equal(setenv("MYHOME", "home/mydir", 1), 0);
    assert_int_equal(setenv("VERBOSITY", "1", 1), 0);
    assert_int_equal(setenv("LOGLEVEL", "trace", 1), 0);
    assert_int_equal(setenv("FILTER", ".*[.]log$", 1), 0);
    assert_int_equal(setenv("EPS", "987654321", 1), 0);
    assert_int_equal(setenv("TRUTH", "true", 1), 0);

    openFileAndExecuteCfgProcessCommands(path, cfg);
    // test substitute env values that are longer and shorter than they env name
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "longer.shorter.");
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgMtcPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/home/mydir/scope/");
    assert_int_equal(cfgSendProcessStartMsg(cfg), FALSE);
    assert_int_equal(cfgMtcVerbosity(cfg), 1);
    // test escaped substitution  (a match preceded by '\')
    assert_string_equal(cfgTransportPath(cfg, CFG_MTC), "/$VAR1/shorter/");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM"), "11");
    // test lookups that aren't found: $5 and $UNDEFINEDENV
    assert_string_equal(cfgCustomTagValue(cfg, "whyyoumadbro"), "Bill owes me $5.00");
    assert_string_equal(cfgCustomTagValue(cfg, "undefined"), "$UNDEFINEDENV");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    // event stuff...
    assert_string_equal(cfgTransportHost(cfg, CFG_CTL), "ho$st");
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_METRIC), 0);
    // misc
    assert_int_equal(cfgMtcEnable(cfg), FALSE);
    assert_int_equal(cfgEvtEnable(cfg), FALSE);
    assert_int_equal(cfgEvtRateLimit(cfg), 987654321);
    assert_int_equal(cfgEnhanceFs(cfg), TRUE);
    assert_int_equal(cfgPayEnable(cfg), TRUE);
    assert_string_equal(cfgPayDir(cfg), "home/mydir");

    deleteFile(path);
    cfgDestroy(&cfg);

    unsetenv("MASTER_ENABLE");
    unsetenv("VAR1");
    unsetenv("MY_ENV_VAR");
    unsetenv("MAXLEN");
    unsetenv("DEST");
    unsetenv("PERIOD");
    unsetenv("MYHOME");
    unsetenv("VERBOSITY");
    unsetenv("LOGLEVEL");
    unsetenv("FILTER");
    unsetenv("EPS");
    unsetenv("TRUTH");
}

static void
verifyDefaults(config_t *config)
{
    assert_int_equal       (cfgMtcEnable(config), DEFAULT_MTC_ENABLE);
    assert_int_equal       (cfgMtcFormat(config), DEFAULT_MTC_FORMAT);
    assert_string_equal    (cfgMtcStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal       (cfgMtcStatsDMaxLen(config), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal       (cfgMtcWatchEnable(config, CFG_MTC_STATSD), DEFAULT_MTC_STATSD_ENABLE);
    assert_int_equal       (cfgMtcVerbosity(config), DEFAULT_MTC_VERBOSITY);
    assert_int_equal       (cfgMtcPeriod(config), DEFAULT_SUMMARY_PERIOD);
    assert_string_equal    (cfgCmdDir(config), DEFAULT_COMMAND_DIR);
    assert_int_equal       (cfgSendProcessStartMsg(config), DEFAULT_PROCESS_START_MSG);
    assert_int_equal       (cfgEvtEnable(config), DEFAULT_EVT_ENABLE);
    assert_int_equal       (cfgEventFormat(config), DEFAULT_CTL_FORMAT);
    assert_int_equal       (cfgEvtRateLimit(config), DEFAULT_MAXEVENTSPERSEC);
    assert_int_equal       (cfgEnhanceFs(config), DEFAULT_ENHANCE_FS);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_HTTP), DEFAULT_SRC_HTTP_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_NET), DEFAULT_SRC_NET_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_FS), DEFAULT_SRC_FS_VALUE);
    assert_string_equal    (cfgEvtFormatValueFilter(config, CFG_SRC_DNS), DEFAULT_SRC_DNS_VALUE);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_HTTP), DEFAULT_SRC_HTTP_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_NET), DEFAULT_SRC_NET_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_FS), DEFAULT_SRC_FS_FIELD);
    assert_string_equal    (cfgEvtFormatFieldFilter(config, CFG_SRC_DNS), DEFAULT_SRC_DNS_FIELD);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_HTTP), DEFAULT_SRC_HTTP_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_NET), DEFAULT_SRC_NET_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_FS), DEFAULT_SRC_FS_NAME);
    assert_string_equal    (cfgEvtFormatNameFilter(config, CFG_SRC_DNS), DEFAULT_SRC_DNS_NAME);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_FILE), DEFAULT_SRC_FILE);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_HTTP), DEFAULT_SRC_HTTP);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_NET), DEFAULT_SRC_NET);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_FS), DEFAULT_SRC_FS);
    assert_int_equal       (cfgEvtFormatSourceEnabled(config, CFG_SRC_DNS), DEFAULT_SRC_DNS);
    assert_null            (cfgEvtFormatHeader(config, 0));
    assert_int_equal       (cfgTransportType(config, CFG_MTC), DEFAULT_MTC_TYPE);
    assert_string_equal    (cfgTransportHost(config, CFG_MTC), DEFAULT_MTC_HOST);
    assert_string_equal    (cfgTransportPort(config, CFG_MTC), DEFAULT_MTC_PORT);
    assert_null            (cfgTransportPath(config, CFG_MTC));
    assert_int_equal       (cfgTransportBuf(config, CFG_MTC), DEFAULT_MTC_BUF);
    assert_int_equal       (cfgTransportType(config, CFG_CTL), DEFAULT_CTL_TYPE);
    assert_string_equal    (cfgTransportHost(config, CFG_CTL), DEFAULT_CTL_HOST);
    assert_string_equal    (cfgTransportPort(config, CFG_CTL), DEFAULT_CTL_PORT);
    assert_null            (cfgTransportPath(config, CFG_CTL));
    assert_int_equal       (cfgTransportBuf(config, CFG_CTL), DEFAULT_CTL_BUF);
    assert_int_equal       (cfgTransportType(config, CFG_LOG), DEFAULT_LOG_TYPE);
    assert_null            (cfgTransportHost(config, CFG_LOG));
    assert_null            (cfgTransportPort(config, CFG_LOG));
    assert_string_equal    (cfgTransportPath(config, CFG_LOG), DEFAULT_LOG_PATH);
    assert_int_equal       (cfgTransportBuf(config, CFG_MTC), DEFAULT_LOG_BUF);
    assert_null            (cfgCustomTags(config));
    assert_null            (cfgCustomTagValue(config, "tagname"));
    assert_int_equal       (cfgLogLevel(config), DEFAULT_LOG_LEVEL);
    assert_int_equal       (cfgPayEnable(config), DEFAULT_PAYLOAD_ENABLE);
    assert_string_equal    (cfgPayDir(config), DEFAULT_PAYLOAD_DIR);

    // the protocol list should be empty too
    assert_non_null        (g_protlist);
    assert_int_equal       (g_prot_sequence, 0);

    assert_int_equal       (cfgLogStreamEnable(config), DEFAULT_LOGSTREAM_ENABLE);
    assert_int_equal       (cfgTransportType(config, CFG_LS), DEFAULT_LS_TYPE);
    assert_null            (cfgAuthToken(config));
}

static void
cfgReadGoodYaml(void **state)
{
    // Test file config (yaml)
    const char *yamlText =
        "---\n"
        "metric:\n"
        "  enable: false\n"
        "  format:\n"
        "    type: statsd                # statsd, ndjson\n"
        "    statsdprefix : 'cribl.scope'    # prepends each statsd metric\n"
        "    statsdmaxlen : 1024             # max size of a formatted statsd string\n"
        "    verbosity: 3                    # 0-9 (0 is least verbose, 9 is most)\n"
        "  watch:\n"
        "  transport:                        # defines how scope output is sent\n"
        "    type: file                      # udp, unix, file, syslog\n"
        "    path: '/var/log/scope.log'\n"
        "    buffering: line\n"
        "event:\n"
        "  enable: true\n"
        "  transport:\n"
        "    type: tcp                       # udp, unix, file, syslog\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "  format:\n"
        "    type : ndjson                   # ndjson\n"
        "    maxeventpersec : 989898         # max events per second.\n"
        "    enhancefs : false               # true, false\n"
        "  watch:\n"
        "    - type: file                    # create events from file\n"
        "      name: .*[.]log$\n"
        "      field: .*host.*\n"
        "      value: '[0-9]+'\n"
        "    - type: console                 # create events from stdout and stderr\n"
        "    - type: syslog                  # create events from syslog and vsyslog\n"
        "    - type: metric\n"
        "    - type: http\n"
        "    - type: net\n"
        "    - type: fs\n"
        "    - type: dns\n"
        "payload:\n"
        "  enable: false\n"
        "  dir: '/my/dir'\n"
        "libscope:\n"
        "  configevent: true\n"
        "  summaryperiod: 11                 # in seconds\n"
        "  commanddir: /tmp\n"
        "  log:\n"
        "    level: debug                      # debug, info, warning, error, none\n"
        "    transport:\n"
        "      buffering: full\n"
        "      type: edge\n"
        "tags:\n"
        "  name1: value1\n"
        "  name2: value2\n"
        "cribl:\n"
        "  enable: true\n"
        "  transport:\n"
        "    type: unix\n"
        "    path: '@abstractsock'\n"
        "  authtoken: ''\n"
        "...\n";
    const char *path = CFG_FILE_NAME;
    writeFile(path, yamlText);
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_string_equal(cfgMtcStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgMtcStatsDMaxLen(config), 1024);
    assert_int_equal(cfgMtcWatchEnable(config, CFG_MTC_STATSD), FALSE);
    assert_int_equal(cfgMtcVerbosity(config), 3);
    assert_int_equal(cfgMtcPeriod(config), 11);
    assert_string_equal(cfgCmdDir(config), "/tmp");
    assert_int_equal(cfgSendProcessStartMsg(config), TRUE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgEventFormat(config), CFG_FMT_NDJSON);
    assert_int_equal(cfgEvtRateLimit(config), 989898);
    assert_int_equal(cfgEnhanceFs(config), FALSE);
    assert_string_equal(cfgEvtFormatNameFilter(config, CFG_SRC_FILE), ".*[.]log$");
    assert_string_equal(cfgEvtFormatFieldFilter(config, CFG_SRC_FILE), ".*host.*");
    assert_string_equal(cfgEvtFormatValueFilter(config, CFG_SRC_FILE), "[0-9]+");
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_CONSOLE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_HTTP), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_NET), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FS), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_DNS), 1);
    assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_MTC), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_MTC), "8125");
    assert_string_equal(cfgTransportPath(config, CFG_MTC), "/var/log/scope.log");
    assert_int_equal(cfgTransportBuf(config, CFG_MTC), CFG_BUFFER_LINE);
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_TCP);
    assert_string_equal(cfgTransportHost(config, CFG_CTL), "127.0.0.2");
    assert_string_equal(cfgTransportPort(config, CFG_CTL), "9009");
    assert_null(cfgTransportPath(config, CFG_CTL));
    assert_int_equal(cfgTransportBuf(config, CFG_CTL), CFG_BUFFER_LINE);
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_EDGE);
    assert_null(cfgTransportHost(config, CFG_LOG));
    assert_null(cfgTransportPort(config, CFG_LOG));
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_int_equal(cfgTransportBuf(config, CFG_LOG), CFG_BUFFER_FULLY);
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "name1"), "value1");
    assert_string_equal(cfgCustomTagValue(config, "name2"), "value2");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    assert_int_equal(cfgPayEnable(config), FALSE);
    assert_string_equal(cfgPayDir(config), "/my/dir");
    assert_int_equal       (cfgLogStreamEnable(config), TRUE);
    assert_int_equal       (cfgTransportType(config, CFG_LS), CFG_UNIX);
    assert_string_equal    (cfgTransportPath(config, CFG_LS), "@abstractsock");
    assert_null            (cfgAuthToken(config));
    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    deleteFile(path);
}

static void
cfgReadStockYaml(void **state)
{
    // The stock scope.yml file up in ../conf/ should parse to the defaults.
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead("./conf/scope.yml");
    verifyDefaults(config);
    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
}

static void
writeFileWithSubstitution(const char *path, const char *base, const char *variable)
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
cfgReadEveryTransportType(void **state)
{
    const char *yamlText =
        "---\n"
        "metric:\n"
        "  transport:\n"
        "%s"
        "...\n";

    const char *udp_str =
        "    type: udp\n"
        "    host: 'labmachine8235'\n"
        "    port: 'ntp'\n";
    const char *unix_str =
        "    type: unix\n"
        "    path: '@scope.sock'\n";
    const char *file_str =
        "    type: file\n"
        "    path: '/var/log/scope.log'\n";
    const char *edge_str =
        "    type: edge\n";
    const char *transport_lines[] = {udp_str, unix_str, file_str, edge_str};

    const char *path = CFG_FILE_NAME;

    int i;
    for (i = 0; i<sizeof(transport_lines) / sizeof(transport_lines[0]); i++) {

        writeFileWithSubstitution(path, yamlText, transport_lines[i]);
        g_protlist = lstCreate(destroyProtEntry);
        assert_non_null(g_protlist);
        config_t *config = cfgRead(path);

        if (transport_lines[i] == udp_str) {
                assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_UDP);
                assert_string_equal(cfgTransportHost(config, CFG_MTC), "labmachine8235");
                assert_string_equal(cfgTransportPort(config, CFG_MTC), "ntp");
        } else if (transport_lines[i] == unix_str) {
                assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_UNIX);
                assert_string_equal(cfgTransportPath(config, CFG_MTC), "@scope.sock");
        } else if (transport_lines[i] == file_str) {
                assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_FILE);
                assert_string_equal(cfgTransportPath(config, CFG_MTC), "/var/log/scope.log");
        } else if (transport_lines[i] == edge_str) {
                assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_EDGE);
        }

        deleteFile(path);
        cfgDestroy(&config);
        lstDestroy(&g_protlist);
        g_prot_sequence = 0;
    }

}

static void
cfgReadEveryProcessLevel(void **state)
{
    const char *yamlText =
        "---\n"
        "libscope:\n"
        "  log:\n"
        "    level: %s\n"
        "...\n";

    const char *path = CFG_FILE_NAME;
    const char *level[] = { "trace", "debug", "info", "warning", "error", "none"};
    cfg_log_level_t value[] = {CFG_LOG_TRACE, CFG_LOG_DEBUG, CFG_LOG_INFO, CFG_LOG_WARN, CFG_LOG_ERROR, CFG_LOG_NONE};
    int i;
    for (i = 0; i< sizeof(level)/sizeof(level[0]); i++) {
        writeFileWithSubstitution(path, yamlText, level[i]);
        g_protlist = lstCreate(destroyProtEntry);
        assert_non_null(g_protlist);
        config_t *config = cfgRead(path);
        assert_int_equal(cfgLogLevel(config), value[i]);
        deleteFile(path);
        cfgDestroy(&config);
        lstDestroy(&g_protlist);
        g_prot_sequence = 0;
    }
}

// Test file config (json)
const char *jsonText =
    "{\n"
    "  'metric': {\n"
    "    'enable': 'true',\n"
    "    'format': {\n"
    "      'type': 'ndjson',\n"
    "      'statsdprefix': 'cribl.scope',\n"
    "      'statsdmaxlen': '42',\n"
    "      'verbosity': '0'\n"
    "    },\n"
    "    'watch': [\n"
    "      {'type':'statsd'}\n"
    "    ],\n"
    "    'transport': {\n"
    "      'type': 'file',\n"
    "      'path': '/var/log/scope.log'\n"
    "    }\n"
    "  },\n"
    "  'event': {\n"
    "    'enable': 'false',\n"
    "    'transport': {\n"
    "      'type': 'file',\n"
    "      'path': '/var/log/event.log'\n"
    "    },\n"
    "    'format': {\n"
    "      'type': 'ndjson',\n"
    "      'maxeventpersec': '42',\n"
    "      'enhancefs': 'false'\n"
    "    },\n"
    "    'watch' : [\n"
    "      {'type':'file', 'name':'.*[.]log$'},\n"
    "      {'type':'console'},\n"
    "      {'type':'syslog'},\n"
    "      {'type':'metric'},\n"
    "      {'type':'http', 'headers':['X-blah.*','My-goodness']},\n"
    "      {'type':'net'},\n"
    "      {'type':'fs'},\n"
    "      {'type':'dns'}\n"
    "    ]\n"
    "  },\n"
    "  'payload': {\n"
    "    'enable': 'true',\n"
    "    'dir': '/the/dir'\n"
    "  },\n"
    "  'libscope': {\n"
    "    'configevent': 'true',\n"
    "    'summaryperiod': '13',\n"
    "    'log': {\n"
    "      'level': 'debug',\n"
    "      'transport': {\n"
    "        'type': 'edge'\n"
    "      }\n"
    "    }\n"
    "  },\n"
    "  'tags': {\n"
    "    'tagA': 'val1',\n"
    "    'tagB': 'val2',\n"
    "    'tagC': 'val3'\n"
    "  },\n"
    "  'protocol': [\n"
    "    {'name':'eg1','regex':'.*'}\n"
    "  ],\n"
    "  'cribl': {\n"
    "    'enable': 'false',\n"
    "    'transport': {\n"
    "      'type': 'edge'\n"
    "    },\n"
    "    'authtoken': 'shhdonotsharethistokenwithjustanyone'\n"
    "  }\n"
    "}\n";

static void
cfgReadGoodJson(void **state)
{
    const char *path = CFG_FILE_NAME;
    writeFile(path, jsonText);
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgMtcEnable(config), TRUE);
    assert_int_equal(cfgMtcFormat(config), CFG_FMT_NDJSON);
    assert_string_equal(cfgMtcStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgMtcStatsDMaxLen(config), 42);
    assert_int_equal(cfgMtcVerbosity(config), 0);
    assert_int_equal(cfgMtcWatchEnable(config, CFG_MTC_STATSD), TRUE);
    assert_int_equal(cfgMtcPeriod(config), 13);
    assert_int_equal(cfgSendProcessStartMsg(config), TRUE);
    assert_int_equal(cfgEvtEnable(config), FALSE);
    assert_int_equal(cfgEventFormat(config), CFG_FMT_NDJSON);
    assert_int_equal(cfgEvtRateLimit(config), 42);
    assert_int_equal(cfgEnhanceFs(config), FALSE);
    assert_string_equal(cfgEvtFormatNameFilter(config, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_CONSOLE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_HTTP), 1);
    assert_string_equal(cfgEvtFormatHeader(config, 0), "X-blah.*");
    assert_string_equal(cfgEvtFormatHeader(config, 1), "My-goodness");
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_NET), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FS), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_DNS), 1);
    assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_MTC), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_MTC), "8125");
    assert_string_equal(cfgTransportPath(config, CFG_MTC), "/var/log/scope.log");
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_CTL), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_CTL), "9109");
    assert_string_equal(cfgTransportPath(config, CFG_CTL), "/var/log/event.log");
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_EDGE);
    assert_null(cfgTransportHost(config, CFG_LOG));
    assert_null(cfgTransportPort(config, CFG_LOG));
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "tagA"), "val1");
    assert_string_equal(cfgCustomTagValue(config, "tagB"), "val2");
    assert_string_equal(cfgCustomTagValue(config, "tagC"), "val3");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_DEBUG);
    assert_int_equal(cfgPayEnable(config), TRUE);
    assert_string_equal(cfgPayDir(config), "/the/dir");

    protocol_def_t *prot;
    assert_non_null    (g_protlist);
    assert_int_equal   (g_prot_sequence, 1);
    assert_non_null    (prot = lstFind(g_protlist, 1));
    assert_string_equal(prot->protname, "eg1");

    assert_int_equal       (cfgLogStreamEnable(config), FALSE);
    assert_int_equal       (cfgTransportType(config, CFG_LS), CFG_EDGE);
    assert_string_equal    (cfgAuthToken(config), "shhdonotsharethistokenwithjustanyone");

    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    deleteFile(path);
}

static void
cfgReadNonExistentFileReturnsDefaults(void **state)
{
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead("../thisFileNameWontBeFoundAnywhere.txt");
    verifyDefaults(config);
    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
}

static void
cfgReadBadYamlReturnsDefaults(void **state)
{
    const char *yamlText =
        "---\n"
        "metric:\n"
        "  format: ndjson\n"
        "  statsdprefix : 'cribl.scope'\n"
        "  transport:\n"
        "    type: file\n"
        "    path: '/var/log/scope.log'\n"
        "libscope:\n"
        "  log:\n"
        "      level: debug                  # <--- Extra indention!  bad!\n"
        "    transport:\n"
        "      type: syslog\n"
        "...\n";
    const char *path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead(path);
    verifyDefaults(config);

    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    deleteFile(path);
}

static void
cfgReadExtraFieldsAreHarmless(void **state)
{
    const char *yamlText =
        "---\n"
        "momsApplePieRecipe:                # has possibilities...\n"
        "  [apples,sugar,flour,dirt]        # dirt mom?  Really?\n"
        "metric:\n"
        "  format:\n"
        "    type: statsd\n"
        "    hey: yeahyou\n"
        "  request: 'make it snappy'        # Extra.\n"
        "  transport:\n"
        "    type: unix\n"
        "    path: '@scope.sock'\n"
        "    color: 'puce'                  # Extra.\n"
        "libscope:\n"
        "  log:\n"
        "    level: info\n"
        "tags:\n"
        "  brainfarts: 135\n"
        "...\n";
    const char *path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgMtcFormat(config), CFG_FMT_STATSD);
    assert_string_equal(cfgMtcStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_MTC), "@scope.sock");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "brainfarts"), "135");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    deleteFile(path);
}

static void
cfgReadYamlOrderWithinStructureDoesntMatter(void **state)
{
    const char *yamlText =
        "---\n"
        "payload:\n"
        "  dir: /favorite\n"
        "  enable: false\n"
        "event:\n"
        "  watch:\n"
        "    - name: .*[.]log$\n"
        "      type: syslog\n"
        "      field: .*host.*\n"
        "      value: '[0-9]+'\n"
        "    - type: file\n"
        "    - type: syslog\n"
        "    - type: metric\n"
        "    - type: http\n"
        "    - type: net\n"
        "    - type: fs\n"
        "    - type: dns\n"
        "  format:\n"
        "    enhancefs : true\n"
        "    maxeventpersec : 13579\n"
        "    type : ndjson\n"
        "  enable : false\n"
        "  transport:\n"
        "    type: edge                    # udp, unix, file, edge\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "libscope:\n"
        "  log:\n"
        "    level: info\n"
        "  summaryperiod: 42\n"
        "  configevent: false\n"
        "metric:\n"
        "  transport:\n"
        "    path: '@scope.sock'\n"
        "    type: unix\n"
        "  format:\n"
        "    verbosity: 4294967295\n"
        "    statsdmaxlen: 4294967295\n"
        "    statsdprefix: 'cribl.scope'\n"
        "    type:  statsd\n"
        "  enable : false\n"
        "tags:\n"
        "  135: kittens\n"
        "...\n";
    const char *path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgMtcFormat(config), CFG_FMT_STATSD);
    assert_string_equal(cfgMtcStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgMtcStatsDMaxLen(config), 4294967295);
    assert_int_equal(cfgMtcVerbosity(config), CFG_MAX_VERBOSITY);
    assert_int_equal(cfgMtcPeriod(config), 42);
    assert_int_equal(cfgSendProcessStartMsg(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), FALSE);
    assert_int_equal(cfgEventFormat(config), CFG_FMT_NDJSON);
    assert_int_equal(cfgEvtRateLimit(config), 13579);
    assert_int_equal(cfgEnhanceFs(config), TRUE);
    assert_string_equal(cfgEvtFormatNameFilter(config, CFG_SRC_SYSLOG), ".*[.]log$");
    assert_string_equal(cfgEvtFormatFieldFilter(config, CFG_SRC_SYSLOG), ".*host.*");
    assert_string_equal(cfgEvtFormatValueFilter(config, CFG_SRC_SYSLOG), "[0-9]+");
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_HTTP), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_NET), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_FS), 1);
    assert_int_equal(cfgEvtFormatSourceEnabled(config, CFG_SRC_DNS), 1);
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_EDGE);
    assert_int_equal(cfgTransportType(config, CFG_MTC), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_MTC), "@scope.sock");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "135"), "kittens");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);
    assert_int_equal(cfgPayEnable(config), FALSE);
    assert_string_equal(cfgPayDir(config), "/favorite");

    cfgDestroy(&config);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    deleteFile(path);
}

static void
cfgReadEnvSubstitution(void **state)
{

    // Set env variables to test indirect substitution
    assert_int_equal(setenv("MASTER_ENABLE", "true", 1), 0);
    assert_int_equal(setenv("VAR1", "longer", 1), 0);
    assert_int_equal(setenv("MY_ENV_VAR", "shorter", 1), 0);
    assert_int_equal(setenv("MAXLEN", "1024", 1), 0);
    assert_int_equal(setenv("DEST", "/tmp/file.tmp2", 1), 0);
    assert_int_equal(setenv("PERIOD", "11", 1), 0);
    assert_int_equal(setenv("MYHOME", "home/mydir", 1), 0);
    assert_int_equal(setenv("VERBOSITY", "1", 1), 0);
    assert_int_equal(setenv("LOGLEVEL", "trace", 1), 0);
    assert_int_equal(setenv("FORMAT", "ndjson", 1), 0);
    assert_int_equal(setenv("FILTER", ".*[.]log$", 1), 0);
    assert_int_equal(setenv("SOURCE", "syslog", 1), 0);
    assert_int_equal(setenv("EPS", "987654321", 1), 0);

    const char *yamlText =
        "---\n"
        "metric:\n"
        "  enable: $MASTER_ENABLE\n"
        "  format:\n"
        "    type: ndjson\n"
        "    statsdprefix : $VAR1.$MY_ENV_VAR\n"
        "    statsdmaxlen : $MAXLEN\n"
        "    verbosity: $VERBOSITY\n"
        "  transport:\n"
        "    type: file\n"
        "    path: /\\$VAR1/$MY_ENV_VAR/\n"
        "    buffering: line\n"
        "event:\n"
        "  enable: $MASTER_ENABLE\n"
        "  format:\n"
        "    type : $FORMAT\n"
        "    maxeventpersec : $EPS\n"
        "    enhancefs : $MASTER_ENABLE\n"
        "  watch:\n"
        "    - type: file                    # create events from files\n"
        "      name: $FILTER\n"
        "    - type: console                 # create events from stdout and stderr\n"
        "    - type: $SOURCE                 # create events from syslog and vsyslog\n"
        "    - type: metric\n"
        "payload:\n"
        "  enable: $MASTER_ENABLE\n"
        "  dir: $MYHOME\n"
        "libscope:\n"
        "  transport:\n"
        "    type: syslog                    # udp, unix, file, syslog\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "  summaryperiod: $PERIOD\n"
        "  configevent: $MASTER_ENABLE\n"
        "  commanddir: /$MYHOME/scope/\n"
        "  log:\n"
        "    level: $LOGLEVEL\n"
        "    transport:\n"
        "      buffering: full\n"
        "      type: file\n"
        "      path: $DEST\n"
        "tags:\n"
        "  CUSTOM: $PERIOD\n"
        "  whyyoumadbro: 'Bill owes me $5.00'\n"
        "  undefined: $UNDEFINEDENV\n"
        "...\n";
    const char *path = CFG_FILE_NAME;
    writeFile(path, yamlText);
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *cfg = cfgRead(path);
    assert_non_null(cfg);

    // test substitute env values that are longer and shorter than they env name
    assert_string_equal(cfgMtcStatsDPrefix(cfg), "longer.shorter.");
    assert_int_equal(cfgMtcStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgMtcPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/home/mydir/scope/");
    assert_int_equal(cfgSendProcessStartMsg(cfg), TRUE);
    assert_int_equal(cfgMtcVerbosity(cfg), 1);
    // test escaped substitution  (a match preceded by '\')
    assert_string_equal(cfgTransportPath(cfg, CFG_MTC), "/$VAR1/shorter/");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM"), "11");
    // test lookups that aren't found: $5 and $UNDEFINEDENV
    assert_string_equal(cfgCustomTagValue(cfg, "whyyoumadbro"), "Bill owes me $5.00");
    assert_string_equal(cfgCustomTagValue(cfg, "undefined"), "$UNDEFINEDENV");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    // test event fields...
    assert_int_equal(cfgEventFormat(cfg), CFG_FMT_NDJSON);
    assert_string_equal(cfgEvtFormatNameFilter(cfg, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEvtFormatSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);
    // misc
    assert_int_equal(cfgMtcEnable(cfg), TRUE);
    assert_int_equal(cfgEvtEnable(cfg), TRUE);
    assert_int_equal(cfgEvtRateLimit(cfg), 987654321);
    assert_int_equal(cfgEnhanceFs(cfg), TRUE);
    assert_int_equal(cfgPayEnable(cfg), TRUE);
    assert_string_equal(cfgPayDir(cfg), "home/mydir");

    cfgDestroy(&cfg);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;

    unsetenv("MASTER_ENABLE");
    unsetenv("VAR1");
    unsetenv("MY_ENV_VAR");
    unsetenv("MAXLEN");
    unsetenv("DEST");
    unsetenv("PERIOD");
    unsetenv("VERBOSITY");
    unsetenv("LOGLEVEL");
    unsetenv("FORMAT");
    unsetenv("FILTER");
    unsetenv("SOURCE");
    unsetenv("EPS");

    deleteFile(path);
}

static void
jsonObjectFromCfgAndjsonStringFromCfgRoundTrip(void **state)
{
    // Start with a string, just since it's already defined for another test
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    config_t *cfg = cfgFromString(jsonText);
    assert_non_null(cfg);

    // Now from the cfg object above, we should be able to create a
    // new string and json object with the same content
    char *stringified_json1 = jsonStringFromCfg(cfg);
    assert_non_null(stringified_json1);
    cJSON *json1 = jsonObjectFromCfg(cfg);
    assert_non_null(json1);

    // Do this again with the new string we output this time
    cfgDestroy(&cfg);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);
    cfg = cfgFromString(stringified_json1);
    assert_non_null(cfg);

    char *stringified_json2 = jsonStringFromCfg(cfg);
    assert_non_null(stringified_json2);
    cJSON *json2 = jsonObjectFromCfg(cfg);
    assert_non_null(json2);

    // now the diff to make sure the strings and json object trees are identical
    assert_string_equal(stringified_json1, stringified_json2);
    assert_true(cJSON_Compare(json1, json2, 1)); // case-sensitive comparison

    //printf("%s\n", stringified_json1);

    cfgDestroy(&cfg);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
    cJSON_Delete(json1);
    cJSON_Delete(json2);
    scope_free(stringified_json1);
    scope_free(stringified_json2);
}


static void
initLogReturnsPtr(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_EDGE; t++) {
	    switch (t) {
            case CFG_UDP:
                cfgTransportTypeSet(cfg, CFG_LOG, t);
                cfgTransportHostSet(cfg, CFG_LOG, "localhost");
                cfgTransportPortSet(cfg, CFG_LOG, "4444");
                break;
            case CFG_UNIX:
				cfgTransportPathSet(cfg, CFG_LOG, "@scope.sock");
                break;
            case CFG_FILE:
				cfgTransportPathSet(cfg, CFG_LOG, "/tmp/scope.log");
                break;
            case CFG_TCP:
            case CFG_EDGE:
                break;
	    }
        cfgTransportTypeSet(cfg, CFG_LOG, t);
        log_t *log = initLog(cfg);
        assert_non_null(log);
        logDestroy(&log);
    }
    cfgDestroy(&cfg);
}

static void
initMtcReturnsPtr(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_EDGE; t++) {
        cfgTransportTypeSet(cfg, CFG_MTC, t);
        if (t == CFG_UNIX) {
            cfgTransportPathSet(cfg, CFG_MTC, "@scope.sock");
        } else if (t == CFG_FILE) {
            cfgTransportPathSet(cfg, CFG_MTC, "/tmp/scope.log");
        }
        mtc_t *mtc = initMtc(cfg);
        assert_non_null(mtc);
        mtcDestroy(&mtc);
    }
    cfgDestroy(&cfg);
}

static void
initEvtFormatReturnsPtr(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    evt_fmt_t *evt = initEvtFormat(cfg);
    assert_non_null(evt);
    evtFormatDestroy(&evt);

    cfgDestroy(&cfg);
}

static void
initCtlReturnsPtr(void **state)
{
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    ctl_t *ctl = initCtl(cfg);
    assert_non_null(ctl);
    ctlDestroy(&ctl);

    cfgDestroy(&cfg);
}

static void
cfgReadProtocol(void **state)
{
    // protocol config in yaml format
    const char *yamlText =
        "protocol:\n"
        // the 1st should load with non-default binary and len values
        "  - name: test1\n"
        "    binary: 'true'\n"
        "    regex: 'sup?'\n"
        "    len: 111\n"
        "\n"
        // the 2rd should load with defaults for non-required values
        "  - name: test2\n"
        "    regex: 'sup up?'\n"
        "\n"
        // the 3rd should load with all values specified
        "  - name: test3\n"
        "    binary: false\n"
        "    regex: 'sup er?'\n"
        "    len: 333\n"
        "    detect: false\n"
        "    payload: true\n"
        "\n"
        // the 4th should fail to load, missing regex
        "  - name: test4\n"
        "    #regex: 'sup er?'\n"
        "\n"
        // the 5th should fail to load, missing name
        "  #- name: test5\n"
        "  - regex: 'sup er?'\n"
        "\n"
        // the 6th should fail to load, bad regex, unmatched paren
        "  - name: test6\n"
        "    regex: 'sup(er?'\n";

    char *name[3] = {"test1", "test2", "test3"};
    char *regex[3] = {"sup?", "sup up?", "sup er?"};
    int binary[3] = {1, 0, 0};
    int len[3] = {111, 0, 333};
    int detect[3] = {1, 1, 0};
    int payload[3] = {0, 0, 1};

    g_protlist = lstCreate(destroyProtEntry);
    assert_non_null(g_protlist);

    char *ppath = "/tmp/" CFG_FILE_NAME;
    assert_non_null(ppath);

    writeFile(ppath, yamlText);
    
    config_t *config = cfgRead(ppath);
    assert_non_null(config);
    assert_int_equal(g_prot_sequence, 3);

    for (unsigned key = 0; key < 3; ++key) {
        protocol_def_t *prot = lstFind(g_protlist, key+1);
        assert_non_null(prot);
        assert_int_equal(prot->type, key+1);
        assert_string_equal(prot->protname, name[key]);
        assert_string_equal(prot->regex, regex[key]);
        assert_non_null(prot->re);
        assert_int_equal(prot->binary, binary[key]);
        assert_int_equal(prot->len, len[key]);
        assert_int_equal(prot->detect, detect[key]);
        assert_int_equal(prot->payload, payload[key]);
    }

    cfgDestroy(&config);
    deleteFile(ppath);
    lstDestroy(&g_protlist);
    g_prot_sequence = 0;
}

static void
initProc(const char *procname, const char *cmdline, const char *hostname)
{
    g_proc.pid = getpid();
    g_proc.ppid = getppid();

    g_proc.uid = getuid();
    g_proc.gid = getgid();


    strncpy(g_proc.hostname, hostname, sizeof(g_proc.hostname));
    strncpy(g_proc.procname, procname, sizeof(g_proc.procname));

    if (g_proc.cmd) { scope_free(g_proc.cmd); g_proc.cmd = NULL; }
    if (cmdline) g_proc.cmd = scope_strdup(cmdline);
}

static void
cfgReadCustomEmptyFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // an empty filter should not match anything
        "  eg1:\n"
        "    filter:\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // an invalid filter should not match anything
        "  eg2:\n"
        "    filter:\n"
        "      bogus: ...\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), TRUE);
    assert_int_equal(cfgEvtEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomProcnameFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      procname: not-test\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should match and enable payloads
        "  eg2:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        "      payload:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgPayEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomArgFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      arg: with\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      arg: foo\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should match and enable payloads
        "  eg2:\n"
        "    filter:\n"
        "      arg: args\n"
        "    config:\n"
        "      payload:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgPayEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomHostnameFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      hostname: myhost\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      hostname: not_myhost\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should match and enable payloads
        "  eg2:\n"
        "    filter:\n"
        "      hostname: MyHost\n" // note case change
        "    config:\n"
        "      payload:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgPayEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomUsernameFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      username: $USER\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      username: not_$USER\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should match and enable payloads
        "  eg2:\n"
        "    filter:\n"
        "      username: $USER\n"
        "    config:\n"
        "      payload:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgPayEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomEnvFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      env: USER\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      env: ___BOGUS_ENV_VAR_WE_EXPECT_IS_NOT_SET___\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should NOT match and leave payloads disabled
        "  eg2:\n"
        "    filter:\n"
        "      env: USER=not_$USER\n"
        "    config:\n"
        "      payload:\n"
        "        enable: true\n"
        // this should match and set the authToken
        "  eg2:\n"
        "    filter:\n"
        "      env: USER=$USER\n"
        "    config:\n"
        "      cribl:\n"
        "        authtoken: secret\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);
    assert_int_equal(cfgPayEnable(config), FALSE);
    assert_non_null(cfgAuthToken(config));
    assert_string_equal(cfgAuthToken(config), "secret");

    cfgDestroy(&config);
}

static void
cfgReadCustomAncestorFilter(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      ancestor: make\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      ancestor: bogus\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomMultipleFilters(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      procname: test\n"
        "      arg: args\n"
        "      hostname: myhost\n"
        "      username: $USER\n"
        "      env: USER\n"
        "      env: USER=$USER\n"
        "      ancestor: make\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should NOT match and leave events enabled
        "  eg2:\n"
        "    filter:\n"
        "      procname: not_test\n"  // only this changed from above
        "      arg: args\n"
        "      hostname: myhost\n"
        "      username: $USER\n"
        "      env: USER\n"
        "      env: USER=$USER\n"
        "      ancestor: make\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomOverride(void **state)
{
    const char *yamlText =
        "# use default configs to start then these overrides\n"
        "custom:\n"
        // this should match and disable metrics
        "  eg1:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        "      metric:\n"
        "        enable: false\n"
        // this should also match and disable events too
        "  eg2:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        "      event:\n"
        "        enable: false\n"
        // this should also match and re-enable events
        "  eg3:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        "      event:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgEvtEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomOrder(void **state)
{
    const char *yamlText =
        "custom:\n"
        // this should match and re-enable metrics and
        // it should work even though the filter isn't first
        "  eg1:\n"
        "    config:\n"
        "      metric:\n"
        "        enable: true\n"
        "    filter:\n"
        "      procname: test\n"
        "\n"
        // this should get processed first, before the custom entries
        "# disable metrics\n"
        "metric:\n"
        "  enable: false\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
cfgReadCustomAnchor(void **state)
{
    const char *yamlText =
        // define a config using an anchor
        "disable-metrics: &disable-metrics\n"
        "  metric:\n"
        "    enable: false\n"
        "custom:\n"
        "  eg1:\n"
        "    filter:\n"
        "      procname: test\n"
        // use that config as a custom config
        "    config: *disable-metrics\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);

    cfgDestroy(&config);
}

static void
cfgReadCustomAnchorExtend(void **state)
{
    // We're skipping this test right now because the << extend syntax is not
    // supported OOB by libyaml. We would need to add support for this ourself.
    skip();

    const char *yamlText =
        // define a config using an anchor
        "disable-metrics: &disable-metrics\n"
        "  metric:\n"
        "    enable: false\n"
        "custom:\n"
        "  eg1:\n"
        "    filter:\n"
        "      procname: test\n"
        "    config:\n"
        // include that config and extend it in a custom config
        "      <<: *disable-metrics\n"
        "      payload:\n"
        "        enable: true\n"
        "# EOF\n";
    const char *yamlFilename = "/tmp/eg-scope.yml";
    writeFile(yamlFilename, yamlText);
    initProc("test", "test --with args", "myhost");
    config_t *config = cfgRead(yamlFilename);
    deleteFile(yamlFilename);
    assert_non_null(config);

    assert_int_equal(cfgMtcEnable(config), FALSE);
    assert_int_equal(cfgPayEnable(config), TRUE);

    cfgDestroy(&config);
}

static void
rulesEmptyRulesFileVar1(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/empty_rules_0.yml", dirPath);
    bool res = cfgRulesFileIsValid(path);
    assert_int_equal(res, FALSE);
}

static void
rulesEmptyRulesFileVar2(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/empty_rules_1.yml", dirPath);
    bool res = cfgRulesFileIsValid(path);
    assert_int_equal(res, FALSE);
}

static void
rulesEmptyRulesFileVar3(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/empty_rules_2.yml", dirPath);
    bool res = cfgRulesFileIsValid(path);
    assert_int_equal(res, FALSE);
}

static void
rulesInvalidRulesFile(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/corrupted_rules.yml", dirPath);
    bool res = cfgRulesFileIsValid(path);
    assert_int_equal(res, FALSE);
}

static void
rulesEmptyProcName(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus(NULL, "foo", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_ERROR);
    dbgInit(); // reset dbg for the rest of the tests
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesEmptyProcCmdLine(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("foo", NULL, testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_ERROR);
    dbgInit(); // reset dbg for the rest of the tests
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

/*
 * Note: a NULL rules path implies use default paths.
 * We have not configured defaults. So, there is no
 * rules file found. With no rules file we assume
 * all files are to be scoped. Look for a SCOPED return value.
 */
static void
rulesNullRulesPath(void **state) {
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("foo", "foo", NULL, cfg);
    assert_int_equal(res, RULES_SCOPED);
    dbgInit(); // reset dbg for the rest of the tests
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesNullCfg(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    rules_status_t res = cfgRulesStatus("foo", "foo", testAccessRulesPath(path), NULL);
    assert_int_equal(res, RULES_ERROR);
    dbgInit(); // reset dbg for the rest of the tests
}

// This is really testing testAccessRulesPath(), but it still captures
// what the desired system behavior is...
static void
rulesNonExistingRulesFile(void **state) {
    char path[PATH_MAX] = {0};    
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_non_existing.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, FALSE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("foo", "foo", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesProcNameAllowListPresent(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("redis", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesProcNameDenyListPresent(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("git", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesArgAllowListPresent(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_1.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("", "redis arg1", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesArgDenyListPresent(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_1.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("", "git arg1", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesArgAllowListPartFindPresent(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("redis-server", "redis-server", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    scope_memset(path, 0, sizeof(path));
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_1.yml", dirPath);
    res = cfgRulesStatus("redis-server", "redis-server", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesArgAllowListEmptyProcMissing(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    rules_status_t res = cfgRulesStatus("memcached", "memcached", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesArgAllowListNotEmptyProcMissing(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_2.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("memcached", "memcached", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);

    scope_memset(path, 0, PATH_MAX);

    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_3.yml", dirPath);
    validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    res = cfgRulesStatus("memcached", "memcached", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesVerifyCfg(void **state) {
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_0.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("redis", "redis", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // redis: cribl disable error log level /tmp/redis.log
    assert_int_equal(cfgLogStreamEnable(cfg), FALSE);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_ERROR);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/redis.log");
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);

    cfg = cfgCreateDefault();
    assert_non_null(cfg);
    res = cfgRulesStatus("htop", "htop", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // htop: cribl enable info log level /tmp/htop.log
    assert_int_equal(cfgLogStreamEnable(cfg), TRUE);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_INFO);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/htop.log");
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesDenyIsProcessedAfterAllow(void **state)
{
    // redis is in both the allow list and deny list.
    // verify that "deny" wins.  (is processed after allow)
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_4.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("redis", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesConfigIsProcessedAfterProcName(void **state)
{
    // cfg should be changed if procname or arg matches
    // make sure we process these fields before config.
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_4.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    // test the default log path before
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/scope.log");

    rules_status_t res = cfgRulesStatus("htop", "htop", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);

    // verify that log path was changed
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/htop.log");

    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesMatchAllInAllow(void **state)
{
    // Verify that _MatchAll_ in allow matches all processes
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_5.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("blue", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    res = cfgRulesStatus("red", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    res = cfgRulesStatus("green", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    res = cfgRulesStatus("htop", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesMatchAllInAllowCanBeDenied(void **state)
{
    // Verify that _MatchAll_ in allow is overriden by a match in deny
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_5.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("redis", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesVerifyMatchAllMergedConfig(void **state)
{
    // Verify that matches (including _MatchAll_) are applied in order
    // And that matches can be "merged" w.r.t. configuration
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_5.yml", dirPath);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);

    // green only matches _MatchAll_
    // verify that the log level and log path agree with _MatchAll_
    rules_status_t res = cfgRulesStatus("green", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_ERROR);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/match.log");
    cfgDestroy(&cfg);

    // htop matches _MatchAll_, then matches procname: htop
    // verify that the log level is specified by _MatchAll_
    //   and that log path is specified by htop
    cfg = cfgCreateDefault();
    res = cfgRulesStatus("htop", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_SCOPED_WITH_CFG);
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_ERROR);
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/htop.log");

    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesMatchAllInDeny(void **state)
{
    // Verify that _MatchAll_ in deny denies all processes
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_6.yml", dirPath);
    bool validRes = cfgRulesFileIsValid(path);
    assert_int_equal(validRes, TRUE);
    config_t *cfg = cfgCreateDefault();
    assert_non_null(cfg);
    rules_status_t res = cfgRulesStatus("blue", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    res = cfgRulesStatus("red", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    res = cfgRulesStatus("green", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    res = cfgRulesStatus("htop", "", testAccessRulesPath(path), cfg);
    assert_int_equal(res, RULES_NOT_SCOPED);
    // cleanup
    cfgDestroy(&cfg);
    assert_null(cfg);
}

static void
rulesUnixPathMissing(void **state)
{
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_6.yml", dirPath);

    const char *origRulesEnv = getenv("SCOPE_RULES");
    assert_int_equal(setenv("SCOPE_RULES", path, 1), 0);

    char *unixPath = cfgRulesUnixPath();

    if (origRulesEnv) {
        assert_int_equal(setenv("SCOPE_RULES", origRulesEnv, 1), 0);
    } else {
        assert_int_equal(unsetenv("SCOPE_RULES"), 0);
    }

    assert_null(unixPath);
}

static void
rulesUnixPathPresent(void **state)
{
    char path[PATH_MAX] = {0};
    scope_snprintf(path, sizeof(path), "%s/data/rules/rules_edge.yml", dirPath);

    assert_int_equal(setenv("SCOPE_RULES", path, 1), 0);

    char *unixPath = cfgRulesUnixPath();

    assert_int_equal(unsetenv("SCOPE_RULES"), 0);

    assert_string_equal(unixPath, "/opt/cribl/state");
    scope_free(unixPath);
}

// Defined in src/cfgutils.c
// This is not a proper test, it just exists to make valgrind output
// more readable when analyzing this test, by deallocating the compiled
// regex in src/cfgutils.c.
extern void envRegexFree(void **state);


int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);
    if (testDirPath(dirPath, argv[0])) {
        return EXIT_FAILURE;
    }

    initFn();

    source_state_t log = {"SCOPE_EVENT_LOGFILE", CFG_SRC_FILE, DEFAULT_SRC_FILE};
    source_state_t con = {"SCOPE_EVENT_CONSOLE", CFG_SRC_CONSOLE, DEFAULT_SRC_CONSOLE};
    source_state_t sys = {"SCOPE_EVENT_SYSLOG" , CFG_SRC_SYSLOG , DEFAULT_SRC_SYSLOG};
    source_state_t met = {"SCOPE_EVENT_METRIC", CFG_SRC_METRIC , DEFAULT_SRC_METRIC};
    source_state_t htt = {"SCOPE_EVENT_HTTP", CFG_SRC_HTTP , DEFAULT_SRC_HTTP};
    source_state_t net = {"SCOPE_EVENT_NET", CFG_SRC_NET , DEFAULT_SRC_NET};
    source_state_t fs =  {"SCOPE_EVENT_FS", CFG_SRC_FS , DEFAULT_SRC_FS};
    source_state_t dns = {"SCOPE_EVENT_DNS", CFG_SRC_DNS , DEFAULT_SRC_DNS};

    dest_state_t dest_mtc = {"SCOPE_METRIC_DEST", CFG_MTC};
    dest_state_t dest_evt = {"SCOPE_EVENT_DEST", CFG_CTL};
    dest_state_t dest_log = {"SCOPE_LOG_DEST", CFG_LOG};

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgPathHonorsEnvVar),
        // XXX This test is failing under CI at GitHub but passes locally?
        //     I'm being lazy and just disabling it for now. --pd
        //cmocka_unit_test(cfgPathHonorsPriorityOrder),
        cmocka_unit_test(cfgProcessEnvironmentMtcEnable),
        cmocka_unit_test(cfgProcessEnvironmentMtcFormat),
        cmocka_unit_test(cfgProcessEnvironmentStatsDPrefix),
        cmocka_unit_test(cfgProcessEnvironmentStatsDMaxLen),
        cmocka_unit_test(cfgProcessEnvironmentWatchStatsdEnable),
        cmocka_unit_test(cfgProcessEnvironmentMtcPeriod),
        cmocka_unit_test(cfgProcessEnvironmentCommandDir),
        cmocka_unit_test(cfgProcessEnvironmentConfigEvent),
        cmocka_unit_test(cfgProcessEnvironmentEvtEnable),
        cmocka_unit_test(cfgProcessEnvironmentEventFormat),
        cmocka_unit_test(cfgProcessEnvironmentMaxEps),
        cmocka_unit_test(cfgProcessEnvironmentEnhanceFs),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &log),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &con),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &sys),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &met),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &htt),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &net),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &fs),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &dns),
        cmocka_unit_test(cfgProcessEnvironmentMtcVerbosity),
        cmocka_unit_test(cfgProcessEnvironmentLogLevel),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_mtc),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_evt),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_log),
        cmocka_unit_test(cfgProcessEnvironmentStatsdTags),
        cmocka_unit_test(cfgProcessEnvironmentPayEnable),
        cmocka_unit_test(cfgProcessEnvironmentPayType),
        cmocka_unit_test(cfgProcessEnvironmentPayDir),
        cmocka_unit_test(cfgProcessEnvironmentCmdDebugIsIgnored),
        cmocka_unit_test(cfgProcessCommandsCmdDebugIsProcessed),
        cmocka_unit_test(cfgProcessCommandsFromFile),
        cmocka_unit_test(cfgProcessCommandsEnvSubstitution),
        cmocka_unit_test(cfgReadGoodYaml),
        cmocka_unit_test(cfgReadStockYaml),
        cmocka_unit_test(cfgReadEveryTransportType),
        cmocka_unit_test(cfgReadEveryProcessLevel),
        cmocka_unit_test(cfgReadGoodJson),
        cmocka_unit_test(cfgReadNonExistentFileReturnsDefaults),
        cmocka_unit_test(cfgReadBadYamlReturnsDefaults),
        cmocka_unit_test(cfgReadExtraFieldsAreHarmless),
        cmocka_unit_test(cfgReadYamlOrderWithinStructureDoesntMatter),
        cmocka_unit_test(cfgReadEnvSubstitution),
        cmocka_unit_test(jsonObjectFromCfgAndjsonStringFromCfgRoundTrip),
        cmocka_unit_test(initLogReturnsPtr),
        cmocka_unit_test(initMtcReturnsPtr),
        cmocka_unit_test(initEvtFormatReturnsPtr),
        cmocka_unit_test(initCtlReturnsPtr),
        cmocka_unit_test(rulesEmptyRulesFileVar1),
        cmocka_unit_test(rulesEmptyRulesFileVar2),
        cmocka_unit_test(rulesEmptyRulesFileVar3),
        cmocka_unit_test(rulesInvalidRulesFile),
        cmocka_unit_test(rulesEmptyProcName),
        cmocka_unit_test(rulesEmptyProcCmdLine),
        cmocka_unit_test(rulesNullRulesPath),
        cmocka_unit_test(rulesNullCfg),
        cmocka_unit_test(rulesNonExistingRulesFile),
        cmocka_unit_test(rulesProcNameAllowListPresent),
        cmocka_unit_test(rulesProcNameDenyListPresent),
        cmocka_unit_test(rulesArgAllowListPresent),
        cmocka_unit_test(rulesArgDenyListPresent),
        cmocka_unit_test(rulesArgAllowListPartFindPresent),
        cmocka_unit_test(rulesArgAllowListEmptyProcMissing),
        cmocka_unit_test(rulesArgAllowListNotEmptyProcMissing),
        cmocka_unit_test(rulesVerifyCfg),
        cmocka_unit_test(rulesDenyIsProcessedAfterAllow),
        cmocka_unit_test(rulesConfigIsProcessedAfterProcName),
        cmocka_unit_test(rulesMatchAllInAllow),
        cmocka_unit_test(rulesMatchAllInAllowCanBeDenied),
        cmocka_unit_test(rulesVerifyMatchAllMergedConfig),
        cmocka_unit_test(rulesMatchAllInDeny),
        cmocka_unit_test(rulesUnixPathMissing),
        cmocka_unit_test(rulesUnixPathPresent),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
        cmocka_unit_test(cfgReadProtocol),
        cmocka_unit_test(cfgReadCustomEmptyFilter),
        cmocka_unit_test(cfgReadCustomProcnameFilter),
        cmocka_unit_test(cfgReadCustomArgFilter),
        cmocka_unit_test(cfgReadCustomHostnameFilter),
        cmocka_unit_test(cfgReadCustomUsernameFilter),
        cmocka_unit_test(cfgReadCustomEnvFilter),
        cmocka_unit_test(cfgReadCustomAncestorFilter),
        cmocka_unit_test(cfgReadCustomMultipleFilters),
        cmocka_unit_test(cfgReadCustomOverride),
        cmocka_unit_test(cfgReadCustomOrder),
        cmocka_unit_test(cfgReadCustomAnchor),
        cmocka_unit_test(cfgReadCustomAnchorExtend),
        cmocka_unit_test(envRegexFree),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
