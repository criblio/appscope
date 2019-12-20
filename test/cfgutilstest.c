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


static void
openFileAndExecuteCfgProcessCommands(const char* path, config_t* cfg)
{
    FILE* f = fopen(path, "r");
    cfgProcessCommands(cfg, f);
    fclose(f);
}

static void
cfgPathHonorsEnvVar(void** state)
{
    const char* file_path = "/tmp/myfile.yml";

    // grab the current working directory
    char origdir[MAX_PATH];
    assert_non_null(getcwd(origdir, sizeof(origdir)));
    // create newdir, and switch to it
    char newdir[MAX_PATH];
    snprintf(newdir, sizeof(newdir), "%s/%s", origdir, "newdir");
    assert_int_equal(mkdir(newdir, 0777), 0);
    assert_int_equal(chdir(newdir), 0);


    // Verify that if there is no env variable, cfgPath is null
    assert_null(cfgPath());

    // Verify that if there is an env variable, but no file, cfgPath is null
    assert_int_equal(setenv("SCOPE_CONF_PATH", file_path, 1), 0);
    assert_null(cfgPath());

    // Verify that if there is an env variable, and a file, cfgPath is defined
    int fd = open(file_path, O_RDWR | O_CREAT, S_IRUSR | S_IRGRP | S_IROTH);
    assert_return_code(fd, errno);
    char* path = cfgPath();
    assert_non_null(path);
    assert_string_equal(path, file_path);

    // cleanup
    free(path);
    unlink(file_path);
    assert_int_equal(unsetenv("SCOPE_CONF_PATH"), 0);

    // change back to origdir
    assert_int_equal(chdir(origdir), 0);
    // Delete the directory we created
    assert_int_equal(rmdir(newdir), 0);
}

static void
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
        if (access(newdir[i], F_OK) == -1) {
            assert_int_equal(mkdir(newdir[i], 0777), 0);
        }
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
    const char file[] = CFG_FILE_NAME; // scope.yml
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
        char* result = cfgPath();
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

static void
cfgProcessEnvironmentOutFormat(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutFormatSet(cfg, CFG_METRIC_JSON);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "metricstatsd", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_STATSD);

    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "ndjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_EVENT_ND_JSON);

    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "metricjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_FORMAT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_FORMAT", "bson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
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

static void
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

static void
cfgProcessEnvironmentOutPeriod(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutPeriodSet(cfg, 0);
    assert_int_equal(cfgOutPeriod(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 3);

    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "12", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_SUMMARY_PERIOD"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_SUMMARY_PERIOD", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutPeriod(cfg), 12);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
cfgProcessEnvironmentCommandDir(void** state)
{
    config_t* cfg = cfgCreateDefault();
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
cfgProcessEnvironmentEventFormat(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgEventFormatSet(cfg, CFG_METRIC_JSON);
    assert_int_equal(cfgEventFormat(cfg), CFG_METRIC_JSON);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "ndjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_EVENT_ND_JSON);

    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "metricjson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_METRIC_JSON);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_EVENT_FORMAT"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_METRIC_JSON);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_EVENT_FORMAT", "bson", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventFormat(cfg), CFG_METRIC_JSON);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}


typedef struct
{
    const char* env_name;
    cfg_evt_t   src;
    unsigned    default_val;
} source_state_t;

static void
cfgProcessEnvironmentEventSource(void** state)
{
    source_state_t* data = (source_state_t*)state[0];

    config_t* cfg = cfgCreateDefault();
    cfgEventSourceEnabledSet(cfg, data->src, 0);
    assert_int_equal(cfgEventSourceEnabled(cfg, data->src), 0);

    // should override current cfg
    assert_int_equal(setenv(data->env_name, "true", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventSourceEnabled(cfg, data->src), 1);

    assert_int_equal(setenv(data->env_name, "false", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventSourceEnabled(cfg, data->src), 0);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv(data->env_name), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventSourceEnabled(cfg, data->src), 0);

    // empty string
    assert_int_equal(setenv(data->env_name, "", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgEventSourceEnabled(cfg, data->src), data->default_val);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}


static void
cfgProcessEnvironmentOutVerbosity(void** state)
{
    config_t* cfg = cfgCreateDefault();
    cfgOutVerbositySet(cfg, 0);
    assert_int_equal(cfgOutVerbosity(cfg), 0);

    // should override current cfg
    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "3", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 3);

    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "9", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 9);

    // if env is not defined, cfg should not be affected
    assert_int_equal(unsetenv("SCOPE_METRIC_VERBOSITY"), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 9);

    // unrecognised value should not affect cfg
    assert_int_equal(setenv("SCOPE_METRIC_VERBOSITY", "notEvenANum", 1), 0);
    cfgProcessEnvironment(cfg);
    assert_int_equal(cfgOutVerbosity(cfg), 9);

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
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

typedef struct
{
    const char* env_name;
    which_transport_t transport;
} dest_state_t;

static void
cfgProcessEnvironmentTransport(void** state)
{
    dest_state_t* data = (dest_state_t*)state[0];

    config_t* cfg = cfgCreateDefault();

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

    // Just don't crash on null cfg
    cfgDestroy(&cfg);
    cfgProcessEnvironment(cfg);
}

static void
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

static void
cfgProcessEnvironmentCmdDebugIsIgnored(void** state)
{
    const char* path = "/tmp/dbgoutfile.txt";
    assert_int_equal(setenv("SCOPE_CMD_DBG_PATH", path, 1), 0);

    long file_pos_before = fileEndPosition(path);

    config_t* cfg = cfgCreateDefault();
    cfgProcessEnvironment(cfg);
    cfgDestroy(&cfg);

    long file_pos_after = fileEndPosition(path);

    // since it's not processed, the file position better not have changed.
    assert_int_equal(file_pos_before, file_pos_after);

    unsetenv("SCOPE_CMD_DBG_PATH");
    if (file_pos_after != -1) unlink(path);
}

static void
cfgProcessCommandsCmdDebugIsProcessed(void** state)
{
    const char* outpath = "/tmp/dbgoutfile.txt";
    const char* inpath = "/tmp/dbginfile.txt";

    long file_pos_before = fileEndPosition(outpath);

    config_t* cfg = cfgCreateDefault();
    writeFile(inpath, "SCOPE_CMD_DBG_PATH=/tmp/dbgoutfile.txt");
    openFileAndExecuteCfgProcessCommands(inpath, cfg);
    cfgDestroy(&cfg);

    long file_pos_after = fileEndPosition(outpath);

    // since it's not processed, the file position should be updated
    assert_int_not_equal(file_pos_before, file_pos_after);

    unlink(inpath);
    if (file_pos_after != -1) unlink(outpath);
}

static void
cfgProcessCommandsFromFile(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    const char* path = "/tmp/test.file";

    // Just making sure these don't crash us.
    cfgProcessCommands(NULL, NULL);
    cfgProcessCommands(cfg, NULL);


    // test the basics
    writeFile(path, "SCOPE_METRIC_FORMAT=metricjson");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    writeFile(path, "\nSCOPE_METRIC_FORMAT=metricstatsd\r\nblah");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_STATSD);

    writeFile(path, "blah\nSCOPE_METRIC_FORMAT=metricjson");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_JSON);

    // just demonstrating that the "last one wins"
    writeFile(path, "SCOPE_METRIC_FORMAT=metricjson\n"
                    "SCOPE_METRIC_FORMAT=metricstatsd");
    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_int_equal(cfgOutFormat(cfg), CFG_METRIC_STATSD);


    // test everything else once
    writeFile(path,
        "SCOPE_STATSD_PREFIX=prefix\n"
        "SCOPE_STATSD_MAXLEN=1024\n"
        "SCOPE_SUMMARY_PERIOD=11\n"
        "SCOPE_CMD_DIR=/the/path/\n"
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
        "SCOPE_EVENT_FORMAT=ndjson\n"
        "SCOPE_EVENT_LOGFILE=true\n"
        "SCOPE_EVENT_CONSOLE=false\n"
        "SCOPE_EVENT_SYSLOG=true\n"
        "SCOPE_EVENT_METRIC=false\n"
        "SCOPE_EVENT_LOGFILE_NAME=a\n"
        "SCOPE_EVENT_CONSOLE_NAME=b\n"
        "SCOPE_EVENT_SYSLOG_NAME=c\n"
        "SCOPE_EVENT_METRIC_NAME=d\n"
        "SCOPE_EVENT_LOGFILE_FIELD=e\n"
        "SCOPE_EVENT_CONSOLE_FIELD=f\n"
        "SCOPE_EVENT_SYSLOG_FIELD=g\n"
        "SCOPE_EVENT_METRIC_FIELD=h\n"
        "SCOPE_EVENT_LOGFILE_VALUE=i\n"
        "SCOPE_EVENT_CONSOLE_VALUE=j\n"
        "SCOPE_EVENT_SYSLOG_VALUE=k\n"
        "SCOPE_EVENT_METRIC_VALUE=l\n"
    );

    openFileAndExecuteCfgProcessCommands(path, cfg);
    assert_string_equal(cfgOutStatsDPrefix(cfg), "prefix.");
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgOutPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/the/path/");
    assert_int_equal(cfgOutVerbosity(cfg), 1);
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/tmp/file.tmp");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM1"), "val1");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM2"), "val2");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    assert_int_equal(cfgTransportType(cfg, CFG_CTL), CFG_UDP);
    assert_string_equal(cfgTransportHost(cfg, CFG_CTL), "host");
    assert_string_equal(cfgTransportPort(cfg, CFG_CTL), "1234");
    assert_int_equal(cfgEventFormat(cfg), CFG_EVENT_ND_JSON);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_METRIC), 0);
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_FILE), "a");
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_CONSOLE), "b");
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_SYSLOG), "c");
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_METRIC), "d");
    assert_string_equal(cfgEventFieldFilter(cfg, CFG_SRC_FILE), "e");
    assert_string_equal(cfgEventFieldFilter(cfg, CFG_SRC_CONSOLE), "f");
    assert_string_equal(cfgEventFieldFilter(cfg, CFG_SRC_SYSLOG), "g");
    assert_string_equal(cfgEventFieldFilter(cfg, CFG_SRC_METRIC), "h");
    assert_string_equal(cfgEventValueFilter(cfg, CFG_SRC_FILE), "i");
    assert_string_equal(cfgEventValueFilter(cfg, CFG_SRC_CONSOLE), "j");
    assert_string_equal(cfgEventValueFilter(cfg, CFG_SRC_SYSLOG), "k");
    assert_string_equal(cfgEventValueFilter(cfg, CFG_SRC_METRIC), "l");

    deleteFile(path);
    cfgDestroy(&cfg);
}

static void
cfgProcessCommandsEnvSubstitution(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    const char* path = "/tmp/test.file";

    // test everything else once
    writeFile(path,
        "SCOPE_STATSD_PREFIX=$VAR1.$MY_ENV_VAR\n"
        "SCOPE_STATSD_MAXLEN=$MAXLEN\n"
        "SCOPE_SUMMARY_PERIOD=$PERIOD\n"
        "SCOPE_CMD_DIR=/$MYHOME/scope/\n"
        "SCOPE_METRIC_VERBOSITY=$VERBOSITY\n"
        "SCOPE_LOG_LEVEL=$LOGLEVEL\n"
        "SCOPE_METRIC_DEST=file:///\\$VAR1/$MY_ENV_VAR/\n"
        "SCOPE_LOG_DEST=$DEST\n"
        "SCOPE_TAG_CUSTOM=$PERIOD\n"
        "SCOPE_TAG_whyyoumadbro=Bill owes me $5.00\n"
        "SCOPE_TAG_undefined=$UNDEFINEDENV\n"
        "SCOPE_EVENT_DEST=udp://ho$st:1234\n"
        "SCOPE_EVENT_LOGFILE=$TRUTH\n"
        "SCOPE_EVENT_CONSOLE=false\n"
        "SCOPE_EVENT_SYSLOG=$TRUTH\n"
        "SCOPE_EVENT_METRIC=false\n"
        "SCOPE_EVENT_LOGFILE_NAME=$FILTER\n"
    );


    // Set env varibles to test indirect substitution
    assert_int_equal(setenv("VAR1", "longer", 1), 0);
    assert_int_equal(setenv("MY_ENV_VAR", "shorter", 1), 0);
    assert_int_equal(setenv("MAXLEN", "1024", 1), 0);
    assert_int_equal(setenv("DEST", "file:///tmp/file.tmp2", 1), 0);
    assert_int_equal(setenv("PERIOD", "11", 1), 0);
    assert_int_equal(setenv("MYHOME", "home/mydir", 1), 0);
    assert_int_equal(setenv("VERBOSITY", "1", 1), 0);
    assert_int_equal(setenv("LOGLEVEL", "trace", 1), 0);
    assert_int_equal(setenv("FILTER", ".*[.]log$", 1), 0);
    assert_int_equal(setenv("TRUTH", "true", 1), 0);

    openFileAndExecuteCfgProcessCommands(path, cfg);
    // test substitute env values that are longer and shorter than they env name
    assert_string_equal(cfgOutStatsDPrefix(cfg), "longer.shorter.");
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgOutPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/home/mydir/scope/");
    assert_int_equal(cfgOutVerbosity(cfg), 1);
    // test escaped substitution  (a match preceeded by '\')
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/$VAR1/shorter/");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM"), "11");
    // test lookups that aren't found: $5 and $UNDEFINEDENV
    assert_string_equal(cfgCustomTagValue(cfg, "whyyoumadbro"), "Bill owes me $5.00");
    assert_string_equal(cfgCustomTagValue(cfg, "undefined"), "$UNDEFINEDENV");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    // event stuff...
    assert_string_equal(cfgTransportHost(cfg, CFG_CTL), "ho$st");
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_METRIC), 0);

    deleteFile(path);
    cfgDestroy(&cfg);

    unsetenv("VAR1");
    unsetenv("MY_ENV_VAR");
    unsetenv("MAXLEN");
    unsetenv("DEST");
    unsetenv("PERIOD");
    unsetenv("VERBOSITY");
    unsetenv("LOGLEVEL");
    unsetenv("FILTER");
    unsetenv("TRUTH");
}

static void
verifyDefaults(config_t* config)
{
    assert_int_equal       (cfgOutFormat(config), DEFAULT_OUT_FORMAT);
    assert_string_equal    (cfgOutStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal       (cfgOutStatsDMaxLen(config), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal       (cfgOutVerbosity(config), DEFAULT_OUT_VERBOSITY);
    assert_int_equal       (cfgOutPeriod(config), DEFAULT_SUMMARY_PERIOD);
    assert_string_equal    (cfgCmdDir(config), DEFAULT_COMMAND_DIR);
    assert_int_equal       (cfgEventFormat(config), DEFAULT_CTL_FORMAT);
    assert_string_equal    (cfgEventValueFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_VALUE);
    assert_string_equal    (cfgEventValueFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_VALUE);
    assert_string_equal    (cfgEventValueFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_VALUE);
    assert_string_equal    (cfgEventValueFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_VALUE);
    assert_string_equal    (cfgEventFieldFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_FIELD);
    assert_string_equal    (cfgEventFieldFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_FIELD);
    assert_string_equal    (cfgEventFieldFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_FIELD);
    assert_string_equal    (cfgEventFieldFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_FIELD);
    assert_string_equal    (cfgEventNameFilter(config, CFG_SRC_FILE), DEFAULT_SRC_FILE_NAME);
    assert_string_equal    (cfgEventNameFilter(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE_NAME);
    assert_string_equal    (cfgEventNameFilter(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG_NAME);
    assert_string_equal    (cfgEventNameFilter(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC_NAME);
    assert_int_equal       (cfgEventSourceEnabled(config, CFG_SRC_FILE), DEFAULT_SRC_FILE);
    assert_int_equal       (cfgEventSourceEnabled(config, CFG_SRC_CONSOLE), DEFAULT_SRC_CONSOLE);
    assert_int_equal       (cfgEventSourceEnabled(config, CFG_SRC_SYSLOG), DEFAULT_SRC_SYSLOG);
    assert_int_equal       (cfgEventSourceEnabled(config, CFG_SRC_METRIC), DEFAULT_SRC_METRIC);
    assert_int_equal       (cfgTransportType(config, CFG_OUT), CFG_UDP);
    assert_string_equal    (cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_string_equal    (cfgTransportPort(config, CFG_OUT), "8125");
    assert_null            (cfgTransportPath(config, CFG_OUT));
    assert_int_equal       (cfgTransportBuf(config, CFG_OUT), CFG_BUFFER_FULLY);
    assert_int_equal       (cfgTransportType(config, CFG_CTL), CFG_TCP);
    assert_string_equal    (cfgTransportHost(config, CFG_CTL), "127.0.0.1");
    assert_string_equal    (cfgTransportPort(config, CFG_CTL), DEFAULT_CTL_PORT);
    assert_null            (cfgTransportPath(config, CFG_CTL));
    assert_int_equal       (cfgTransportBuf(config, CFG_CTL), CFG_BUFFER_FULLY);
    assert_int_equal       (cfgTransportType(config, CFG_LOG), CFG_FILE);
    assert_null            (cfgTransportHost(config, CFG_LOG));
    assert_null            (cfgTransportPort(config, CFG_LOG));
    assert_string_equal    (cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_int_equal       (cfgTransportBuf(config, CFG_OUT), CFG_BUFFER_FULLY);
    assert_null            (cfgCustomTags(config));
    assert_null            (cfgCustomTagValue(config, "tagname"));
    assert_int_equal       (cfgLogLevel(config), DEFAULT_LOG_LEVEL);
}


static void
cfgReadGoodYaml(void** state)
{
    // Test file config (yaml)
    const char* yamlText =
        "---\n"
        "metric:\n"
        "  format:\n"
        "    type: metricjson                # metricstatsd, metricjson\n"
        "    statsdprefix : 'cribl.scope'    # prepends each statsd metric\n"
        "    statsdmaxlen : 1024             # max size of a formatted statsd string\n"
        "    verbosity: 3                    # 0-9 (0 is least verbose, 9 is most)\n"
        "    tags:\n"
        "    - name1 : value1\n"
        "    - name2 : value2\n"
        "  transport:                        # defines how scope output is sent\n"
        "    type: file                      # udp, unix, file, syslog\n"
        "    path: '/var/log/scope.log'\n"
        "    buffering: line\n"
        "event:\n"
        "  format:\n"
        "    type : metricjson               # ndjson\n"
        "  watch:\n"
        "    - type: file                    # create events from file\n"
        "      name: .*[.]log$\n"
        "      field: .*host.*\n"
        "      value: '[0-9]+'\n"
        "    - type: console                 # create events from stdout and stderr\n"
        "    - type: syslog                  # create events from syslog and vsyslog\n"
        "    - type: metric\n"
        "libscope:\n"
        "  transport:\n"
        "    type: tcp                       # udp, unix, file, syslog\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "  summaryperiod: 11                 # in seconds\n"
        "  commanddir: /tmp\n"
        "  log:\n"
        "    level: debug                      # debug, info, warning, error, none\n"
        "    transport:\n"
        "      buffering: full\n"
        "      type: syslog\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_METRIC_JSON);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 1024);
    assert_int_equal(cfgOutVerbosity(config), 3);
    assert_int_equal(cfgOutPeriod(config), 11);
    assert_string_equal(cfgCmdDir(config), "/tmp");
    assert_int_equal(cfgEventFormat(config), CFG_METRIC_JSON);
    assert_string_equal(cfgEventNameFilter(config, CFG_SRC_FILE), ".*[.]log$");
    assert_string_equal(cfgEventFieldFilter(config, CFG_SRC_FILE), ".*host.*");
    assert_string_equal(cfgEventValueFilter(config, CFG_SRC_FILE), "[0-9]+");
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_CONSOLE), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_OUT), "8125");
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/log/scope.log");
    assert_int_equal(cfgTransportBuf(config, CFG_OUT), CFG_BUFFER_LINE);
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_TCP);
    assert_string_equal(cfgTransportHost(config, CFG_CTL), "127.0.0.2");
    assert_string_equal(cfgTransportPort(config, CFG_CTL), "9009");
    assert_null(cfgTransportPath(config, CFG_CTL));
    assert_int_equal(cfgTransportBuf(config, CFG_CTL), CFG_BUFFER_LINE);
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_SYSLOG);
    assert_null(cfgTransportHost(config, CFG_LOG));
    assert_null(cfgTransportPort(config, CFG_LOG));
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_int_equal(cfgTransportBuf(config, CFG_LOG), CFG_BUFFER_FULLY);
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "name1"), "value1");
    assert_string_equal(cfgCustomTagValue(config, "name2"), "value2");
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
        "metric:\n"
        "  transport:\n"
        "%s"
        "...\n";

    const char* udp_str =
        "    type: udp\n"
        "    host: 'labmachine8235'\n"
        "    port: 'ntp'\n";
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
                assert_string_equal(cfgTransportPort(config, CFG_OUT), "ntp");
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
        "libscope:\n"
        "  log:\n"
        "    level: %s\n"
        "...\n";

    const char* path = CFG_FILE_NAME;
    const char* level[] = { "trace", "debug", "info", "warning", "error", "none"};
    cfg_log_level_t value[] = {CFG_LOG_TRACE, CFG_LOG_DEBUG, CFG_LOG_INFO, CFG_LOG_WARN, CFG_LOG_ERROR, CFG_LOG_NONE};
    int i;
    for (i = 0; i< sizeof(level)/sizeof(level[0]); i++) {
        writeFileWithSubstitution(path, yamlText, level[i]);
        config_t* config = cfgRead(path);
        assert_int_equal(cfgLogLevel(config), value[i]);
        deleteFile(path);
        cfgDestroy(&config);
    }
}

// Test file config (json)
const char* jsonText =
    "{\n"
    "  'metric': {\n"
    "    'format': {\n"
    "      'type': 'metricjson',\n"
    "      'statsdprefix': 'cribl.scope',\n"
    "      'statsdmaxlen': '42',\n"
    "      'verbosity': '0',\n"
    "      'tags': [\n"
    "        {'tagA': 'val1'},\n"
    "        {'tagB': 'val2'},\n"
    "        {'tagC': 'val3'}\n"
    "      ]\n"
    "    },\n"
    "    'transport': {\n"
    "      'type': 'file',\n"
    "      'path': '/var/log/scope.log'\n"
    "    }\n"
    "  },\n"
    "  'event': {\n"
    "    'format': {\n"
    "      'type': 'ndjson'\n"
    "    },\n"
    "    'watch' : [\n"
    "      {'type':'file', 'name':'.*[.]log$'},\n"
    "      {'type':'console'},\n"
    "      {'type':'syslog'},\n"
    "      {'type':'metric'}\n"
    "    ]\n"
    "  },\n"
    "  'libscope': {\n"
    "    'transport': {\n"
    "      'type': 'file',\n"
    "      'path': '/var/log/event.log'\n"
    "    },\n"
    "    'summaryperiod': '13',\n"
    "    'log': {\n"
    "      'level': 'debug',\n"
    "      'transport': {\n"
    "        'type': 'shm'\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";

static void
cfgReadGoodJson(void** state)
{
    const char* path = CFG_FILE_NAME;
    writeFile(path, jsonText);
    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_METRIC_JSON);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 42);
    assert_int_equal(cfgOutVerbosity(config), 0);
    assert_int_equal(cfgOutPeriod(config), 13);
    assert_int_equal(cfgEventFormat(config), CFG_EVENT_ND_JSON);
    assert_string_equal(cfgEventNameFilter(config, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_CONSOLE), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_OUT), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_OUT), "8125");
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/log/scope.log");
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_FILE);
    assert_string_equal(cfgTransportHost(config, CFG_CTL), "127.0.0.1");
    assert_string_equal(cfgTransportPort(config, CFG_CTL), "9109");
    assert_string_equal(cfgTransportPath(config, CFG_CTL), "/var/log/event.log");
    assert_int_equal(cfgTransportType(config, CFG_LOG), CFG_SHM);
    assert_null(cfgTransportHost(config, CFG_LOG));
    assert_null(cfgTransportPort(config, CFG_LOG));
    assert_string_equal(cfgTransportPath(config, CFG_LOG), "/tmp/scope.log");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "tagA"), "val1");
    assert_string_equal(cfgCustomTagValue(config, "tagB"), "val2");
    assert_string_equal(cfgCustomTagValue(config, "tagC"), "val3");
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
        "metric:\n"
        "  format: metricjson\n"
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
        "metric:\n"
        "  format:\n"
        "    type: metricstatsd\n"
        "    hey: yeahyou\n"
        "    tags:\n"
        "    - brainfarts: 135\n"
        "  request: 'make it snappy'        # Extra.\n"
        "  transport:\n"
        "    type: unix\n"
        "    path: '/var/run/scope.sock'\n"
        "    color: 'puce'                  # Extra.\n"
        "libscope:\n"
        "  log:\n"
        "    level: info\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_METRIC_STATSD);
    assert_string_equal(cfgOutStatsDPrefix(config), DEFAULT_STATSD_PREFIX);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/run/scope.sock");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "brainfarts"), "135");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadYamlOrderWithinStructureDoesntMatter(void** state)
{
    const char* yamlText =
        "---\n"
        "event:\n"
        "  watch:\n"
        "    - name: .*[.]log$\n"
        "      type: syslog\n"
        "      field: .*host.*\n"
        "      value: '[0-9]+'\n"
        "    - type: file\n"
        "    - type: metric\n"
        "  format:\n"
        "    type : metricjson\n"
        "libscope:\n"
        "  log:\n"
        "    level: info\n"
        "  summaryperiod: 42\n"
        "  transport:\n"
        "    type: syslog                    # udp, unix, file, syslog\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "metric:\n"
        "  transport:\n"
        "    path: '/var/run/scope.sock'\n"
        "    type: unix\n"
        "  format:\n"
        "    tags:\n"
        "    - 135: kittens\n"
        "    verbosity: 4294967295\n"
        "    statsdmaxlen: 4294967295\n"
        "    statsdprefix: 'cribl.scope'\n"
        "    type:  metricstatsd\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);

    config_t* config = cfgRead(path);
    assert_non_null(config);
    assert_int_equal(cfgOutFormat(config), CFG_METRIC_STATSD);
    assert_string_equal(cfgOutStatsDPrefix(config), "cribl.scope.");
    assert_int_equal(cfgOutStatsDMaxLen(config), 4294967295);
    assert_int_equal(cfgOutVerbosity(config), CFG_MAX_VERBOSITY);
    assert_int_equal(cfgOutPeriod(config), 42);
    assert_int_equal(cfgEventFormat(config), CFG_METRIC_JSON);
    assert_string_equal(cfgEventNameFilter(config, CFG_SRC_SYSLOG), ".*[.]log$");
    assert_string_equal(cfgEventFieldFilter(config, CFG_SRC_SYSLOG), ".*host.*");
    assert_string_equal(cfgEventValueFilter(config, CFG_SRC_SYSLOG), "[0-9]+");
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_FILE), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_CONSOLE), 0);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_SYSLOG), 1);
    assert_int_equal(cfgEventSourceEnabled(config, CFG_SRC_METRIC), 1);
    assert_int_equal(cfgTransportType(config, CFG_CTL), CFG_SYSLOG);
    assert_int_equal(cfgTransportType(config, CFG_OUT), CFG_UNIX);
    assert_string_equal(cfgTransportPath(config, CFG_OUT), "/var/run/scope.sock");
    assert_non_null(cfgCustomTags(config));
    assert_string_equal(cfgCustomTagValue(config, "135"), "kittens");
    assert_int_equal(cfgLogLevel(config), CFG_LOG_INFO);

    cfgDestroy(&config);
    deleteFile(path);
}

static void
cfgReadEnvSubstitution(void** state)
{

    // Set env varibles to test indirect substitution
    assert_int_equal(setenv("VAR1", "longer", 1), 0);
    assert_int_equal(setenv("MY_ENV_VAR", "shorter", 1), 0);
    assert_int_equal(setenv("MAXLEN", "1024", 1), 0);
    assert_int_equal(setenv("DEST", "/tmp/file.tmp2", 1), 0);
    assert_int_equal(setenv("PERIOD", "11", 1), 0);
    assert_int_equal(setenv("MYHOME", "home/mydir", 1), 0);
    assert_int_equal(setenv("VERBOSITY", "1", 1), 0);
    assert_int_equal(setenv("LOGLEVEL", "trace", 1), 0);
    assert_int_equal(setenv("FORMAT", "metricstatsd", 1), 0);
    assert_int_equal(setenv("FILTER", ".*[.]log$", 1), 0);
    assert_int_equal(setenv("SOURCE", "syslog", 1), 0);

    const char* yamlText =
        "---\n"
        "metric:\n"
        "  format:\n"
        "    type: metricjson\n"
        "    statsdprefix : $VAR1.$MY_ENV_VAR\n"
        "    statsdmaxlen : $MAXLEN\n"
        "    verbosity: $VERBOSITY\n"
        "    tags:\n"
        "    - CUSTOM: $PERIOD\n"
        "    - whyyoumadbro: 'Bill owes me $5.00'\n"
        "    - undefined: $UNDEFINEDENV\n"
        "  transport:\n"
        "    type: file\n"
        "    path: /\\$VAR1/$MY_ENV_VAR/\n"
        "    buffering: line\n"
        "event:\n"
        "  format:\n"
        "    type : $FORMAT\n"
        "  watch:\n"
        "    - type: file                    # create events from files\n"
        "      name: $FILTER\n"
        "    - type: console                 # create events from stdout and stderr\n"
        "    - type: $SOURCE                 # create events from syslog and vsyslog\n"
        "    - type: metric\n"
        "libscope:\n"
        "  transport:\n"
        "    type: syslog                    # udp, unix, file, syslog\n"
        "    host: 127.0.0.2\n"
        "    port: 9009\n"
        "    buffering: line\n"
        "  summaryperiod: $PERIOD\n"
        "  commanddir: /$MYHOME/scope/\n"
        "  log:\n"
        "    level: $LOGLEVEL\n"
        "    transport:\n"
        "      buffering: full\n"
        "      type: file\n"
        "      path: $DEST\n"
        "...\n";
    const char* path = CFG_FILE_NAME;
    writeFile(path, yamlText);
    config_t* cfg = cfgRead(path);
    assert_non_null(cfg);

    // test substitute env values that are longer and shorter than they env name
    assert_string_equal(cfgOutStatsDPrefix(cfg), "longer.shorter.");
    assert_int_equal(cfgOutStatsDMaxLen(cfg), 1024);
    assert_int_equal(cfgOutPeriod(cfg), 11);
    assert_string_equal(cfgCmdDir(cfg), "/home/mydir/scope/");
    assert_int_equal(cfgOutVerbosity(cfg), 1);
    // test escaped substitution  (a match preceeded by '\')
    assert_string_equal(cfgTransportPath(cfg, CFG_OUT), "/$VAR1/shorter/");
    assert_string_equal(cfgTransportPath(cfg, CFG_LOG), "/tmp/file.tmp2");
    assert_string_equal(cfgCustomTagValue(cfg, "CUSTOM"), "11");
    // test lookups that aren't found: $5 and $UNDEFINEDENV
    assert_string_equal(cfgCustomTagValue(cfg, "whyyoumadbro"), "Bill owes me $5.00");
    assert_string_equal(cfgCustomTagValue(cfg, "undefined"), "$UNDEFINEDENV");
    assert_int_equal(cfgLogLevel(cfg), CFG_LOG_TRACE);
    // test event fields...
    assert_int_equal(cfgEventFormat(cfg), CFG_METRIC_STATSD);
    assert_string_equal(cfgEventNameFilter(cfg, CFG_SRC_FILE), ".*[.]log$");
    assert_int_equal(cfgEventSourceEnabled(cfg, CFG_SRC_SYSLOG), 1);

    cfgDestroy(&cfg);

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

    deleteFile(path);
}

static void
jsonObjectFromCfgAndjsonStringFromCfgRoundTrip(void** state)
{
    // Start with a string, just since it's already defined for another test
    config_t* cfg = cfgFromString(jsonText);
    assert_non_null(cfg);

    // Now from the cfg object above, we should be able to create a
    // new string and json object with the same content
    char* stringified_json1 = jsonStringFromCfg(cfg);
    assert_non_null(stringified_json1);
    cJSON* json1 = jsonObjectFromCfg(cfg);
    assert_non_null(json1);

    // Do this again with the new string we output this time
    cfgDestroy(&cfg);
    cfg = cfgFromString(stringified_json1);
    assert_non_null(cfg);

    char* stringified_json2 = jsonStringFromCfg(cfg);
    assert_non_null(stringified_json2);
    cJSON* json2 = jsonObjectFromCfg(cfg);
    assert_non_null(json2);

    // now the diff to make sure the strings and json object trees are identical
    assert_string_equal(stringified_json1, stringified_json2);
    assert_true(cJSON_Compare(json1, json2, 1)); // case-sensitive comparison

    //printf("%s\n", stringified_json1);

    cfgDestroy(&cfg);
    cJSON_Delete(json1);
    cJSON_Delete(json2);
    free(stringified_json1);
    free(stringified_json2);
}


static void
initLogReturnsPtr(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_SHM; t++) {
	    switch (t) {
            case CFG_UDP:
                cfgTransportTypeSet(cfg, CFG_LOG, t);
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
            case CFG_TCP:
                break;
	    }
        cfgTransportTypeSet(cfg, CFG_LOG, t);
        log_t* log = initLog(cfg);
        assert_non_null(log);
        logDestroy(&log);
    }
    cfgDestroy(&cfg);
}

static void
initOutReturnsPtr(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    cfg_transport_t t;
    for (t=CFG_UDP; t<=CFG_SHM; t++) {
        cfgTransportTypeSet(cfg, CFG_OUT, t);
        if (t==CFG_UNIX || t==CFG_FILE) {
            cfgTransportPathSet(cfg, CFG_OUT, "/tmp/scope.log");
        }
        out_t* out = initOut(cfg);
        assert_non_null(out);
        outDestroy(&out);
    }
    cfgDestroy(&cfg);
}

static void
initEvtReturnsPtr(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    evt_t* evt = initEvt(cfg);
    assert_non_null(evt);
    evtDestroy(&evt);

    cfgDestroy(&cfg);
}

static void
initCtlReturnsPtr(void** state)
{
    config_t* cfg = cfgCreateDefault();
    assert_non_null(cfg);

    ctl_t* ctl = initCtl(cfg);
    assert_non_null(ctl);
    ctlDestroy(&ctl);

    cfgDestroy(&cfg);
}

// Defined in src/cfgutils.c
// This is not a proper test, it just exists to make valgrind output
// more readable when analyzing this test, by deallocating the compiled
// regex in src/cfgutils.c.
extern void envRegexFree(void** state);


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    source_state_t log = {"SCOPE_EVENT_LOGFILE", CFG_SRC_FILE, DEFAULT_SRC_FILE};
    source_state_t con = {"SCOPE_EVENT_CONSOLE", CFG_SRC_CONSOLE, DEFAULT_SRC_CONSOLE};
    source_state_t sys = {"SCOPE_EVENT_SYSLOG" , CFG_SRC_SYSLOG , DEFAULT_SRC_SYSLOG};
    source_state_t met = {"SCOPE_EVENT_METRIC", CFG_SRC_METRIC , DEFAULT_SRC_METRIC};

    dest_state_t dest_out = {"SCOPE_METRIC_DEST", CFG_OUT};
    dest_state_t dest_evt = {"SCOPE_EVENT_DEST", CFG_CTL};
    dest_state_t dest_log = {"SCOPE_LOG_DEST", CFG_LOG};

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(cfgPathHonorsEnvVar),
        cmocka_unit_test(cfgPathHonorsPriorityOrder),
        cmocka_unit_test(cfgProcessEnvironmentOutFormat),
        cmocka_unit_test(cfgProcessEnvironmentStatsDPrefix),
        cmocka_unit_test(cfgProcessEnvironmentStatsDMaxLen),
        cmocka_unit_test(cfgProcessEnvironmentOutPeriod),
        cmocka_unit_test(cfgProcessEnvironmentCommandDir),
        cmocka_unit_test(cfgProcessEnvironmentEventFormat),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &log),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &con),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &sys),
        cmocka_unit_test_prestate(cfgProcessEnvironmentEventSource, &met),
        cmocka_unit_test(cfgProcessEnvironmentOutVerbosity),
        cmocka_unit_test(cfgProcessEnvironmentLogLevel),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_out),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_evt),
        cmocka_unit_test_prestate(cfgProcessEnvironmentTransport, &dest_log),
        cmocka_unit_test(cfgProcessEnvironmentStatsdTags),
        cmocka_unit_test(cfgProcessEnvironmentCmdDebugIsIgnored),
        cmocka_unit_test(cfgProcessCommandsCmdDebugIsProcessed),
        cmocka_unit_test(cfgProcessCommandsFromFile),
        cmocka_unit_test(cfgProcessCommandsEnvSubstitution),
        cmocka_unit_test(cfgReadGoodYaml),
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
        cmocka_unit_test(initOutReturnsPtr),
        cmocka_unit_test(initEvtReturnsPtr),
        cmocka_unit_test(initCtlReturnsPtr),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
        cmocka_unit_test(envRegexFree),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
