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
        cmocka_unit_test(initLogReturnsPtr),
        cmocka_unit_test(initOutReturnsPtrWithNullLogReference),
        cmocka_unit_test(initOutReturnsPtrWithLogReference),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}


