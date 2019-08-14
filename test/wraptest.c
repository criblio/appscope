#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "test.h"
#include "wrap.h"

extern uint64_t getDuration(uint64_t);
extern int osInitTSC(struct rtconfig_t *);

// A port that is not likely to be used
#define PORT1 65430
#define PORT2 65431
#define LOG_PATH "/test/conf/scope.log"

rtconfig g_cfg = {0};

static void
testFSDuration(void** state)
{
    int rc, fd;
    char *log, *last;
    const char delim[] = ":";
    char buf[1024];
    char path[PATH_MAX];

    /*
     * The env var SCOPE_HOME is set in
     * the Makefile or script that runs 
     * this test. It points to a config
     * file in scope/test/conf/scope.cfg.
     * Using a config file for test we ensure
     * we have debug logs enabled and that 
     * we know the path to the log file. 
     */

    // Now get the path to the log file
    last = getcwd(path, sizeof(path));
    assert_non_null(last);
    strcat(path, LOG_PATH);

    // Start the duration timer with a read
    fd = open("./scope.sh", O_RDONLY);
    assert_return_code(fd, errno);

    rc = read(fd, buf, 16);
    assert_return_code(rc, errno);
    
    rc = close(fd);
    assert_return_code(rc, errno);

    // In macOS, the makefile is setting DYLD_INSERT_LIBRARIES.  We need to 
    // clear it to avoid an infinite loop where by reading the log file, 
    // grep adds to the log file.  On linux, unset should be harmless.
    snprintf(buf, strlen(path) + 128, "unset DYLD_INSERT_LIBRARIES ; grep wraptest %s | grep duration | tail -n 1", path);
    FILE *fs = popen(buf, "r");
    assert_non_null(fs);

    size_t len = fread(buf, sizeof(buf), (size_t)1, fs);
    //printf("len %ld %s\n", len, buf);
    assert_int_equal(len, 0);

    log = strtok_r(buf, delim, &last);
    assert_non_null(log);
    log = strtok_r(NULL, delim, &last);
    assert_non_null(log);
    int duration = strtol(log, NULL, 0);
    //printf("Duration: %d\n", duration);
    assert_int_not_equal(duration, 0);
    assert_true((duration > 0) && (duration < 100));
}

static void
testConnDuration(void** state)
{
    int rc, sdl, sds;
    struct sockaddr_in saddr;
    char *log, *last;
    const char* hostname = "127.0.0.1";
    const char delim[] = ":";
    char buf[1024];
    char path[PATH_MAX];

    /*
     * The env var SCOPE_HOME is set in
     * the Makefile or script that runs 
     * this test. It points to a config
     * file in scope/test/conf/scope.cfg.
     * Using a config file for test we ensure
     * we have debug logs enabled and that 
     * we know the path to the log file. 
     */

    // Now get the path to the log file
    last = getcwd(path, sizeof(path));
    assert_non_null(last);
    strcat(path, LOG_PATH);

    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(PORT1);
    saddr.sin_addr.s_addr = inet_addr(hostname);

    // Create a listen socket
    sdl = socket(AF_INET, SOCK_STREAM, 0);
    assert_return_code(sdl, errno);

    rc = bind(sdl, (const struct sockaddr *)&saddr, sizeof(saddr));
    assert_return_code(rc, errno);
    
    rc = listen(sdl, 2);
    assert_return_code(rc, errno);

    // create a send socket
    sds = socket(AF_INET, SOCK_STREAM, 0);
    assert_return_code(sds, errno);

    saddr.sin_port = htons(PORT2);
    rc = bind(sds, (const struct sockaddr *)&saddr, sizeof(saddr));
    assert_return_code(rc, errno);
    
    
    // Start the duration timer
    saddr.sin_port = htons(PORT1);
    rc = connect(sds, (const struct sockaddr *)&saddr, sizeof(saddr));
    assert_return_code(rc, errno);

    // Create some time
    sleep(1);
    
    // Stop the duration timer
    rc = close(sds);
    assert_return_code(rc, errno);

    rc = close(sdl);
    assert_return_code(rc, errno);

    // In macOS, the makefile is setting DYLD_INSERT_LIBRARIES.  We need to 
    // clear it to avoid an infinite loop where by reading the log file, 
    // grep adds to the log file.  On linux, unset should be harmless.
    snprintf(buf, strlen(path) + 128, "unset DYLD_INSERT_LIBRARIES ; grep wraptest %s | grep duration | tail -n 1", path);
    FILE *fs = popen(buf, "r");
    assert_non_null(fs);

    size_t len = fread(buf, sizeof(buf), (size_t)1, fs);
    //printf("len %ld %s\n", len, buf);
    assert_int_equal(len, 0);

    log = strtok_r(buf, delim, &last);
    assert_non_null(log);
    log = strtok_r(NULL, delim, &last);
    assert_non_null(log);
    int duration = strtol(log, NULL, 0);
    //printf("Duration: %d\n", duration);
    assert_int_not_equal(duration, 0);
    assert_true((duration > 1000) && (duration < 1300));
}

static void
testTSCInit(void** state)
{
    rtconfig cfg;
    
    assert_int_equal(osInitTSC(&cfg), 0);
}

static void
testTSCRollover(void** state)
{
    uint64_t elapsed, now = ULONG_MAX -2;
    elapsed = getDuration(now);
    assert_non_null(elapsed);
    //printf("Now %"PRIu64" Elapsed %"PRIu64"\n", now, elapsed);
    assert_true(elapsed > 250000);
}

static void
testTSCValue(void** state)
{
    uint64_t now, elapsed;

    now = getTime();
    elapsed = getDuration(now);
    assert_non_null(elapsed);
    //printf("Now %"PRIu64" Elapsed %"PRIu64"\n", now, elapsed);
    assert_true((elapsed < 350) && (elapsed > 20));
}

int
main (int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testFSDuration),
        cmocka_unit_test(testConnDuration),
        cmocka_unit_test(testTSCInit),
        cmocka_unit_test(testTSCRollover),
        cmocka_unit_test(testTSCValue),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
