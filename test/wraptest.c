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
#include <sched.h>

#include "test.h"
#include "cfg.h"
#include "cfgutils.h"
#include "dbg.h"
#include "scopetypes.h"
#include "wrap.h"

extern uint64_t getDuration(uint64_t);
extern int osInitTSC(struct rtconfig_t *);

// A port that is not likely to be used
#define PORT1 65430
#define PORT2 65431
#define LIKELY_FILE_SIZE 12000
#define NET_DURATION 2000
#define FS_DURATION 100

rtconfig g_cfg = {0};

/*
 * The env var SCOPE_HOME is set in
 * the Makefile or script that runs 
 * this test. It points to a config
 * file in scope/test/conf/scope.cfg.
 * Using a config file for test we ensure
 * we have debug logs enabled and that 
 * we know the path to the log file
 * and to a file that contains metrics. 
 */
static void
testFSDuration(void** state)
{
    int rc, fd;
    char *log, *last;
    FILE *fs;
    const char delim[] = ":";
    char buf[1024];
    char* cpath = cfgPath(CFG_FILE_NAME);
    config_t* cfg = cfgRead(cpath);
    const char *path = cfgTransportPath(cfg, CFG_OUT);
    assert_int_equal(cfgOutVerbosity(cfg), CFG_NET_FS_EVENTS_VERBOSITY);
   
    // Start the duration timer with a read
    fd = open("./scope.sh", O_RDONLY);
    assert_return_code(fd, errno);

    rc = read(fd, buf, 16);
    assert_return_code(rc, errno);

    rc = close(fd);
    assert_return_code(rc, errno);

    // Read the metrics file
    fs = fopen(path, "r");
    assert_non_null(fs);
    
    fread(buf, sizeof(buf), (size_t)1, fs);
    assert_int_equal(ferror(fs), 0);
    //printf("%s\n", buf);

    rc = fclose(fs);
    assert_return_code(rc, errno);

    char *start = strstr(buf, "duration");
    log = strtok_r(start, delim, &last);
    assert_non_null(log);
    log = strtok_r(NULL, delim, &last);
    assert_non_null(log);
    int duration = strtol(log, NULL, 0);
    if ((duration < 1) || (duration > FS_DURATION))
        fail_msg("Duration %d is outside of allowed bounds (1, 100)", duration);
}

/*
 * The env var SCOPE_HOME is set in
 * the Makefile or script that runs 
 * this test. It points to a config
 * file in scope/test/conf/scope.cfg.
 * Using a config file for test we ensure
 * we have debug logs enabled and that 
 * we know the path to the log file
 * and a metrics file. 
 */
static void
testConnDuration(void** state)
{
    int rc, sdl, sds;
    struct sockaddr_in saddr;
    char *log, *last;
    FILE *fs;
    const char* hostname = "127.0.0.1";
    const char delim[] = ":";
    char *buf;
    char* cpath = cfgPath(CFG_FILE_NAME);
    config_t* cfg = cfgRead(cpath);
    const char *path = cfgTransportPath(cfg, CFG_OUT);
    assert_int_equal(cfgOutVerbosity(cfg), CFG_NET_FS_EVENTS_VERBOSITY);

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

    fs = fopen(path, "r");
    assert_non_null(fs);

    // Yuck! should stat the file, but issues with stat on Ubuntu 19
    // Talk to John
    buf = malloc(LIKELY_FILE_SIZE);
    assert_non_null(buf);

    fread(buf, LIKELY_FILE_SIZE, (size_t)1, fs);
    assert_int_equal(ferror(fs), 0);
    //printf("%s\n", buf);

    rc = fclose(fs);
    assert_return_code(rc, errno);

    char *start = strstr(buf, "conn_duration");

    log = strtok_r(start, delim, &last);
    assert_non_null(log);
    log = strtok_r(NULL, delim, &last);
    assert_non_null(log);
    int duration = strtol(log, NULL, 0);
    if ((duration < 1000) || (duration > NET_DURATION))
        fail_msg("Duration %d is outside of allowed bounds (1000, 1300)", duration);

    free(buf);

    // Delete these after tests that use the metrics and log files are done
    assert_return_code(unlink(path), errno);
    assert_return_code(unlink(cfgTransportPath(cfg, CFG_LOG)), errno);
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
    if (elapsed < 250000)
        fail_msg("Elapsed %" PRIu64 " is less than allowed 250000", elapsed);
}

static void
testTSCValue(void** state)
{
    uint64_t now, elapsed;

    now = getTime();
    elapsed = getDuration(now);
    if ((elapsed < 20) || (elapsed > 1000))
        fail_msg("Elapsed %" PRIu64 " is outside of allowed bounds (20, 350)", elapsed);
}

/*

This applies to __LINUX__ only.  And isn't like other tests in this file.
Think about whether this adds enough value to run in conditionally only 
for linux.

static void
testSyscallDbg(void** state)
{
    const char* path = "/tmp/syscalltestoutput.txt";
    int fd;
    struct stat statbuf;

    assert_int_equal(dbgCountMatchingLines("src/wrap.c"), 0);

    assert_return_code(fd = syscall(SYS_openat, AT_FDCWD, path, O_WRONLY|O_CREAT|O_APPEND, 0666), errno);
    assert_return_code(syscall(SYS_lseek, fd, 0, SEEK_END), errno);
    assert_return_code(syscall(SYS_fstat, fd, &statbuf), errno);
    assert_return_code(syscall(SYS_write, fd, "Hey!\n", sizeof("Hey!\n")), errno);
    assert_return_code(syscall(SYS_close, fd), errno);

    // I expect one DBG statement to be hit 5 times, for the 5 syscalls above.
    unsigned long long lines = dbgCountMatchingLines("src/wrap.c");
    char buf[4096] = {0};
    dbgDumpAllToBuffer(buf, sizeof(buf));
    if ((lines != 1) || !strstr(buf, "5: src/wrap.c:")) {
        printf("%s", buf);
        fail();
    }
    dbgInit();

    assert_return_code(unlink(path), errno);
}

*/

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testFSDuration),
        cmocka_unit_test(testConnDuration),
        cmocka_unit_test(testTSCInit),
        cmocka_unit_test(testTSCRollover),
        cmocka_unit_test(testTSCValue),
        //cmocka_unit_test(testSyscallDbg),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
