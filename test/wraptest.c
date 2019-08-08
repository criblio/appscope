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
extern int osInitTSC(struct config_t *);

// A port that is not likely to be used
#define PORT1 65430
#define PORT2 65431

config g_cfg = {0};

static void
testConnDuration(void** state)
{
    int rc, sdl, sds;
    struct sockaddr_in saddr;
    char *log, *last;
    const char* hostname = "127.0.0.1";
    const char delim[] = ":";
    char buf[128];
    
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

    FILE *fs = popen("grep wraptest /tmp/scope.log | grep duration | tail -n 1", "r");
    assert_non_null(fs);

    size_t len = fread(buf, sizeof(buf), (size_t)1, fs);
    printf("len %d %s\n", len, buf);
    //assert_int_not_equal(len, 0);

    log = strtok_r(buf, delim, &last);
    assert_non_null(log);
    log = strtok_r(NULL, delim, &last);
    assert_non_null(log);
    int duration = strtol(log, NULL, 0);
    printf("Duration: %d\n", duration);
    assert_int_not_equal(duration, 0);
    assert_true((duration > 1000) && (duration < 2000));
}

static void
testTSCInit(void** state)
{
    config cfg;
    
    assert_int_equal(osInitTSC(&cfg), 0);
}

static void
testTSCRollover(void** state)
{
    uint64_t elapsed, now = ULONG_MAX -2;
    elapsed = getDuration(now);
    assert_non_null(elapsed);
    printf("Now %"PRIu64" Elapsed %"PRIu64"\n", now, elapsed);
    assert_true(elapsed > 250000);
}

static void
testTSCValue(void** state)
{
    uint64_t now, elapsed;

    now = getTime();
    elapsed = getDuration(now);
    assert_non_null(elapsed);
    printf("Now %"PRIu64" Elapsed %"PRIu64"\n", now, elapsed);
    assert_true((elapsed < 250) && (elapsed > 20));
}

int
main (int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testTSCRollover),
        cmocka_unit_test(testTSCValue),
        cmocka_unit_test(testTSCInit),
        cmocka_unit_test(testConnDuration),
    };
    cmocka_run_group_tests(tests, NULL, NULL);
    return 0;
}
