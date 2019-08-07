#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

#include "test.h"
#include "wrap.h"

extern uint64_t getDuration(uint64_t);

config g_cfg = {0};

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
    };
    cmocka_run_group_tests(tests, NULL, NULL);
    return 0;
}
