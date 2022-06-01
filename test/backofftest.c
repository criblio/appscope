#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include "backoff.h"

#include "scopestdlib.h"
#include "test.h"

static void
backoffInitOK(void** state)
{
    const unsigned attempt_limit = 1;
    const unsigned base_delay = 5;
    const unsigned max_delay = 10;
    const unsigned seed = 0;

    int res = backoffInit(attempt_limit, base_delay, max_delay, seed);
    assert_int_equal(res, BACKOFF_OK);
}

static void
backoffInitError(void** state)
{
    const unsigned attempt_limit = 1;
    const unsigned base_delay = 10;
    const unsigned max_delay = 5;
    const unsigned seed = 0;

    int res = backoffInit(attempt_limit, base_delay, max_delay, seed);
    assert_int_equal(res, BACKOFF_ERROR);
}

static void
backoffRetryLimitHit(void** state)
{
    const unsigned attempt_limit = 2;
    const unsigned base_delay = 5;
    const unsigned max_delay = 10;
    const unsigned seed = 0;
    unsigned dummy_time = 0;

    int res = backoffInit(attempt_limit, base_delay, max_delay, seed);
    assert_int_equal(res, BACKOFF_OK);
    for (unsigned i = 0; i< attempt_limit; ++i) {
        res = backoffGetTime(&dummy_time);
        assert_int_equal(res, BACKOFF_OK);
    }
    res = backoffGetTime(&dummy_time);
    assert_int_equal(res, BACKOFF_RETRY_LIMIT);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(backoffInitOK),
        cmocka_unit_test(backoffInitError),
        cmocka_unit_test(backoffRetryLimitHit),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
