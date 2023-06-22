#define _GNU_SOURCE

#include "sig.h"
#include "scopestdlib.h"
#include "fn.h"
#include "test.h"

static bool scopeSignal = FALSE;

static void
dummyHandler(int sig, siginfo_t *info, void *secret) {
    assert_int_equal(sig, SIGUSR2);
    if (scopeSignal) {
        bool res = sigIsSigFromAppscopeTimer(info);
        assert_true(res);
    }
}

static void
sigTestTimerStopNotInit(void **state) {
    bool res = sigTimerStop();
    assert_false(res);
}

static void
sigTestTimerBasic(void **state) {
    scopeSignal = TRUE;
    bool res = sigHandlerRegister(SIGUSR2, dummyHandler);
    assert_true(res);
    res = sigTimerStart(SIGUSR2, 1);
    assert_true(res);
    scope_sleep(2);
    scopeSignal = FALSE;
    res = sigTimerStop();
    assert_true(res);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(sigTestTimerBasic),
        cmocka_unit_test(sigTestTimerStopNotInit),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
