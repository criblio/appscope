#define _GNU_SOURCE

#include <sys/wait.h>

#include "nsinfo.h"
#include "test.h"
#include "scopestdlib.h"


static void
nsInfoIsPidInSameMntNsSameProcess(void **state) {
    pid_t pid = scope_getpid();
    bool status = nsInfoIsPidInSameMntNs(pid);
    assert_int_equal(status, TRUE);
}

static void
nsInfoIsPidInSameMntNsChildProcess(void **state) {
    pid_t parentPid = scope_getpid();
    pid_t pid = fork();
    assert_int_not_equal(pid, -1);
    if (pid == 0) {
        bool status = nsInfoIsPidInSameMntNs(parentPid);
        assert_int_equal(status, TRUE);
    } else {
        int status = -1;
        pid_t pres = wait(&status);
        assert_int_not_equal(pres, -1);
    }
}

static void
nsInfoIsPidGotNestedPidNsSameProcess(void **state) {
    pid_t nsPid = 0;
    pid_t pid = scope_getpid();
    bool status = nsInfoIsPidGotNestedPidNs(pid, &nsPid);
    assert_int_equal(status, FALSE);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(nsInfoIsPidInSameMntNsSameProcess),
        cmocka_unit_test(nsInfoIsPidInSameMntNsChildProcess),
        cmocka_unit_test(nsInfoIsPidGotNestedPidNsSameProcess),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
