#define _GNU_SOURCE

#include <sys/wait.h>

#include "ns.h"
#include "test.h"
#include "scopestdlib.h"

/*
 * Define the extern offset for integration test compilation 
 * See details in libdir.c
 */
unsigned char _binary_ldscopedyn_start;
unsigned char _binary_ldscopedyn_end;
unsigned char _binary_libscope_so_start;
unsigned char _binary_libscope_so_end;

static void
nsIsPidInSameMntNsSameProcess(void **state) {
    pid_t pid = scope_getpid();
    bool status = nsIsPidInSameMntNs(pid);
    assert_int_equal(status, TRUE);
}

static void
nsIsPidInSameMntNsChildProcess(void **state) {
    pid_t parentPid = scope_getpid();
    pid_t pid = fork();
    assert_int_not_equal(pid, -1);
    if (pid == 0) {
        bool status = nsIsPidInSameMntNs(parentPid);
        assert_int_equal(status, TRUE);
    } else {
        int status = -1;
        pid_t pres = wait(&status);
        assert_int_not_equal(pres, -1);
    }
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(nsIsPidInSameMntNsSameProcess),
        cmocka_unit_test(nsIsPidInSameMntNsChildProcess),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
