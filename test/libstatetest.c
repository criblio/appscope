#define _GNU_SOURCE
#include "stdlib.h"
#include "libstate.h"
#include "dbg.h"
#include "scopestdlib.h"
#include "test.h"


static bool
isLoadedMappingPresent(pid_t pid) {
    FILE *fstream;
    char buffer[4096] = {0};
    char pidBuf[4096] = {0};
    bool status = FALSE;

    if (scope_snprintf(pidBuf, sizeof(pidBuf),"/tmp/scope_loaded.%d", pid) < 0) {
        scope_perror("failed to create mapping string failed");
        exit(EXIT_FAILURE);
    }

    if ((fstream = scope_fopen("/proc/self/maps", "r")) == NULL) {
        scope_perror("fopen /proc/self/maps failed");
        exit(EXIT_FAILURE);
    }

    while (scope_fgets(buffer, sizeof(buffer), fstream)) {
        if (scope_strstr(buffer, pidBuf)) {
            status = TRUE;
            break;
        }
    }

    scope_fclose(fstream);
    return status;
}


static void
libstateOnOff(void **state) {
    bool opRes;
    bool mapRes;
    pid_t pid = scope_getpid();

    opRes = libstateScoped();
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_false(mapRes);

    opRes = libstateLoaded(pid);
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_true(mapRes);
}

static void
libstateOffOn(void **state) {
    bool opRes;
    bool mapRes;
    pid_t pid = scope_getpid();

    opRes = libstateLoaded(pid);
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_true(mapRes);

    opRes = libstateScoped();
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_false(mapRes);
}


static void
libstateOffOff(void **state) {
    bool opRes;
    bool mapRes;
    pid_t pid = scope_getpid();

    opRes = libstateLoaded(pid);
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_true(mapRes);

    opRes = libstateLoaded(pid);
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_true(mapRes);
}

static void
libstateOnOn(void **state) {
    bool opRes;
    bool mapRes;
    pid_t pid = scope_getpid();

    opRes = libstateScoped();
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_false(mapRes);

    opRes = libstateScoped();
    assert_true(opRes);
    mapRes = isLoadedMappingPresent(pid);
    assert_false(mapRes);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(libstateOnOff),
        cmocka_unit_test(libstateOffOn),
        cmocka_unit_test(libstateOffOff),
        cmocka_unit_test(libstateOnOn),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
