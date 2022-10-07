#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libver.h"
#include "test.h"
#include "scopestdlib.h"

static void
normalizedVersionOfficialTest(void **state) {
    const char *version = libverNormalizedVersion("v1.2.0");
    assert_string_equal(version, "1.2.0");
}

static void
normalizedVersionDevTest(void **state) {
    const char *version = libverNormalizedVersion("web-1.1.3-239-g2dfb6670bc1f");
    assert_string_equal(version, "dev");
}

static void
normalizedVersionNullTest(void **state) {
    const char *version = libverNormalizedVersion(NULL);
    assert_string_equal(version, "dev");
}


int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(normalizedVersionOfficialTest),
        cmocka_unit_test(normalizedVersionDevTest),
        cmocka_unit_test(normalizedVersionNullTest),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
