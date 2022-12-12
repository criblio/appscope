#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libver.h"
#include "test.h"
#include "scopetypes.h"
#include "scopestdlib.h"

static void
normalizedVersionOfficialTest(void **state) {
    const char *version = libverNormalizedVersion("v1.2.0");
    assert_string_equal(version, "1.2.0");
    bool status = libverIsNormVersionDev(version);
    assert_int_equal(status, FALSE);
}

static void
normalizedVersionRCTest(void **state) {
    const char *version = libverNormalizedVersion("v1.2.0-rc0");
    assert_string_equal(version, "1.2.0-rc0");
    bool status = libverIsNormVersionDev(version);
    assert_int_equal(status, FALSE);
}

static void
normalizedVersionTCTest(void **state) {
    const char *version = libverNormalizedVersion("v1.2.0-tc11");
    assert_string_equal(version, "1.2.0-tc11");
    bool status = libverIsNormVersionDev(version);
    assert_int_equal(status, FALSE);
}

static void
normalizedVersionDevTest(void **state) {
    const char *version = libverNormalizedVersion("web-1.1.3-239-g2dfb6670bc1f");
    assert_string_equal(version, "dev");
    bool status = libverIsNormVersionDev(version);
    assert_int_equal(status, TRUE);
}

static void
normalizedVersionDevTestWrongFormat(void **state) {
    const char *version = libverNormalizedVersion("vv1.2.0");
    assert_string_equal(version, "dev");
    version = libverNormalizedVersion("v1.a.0.");
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
        cmocka_unit_test(normalizedVersionRCTest),
        cmocka_unit_test(normalizedVersionTCTest),
        cmocka_unit_test(normalizedVersionDevTest),
        cmocka_unit_test(normalizedVersionNullTest),
        cmocka_unit_test(normalizedVersionDevTestWrongFormat),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
