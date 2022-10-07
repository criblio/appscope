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

static void
mkdirNestedWrongPathNull(void **state) {
    mkdir_status_t res = libverMkdirNested(NULL);
    assert_int_equal(res, MKDIR_STATUS_NOT_ABSOLUTE_DIR);
}

static void
mkdirNestedWrongPathCurrentDir(void **state) {
    mkdir_status_t res = libverMkdirNested(".");
    assert_int_equal(res, MKDIR_STATUS_NOT_ABSOLUTE_DIR);
}

static void
mkdirNestedWrongPathFile(void **state) {
    int res;
    char buf[PATH_MAX] = {0};
    const char *fileName = "unitTestLibVerFile";

    int fd = scope_open(fileName, O_RDWR|O_CREAT, 0777);
    assert_int_not_equal(fd, -1);
    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, fileName);
    mkdir_status_t status = libverMkdirNested(buf);
    assert_int_equal(status, MKDIR_STATUS_NOT_ABSOLUTE_DIR);
    res = scope_unlink(fileName);
    assert_int_equal(res, 0);
}

static void
mkdirNestedPermissionIssue(void **state) {
    mkdir_status_t res = libverMkdirNested("/root/loremIpsumFile/");
    assert_int_equal(res, MKDIR_STATUS_OTHER_ISSUE);
}

static void
mkdirNestedAlreadyExists(void **state) {
    int res;
    char buf[PATH_MAX] = {0};
    const char *dirName = "unitTestLibVerDir";

    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, dirName);
    res = scope_mkdir(buf, 0755);
    assert_int_equal(res, 0);
    mkdir_status_t status = libverMkdirNested(dirName);
    assert_int_equal(status, MKDIR_STATUS_NOT_ABSOLUTE_DIR);
    status = libverMkdirNested(buf);
    assert_int_equal(status, MKDIR_STATUS_EXISTS);
    res = scope_rmdir(buf);
    assert_int_equal(res, 0);
}

static void
mkdirNestedSuccessCreated(void **state) {
    int res;
    struct stat dirStat = {0};
    char buf[PATH_MAX] = {0};
    const char *dirName = "unitTestLibVerDir";

    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, dirName);
    res = scope_stat(buf, &dirStat);
    assert_int_not_equal(res, 0);
    mkdir_status_t status = libverMkdirNested(buf);
    assert_int_equal(status, MKDIR_STATUS_CREATED);
    res = scope_stat(buf, &dirStat);
    assert_int_equal(res, 0);
    assert_int_equal(S_ISDIR(dirStat.st_mode), 1);
    res = scope_rmdir(buf);
    assert_int_equal(res, 0);
}


int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(normalizedVersionOfficialTest),
        cmocka_unit_test(normalizedVersionDevTest),
        cmocka_unit_test(normalizedVersionNullTest),
        cmocka_unit_test(mkdirNestedWrongPathNull),
        cmocka_unit_test(mkdirNestedWrongPathFile),
        cmocka_unit_test(mkdirNestedWrongPathCurrentDir),
        cmocka_unit_test(mkdirNestedPermissionIssue),
        cmocka_unit_test(mkdirNestedAlreadyExists),
        cmocka_unit_test(mkdirNestedSuccessCreated),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
