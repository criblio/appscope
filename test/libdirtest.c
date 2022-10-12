#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libdir.h"
#include "libver.h"
#include "test.h"
#include "scopestdlib.h"

#define TEST_INSTALL_BASE "/tmp/appscope-test/install"
#define TEST_TMP_BASE "/tmp/appscope-test/tmp"

/*
 * Define the extern offset for integration test compilation 
 * See details in libdir.c
 */
unsigned char _binary_ldscopedyn_start;
unsigned char _binary_ldscopedyn_end;
unsigned char _binary_libscope_so_start;
unsigned char _binary_libscope_so_end;

static void
CreateDirIfMissingWrongPathNull(void **state) {
    mkdir_status_t res = libdirCreateDirIfMissing(NULL);
    assert_int_equal(res, MKDIR_STATUS_ERR_NOT_ABS_DIR);
}

static void
CreateDirIfMissingWrongPathCurrentDir(void **state) {
    mkdir_status_t res = libdirCreateDirIfMissing(".");
    assert_int_equal(res, MKDIR_STATUS_ERR_NOT_ABS_DIR);
}

static void
CreateDirIfMissingWrongPathFile(void **state) {
    int res;
    char buf[PATH_MAX] = {0};
    const char *fileName = "unitTestLibVerFile";

    int fd = scope_open(fileName, O_RDWR|O_CREAT, 0777);
    assert_int_not_equal(fd, -1);
    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, fileName);
    mkdir_status_t status = libdirCreateDirIfMissing(buf);
    assert_int_equal(status, MKDIR_STATUS_ERR_NOT_ABS_DIR);
    res = scope_unlink(fileName);
    assert_int_equal(res, 0);
}

static void
CreateDirIfMissingPermissionIssue(void **state) {
    mkdir_status_t res = libdirCreateDirIfMissing("/root/loremIpsumFile/");
    assert_int_equal(res, MKDIR_STATUS_ERR_OTHER);
}

static void
CreateDirIfMissingAlreadyExists(void **state) {
    int res;
    char buf[PATH_MAX] = {0};
    const char *dirName = "unitTestLibVerDir";

    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, dirName);
    res = scope_mkdir(buf, 0755);
    assert_int_equal(res, 0);
    mkdir_status_t status = libdirCreateDirIfMissing(dirName);
    assert_int_equal(status, MKDIR_STATUS_ERR_NOT_ABS_DIR);
    status = libdirCreateDirIfMissing(buf);
    assert_int_equal(status, MKDIR_STATUS_EXISTS);
    res = scope_rmdir(buf);
    assert_int_equal(res, 0);
}

static void
CreateDirIfMissingAlreadyExistsPermIssue(void **state) {
    mkdir_status_t status = libdirCreateDirIfMissing("/root");
    assert_int_equal(status, MKDIR_STATUS_ERR_PERM_ISSUE);
}

static void
CreateDirIfMissingSuccessCreated(void **state) {
    int res;
    struct stat dirStat = {0};
    char buf[PATH_MAX] = {0};
    const char *dirName = "unitTestLibVerDir";

    assert_non_null(scope_getcwd(buf, PATH_MAX));
    scope_strcat(buf, "/");
    scope_strcat(buf, dirName);
    res = scope_stat(buf, &dirStat);
    assert_int_not_equal(res, 0);
    mkdir_status_t status = libdirCreateDirIfMissing(buf);
    assert_int_equal(status, MKDIR_STATUS_CREATED);
    res = scope_stat(buf, &dirStat);
    assert_int_equal(res, 0);
    assert_int_equal(S_ISDIR(dirStat.st_mode), 1);
    res = scope_rmdir(buf);
    assert_int_equal(res, 0);
}

static void
SetLibraryBaseInvalid(void **state) {
    int res = libdirSetLibraryBase("/does_not_exist");
    assert_int_equal(res, -1);
}

static void
SetLibraryBaseNull(void **state) {
    int res = libdirSetLibraryBase("");
    assert_int_equal(res, -1);
}

static void
ExtractNewFile(void **state) {
    libdirInit(TEST_INSTALL_BASE, TEST_TMP_BASE);
    const char *ver = libverNormalizedVersion(SCOPE_VER);
    char expected_location[PATH_MAX];
    struct stat dirStat = {0};

    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, ver, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    // scope_remove(expected_location);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, ver, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    // scope_remove(expected_location);
}

// depends on ExtractNewFile
static void
ExtractFileExists(void **state) {
    libdirInit(TEST_INSTALL_BASE, TEST_TMP_BASE);
    const char *ver = libverNormalizedVersion(SCOPE_VER);
    char expected_location[PATH_MAX];
    struct stat dirStat = {0};

    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, ver, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, ver, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
}

// depends on ExtractNewFile
static void
GetPath(void **state) {
    const char *ver = libverNormalizedVersion(SCOPE_VER);
    char expected_location[PATH_MAX];

    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, ver, "ldscopedyn");
    const char *path = libdirGetPath(LOADER_FILE);
    int res = scope_strcmp(expected_location, path);
    assert_int_equal(res, 0);
}

static void
GetPathNoFile(void **state) {
    libdirInit(TEST_INSTALL_BASE, TEST_TMP_BASE);

    const char *loader_path = libdirGetPath(LOADER_FILE);
    assert_int_equal(loader_path, NULL);

    const char *library_path = libdirGetPath(LIBRARY_FILE);
    assert_int_equal(library_path, NULL);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(CreateDirIfMissingWrongPathNull),
        cmocka_unit_test(CreateDirIfMissingWrongPathFile),
        cmocka_unit_test(CreateDirIfMissingWrongPathCurrentDir),
        cmocka_unit_test(CreateDirIfMissingPermissionIssue),
        cmocka_unit_test(CreateDirIfMissingAlreadyExists),
        cmocka_unit_test(CreateDirIfMissingAlreadyExistsPermIssue),
        cmocka_unit_test(CreateDirIfMissingSuccessCreated),
        cmocka_unit_test(SetLibraryBaseInvalid),
        cmocka_unit_test(SetLibraryBaseNull),
        cmocka_unit_test(ExtractNewFile),
        cmocka_unit_test(ExtractFileExists),
        cmocka_unit_test(GetPath),
        cmocka_unit_test(GetPathNoFile),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
