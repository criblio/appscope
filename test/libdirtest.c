#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libdir.h"
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
SetLibraryBaseValid(void **state) {
}

static void
SetLibraryBaseNull(void **state) {
}

static void
ExtractNewFile(void **state) {
}

static void
ExtractFileExists(void **state) {
}

static void
ExtractFileNoPerms(void **state) {
}

static void
GetPathInstallPath(void **state) {
}

static void
GetPathTmpPath(void **state) {
}

static void
GetPathNoFile(void **state) {
}

static void
CreateFileIfMissingNewFile(void **state) {
}

static void
CreateFileIfMissingFileExists(void **state) {
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
        cmocka_unit_test(SetBase),
        cmocka_unit_test(ExtractNewFile),
        cmocka_unit_test(ExtractFileExists),
        cmocka_unit_test(GetPathInstallPath),
        cmocka_unit_test(GetPathTmpPath),
        cmocka_unit_test(GetPathNoFile),
        cmocka_unit_test(CheckIfDirExistsDirExists),
        cmocka_unit_test(CheckIfDirExistsNoDir),
        cmocka_unit_test(CreateFileIfMissingNewFile),
        cmocka_unit_test(CreateFileIfMissingFileExists),
        cmocka_unit_test(CheckNoteBadNote),
        cmocka_unit_test(CheckNoteGoodNote),
        cmocka_unit_test(GetNote),
        cmocka_unit_test(GetVer),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
