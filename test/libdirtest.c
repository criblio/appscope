#define _GNU_SOURCE
#define _XOPEN_SOURCE 500
#include <ftw.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libdir.h"
#include "libver.h"
#include "test.h"
#include "scopestdlib.h"

#define TEST_BASE_DIR "/tmp/appscope-test/"
#define TEST_INSTALL_BASE "/tmp/appscope-test/install"
#define TEST_INSTALL_BASE_NOT_REACHABLE "/root/base/appscope"
#define TEST_TMP_BASE "/tmp/appscope-test/tmp"
#define TEST_TMP_BASE_NOT_REACHALBE "/root/tmp/appscope"

static int
rm_callback(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    return (remove(fpath) < 0) ? -1 : 0;
}

static int
rm_recursive(char *path) {
    return nftw(path, rm_callback, 64, FTW_DEPTH | FTW_PHYS);
}

static int
teardownlibdirTest(void **state) {
     rm_recursive(TEST_BASE_DIR);
     return 0;
}

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
    res = rm_recursive(buf);
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
    res = rm_recursive(buf);
    assert_int_equal(res, 0);
}

static void
SetLibraryBaseInvalid(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "dev");
    int res = libdirSetLibraryBase("/does_not_exist");
    assert_int_equal(res, -1);
}

static void
SetLibrarySuccessDev(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "dev");
    // Create dummy file
    mkdir_status_t mkres = libdirCreateDirIfMissing("/tmp/appscope-test/success/dev");
    assert_in_range(mkres, MKDIR_STATUS_CREATED, MKDIR_STATUS_EXISTS);
    FILE *fp = scope_fopen("/tmp/appscope-test/success/dev/libscope.so", "w");
    assert_non_null(fp);
    scope_fclose(fp);
    int res = libdirSetLibraryBase("/tmp/appscope-test/success");
    assert_int_equal(res, 0);
}

static void
SetLibraryBaseNull(void **state) {
    libdirInitTest(NULL, NULL, NULL);
    int res = libdirSetLibraryBase("");
    assert_int_equal(res, -1);
}

static void
ExtractNewFileDev(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "dev");
    const char *normVer = libverNormalizedVersion("dev");
    char expected_location[PATH_MAX] = {0};
    struct stat dirStat = {0};

    // TEST_TMP_BASE will be used
    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    scope_memset(expected_location, 0, PATH_MAX);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
}

static void
ExtractNewFileDevAlternative(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE_NOT_REACHALBE, "dev");
    const char *normVer = libverNormalizedVersion("dev");
    char expected_location[PATH_MAX] = {0};
    struct stat dirStat = {0};

    // Extract will fail because second path is not accessbile
    int res = libdirExtract(LIBRARY_FILE);
    assert_int_not_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_not_equal(res, 0);
    scope_memset(expected_location, 0, PATH_MAX);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_not_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_not_equal(res, 0);
}

static void
ExtractNewFileOfficial(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "v1.1.0");
    const char *normVer = libverNormalizedVersion("v1.1.0");
    char expected_location[PATH_MAX] = {0};
    struct stat dirStat = {0};

    // TEST_INSTALL_BASE will be used
    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
    scope_memset(expected_location, 0, PATH_MAX);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
}

static void
ExtractNewFileOfficialAlternative(void **state) {
    libdirInitTest(TEST_INSTALL_BASE_NOT_REACHABLE, TEST_TMP_BASE, "v1.1.0");
    const char *normVer = libverNormalizedVersion("v1.1.0");
    char expected_location[PATH_MAX] = {0};
    struct stat dirStat = {0};

    // TEST_TMP_BASE will be used
    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
    scope_memset(expected_location, 0, PATH_MAX);
    
    res = libdirExtract(LOADER_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "ldscopedyn");
    res = scope_stat(expected_location, &dirStat);
    assert_int_equal(res, 0);
}

static void
ExtractFileExistsOfficial(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "v1.1.0");
    const char *normVer = libverNormalizedVersion("v1.1.0");
    char expected_location[PATH_MAX] = {0};
    struct stat firstStat = {0};
    struct stat secondStat = {0};

    int res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &firstStat);
    assert_int_equal(res, 0);

    res = libdirExtract(LIBRARY_FILE);
    assert_int_equal(res, 0);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "libscope.so");
    res = scope_stat(expected_location, &secondStat);
    assert_int_equal(res, 0);
    assert_int_equal(firstStat.st_ctim.tv_sec, secondStat.st_ctim.tv_sec);
    assert_int_equal(firstStat.st_ctim.tv_nsec, secondStat.st_ctim.tv_nsec);
}

static void
GetPathDev(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "dev");
    const char *normVer = libverNormalizedVersion("dev");
    char expected_location[PATH_MAX] = {0};
    // Create dummy directory
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/", TEST_TMP_BASE, normVer);
    mkdir_status_t mkres = libdirCreateDirIfMissing(expected_location);
    assert_in_range(mkres, MKDIR_STATUS_CREATED, MKDIR_STATUS_EXISTS);
    // Create dummy file
    scope_memset(expected_location, 0, PATH_MAX);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "libscope.so");
    FILE *fp = scope_fopen(expected_location, "w");
    assert_non_null(fp);
    scope_fclose(fp);

    const char *library_path = libdirGetPath(LIBRARY_FILE);
    int res = scope_strcmp(expected_location, library_path);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
    // Create dummy file
    scope_memset(expected_location, 0, PATH_MAX);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_TMP_BASE, normVer, "ldscopedyn");
    fp = scope_fopen(expected_location, "w");
    assert_non_null(fp);
    scope_fclose(fp);

    const char *loader_path = libdirGetPath(LOADER_FILE);
    res = scope_strcmp(expected_location, loader_path);
    assert_int_equal(res, 0);
}

static void
GetPathOfficial(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "v1.1.0");
    const char *normVer = libverNormalizedVersion("v1.1.0");
    char expected_location[PATH_MAX] = {0};
    // Create dummy directory
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/", TEST_INSTALL_BASE, normVer);
    mkdir_status_t mkres = libdirCreateDirIfMissing(expected_location);
    assert_in_range(mkres, MKDIR_STATUS_CREATED, MKDIR_STATUS_EXISTS);
    // Create dummy file
    scope_memset(expected_location, 0, PATH_MAX);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "libscope.so");
    FILE *fp = scope_fopen(expected_location, "w");
    assert_non_null(fp);
    scope_fclose(fp);

    const char *library_path = libdirGetPath(LIBRARY_FILE);
    int res = scope_strcmp(expected_location, library_path);
    assert_int_equal(res, 0);
    scope_remove(expected_location);
    // Create dummy file
    scope_memset(expected_location, 0, PATH_MAX);
    scope_snprintf(expected_location, PATH_MAX, "%s/%s/%s", TEST_INSTALL_BASE, normVer, "ldscopedyn");
    fp = scope_fopen(expected_location, "w");
    assert_non_null(fp);
    scope_fclose(fp);

    const char *loader_path = libdirGetPath(LOADER_FILE);
    res = scope_strcmp(expected_location, loader_path);
    assert_int_equal(res, 0);
}

static void
GetPathNoFile(void **state) {
    libdirInitTest(TEST_INSTALL_BASE, TEST_TMP_BASE, "v1.1.0");

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
        cmocka_unit_test_teardown(SetLibraryBaseInvalid, teardownlibdirTest),
        cmocka_unit_test_teardown(SetLibraryBaseNull, teardownlibdirTest),
        cmocka_unit_test_teardown(SetLibrarySuccessDev, teardownlibdirTest),
        cmocka_unit_test_teardown(ExtractNewFileDev, teardownlibdirTest),
        cmocka_unit_test_teardown(ExtractNewFileDevAlternative, teardownlibdirTest),
        cmocka_unit_test_teardown(ExtractNewFileOfficial, teardownlibdirTest),
        cmocka_unit_test_teardown(ExtractNewFileOfficialAlternative, teardownlibdirTest),
        cmocka_unit_test_teardown(ExtractFileExistsOfficial, teardownlibdirTest),
        cmocka_unit_test_teardown(GetPathDev, teardownlibdirTest),  
        cmocka_unit_test_teardown(GetPathOfficial, teardownlibdirTest),
        cmocka_unit_test_teardown(GetPathNoFile, teardownlibdirTest),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
