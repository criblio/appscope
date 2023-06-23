#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oci.h"
#include "cJSON.h"
#include "test.h"
#include "scopestdlib.h"

static char dirPath[PATH_MAX];

static const char *testTypeJson[] = {
    "EmptyJson",
    "ProcessOnlyJson",
    "ProcessEnvPresentLDPreload",
    "ProcessEnvEmptyLDPreload",
    "MissingHooks",
    "IncompleteHooks",
    "CompleteHooks"
};

static int
testDirPath(char *path, const char *argv0) {
    char buf[PATH_MAX];
    if (argv0[0] == '/') {
        scope_strcpy(buf, argv0);
    } else {
        if (scope_getcwd(buf, PATH_MAX) == NULL) {
            scope_perror("getcwd error");
            return -1;
        }
        scope_strcat(buf, "/");
        scope_strcat(buf, argv0);
    }

    if (scope_realpath(buf, path) == NULL) {
        scope_perror("scope_realpath error");
        return -1;
    }

    /*
    * Retrieve the test directory path.
    * From:
    * /<dir>/appscope/test/linux/cfgutilsrulestest
    * To:
    * /<dir>/appscope/test/
    */
    for (int i= 0; i < 2; ++i) {
        path = scope_dirname(path);
        if (path == NULL) {
            scope_perror("scope_dirname error");
            return -1;
        }
    }
    return 0;
}

static bool
verifyModifiedCfg(int id, const char *cmpStr, const char *outPath) {
    int fdOut = scope_open(outPath, O_RDONLY);
    if (fdOut == -1) {
        assert_non_null(NULL);
        return FALSE;
    }

    struct stat stOut;
    if (scope_fstat(fdOut, &stOut) == -1) {
        assert_non_null(NULL);
        scope_close(fdOut);
        return FALSE;
    }

    void *fdOutMap = scope_mmap(NULL, stOut.st_size, PROT_READ, MAP_PRIVATE, fdOut, 0);
    if (fdOutMap == MAP_FAILED) {
        assert_non_null(NULL);
        scope_close(fdOut);
        return FALSE;
    }
    scope_close(fdOut);

    cJSON* parseOut = cJSON_Parse(fdOutMap);
    char* outBuf = cJSON_PrintUnformatted(parseOut);
    cJSON_Delete(parseOut);

    // assert_int_equal(scope_strlen(outBuf), scope_strlen(cmpStr));
    if (scope_strcmp(cmpStr, outBuf) != 0 ){
        scope_fprintf(scope_stderr, cmpStr);
        scope_fprintf(scope_stderr, outBuf);
        assert_non_null(NULL);
        scope_munmap(fdOutMap, stOut.st_size);
        return FALSE;
    }

    scope_free(outBuf);
    scope_munmap(fdOutMap, stOut.st_size);

    return TRUE;
}

static bool
rewriteOpenContainersConfigTest(int id, const char* unixSocketPath) {

    char inPath [PATH_MAX] = {0};
    scope_snprintf(inPath, PATH_MAX, "%s/data/oci/oci%din.json", dirPath, id);
    const char *scopeWithVersion = "/usr/lib/appscope/1.2.3/scope";

    void *cfgMem = ociReadCfgIntoMem(inPath);

    char *modifMem = ociModifyCfg(cfgMem, scopeWithVersion, unixSocketPath);

    char outPath [PATH_MAX] = {0};

    if (unixSocketPath) {
        scope_snprintf(outPath, PATH_MAX, "%s/data/oci/oci%doutfull.json", dirPath, id);
    } else {
        scope_snprintf(outPath, PATH_MAX, "%s/data/oci/oci%doutpartial.json", dirPath, id);
    }

    bool res = verifyModifiedCfg(id, modifMem, outPath);

    scope_free(cfgMem);
    scope_free(modifMem);

    return res;
}

static void
ocitest_with_unix_path(void **state) {
    for (int i = 0; i < ARRAY_SIZE(testTypeJson); ++i) {
        bool res = rewriteOpenContainersConfigTest(i, "/var/run/appscope/appscope.sock");
        if (res != TRUE) {
            scope_fprintf(scope_stderr, "Error with test: id=%d name=%s\n", i, testTypeJson[i]);
        }
        assert_int_equal(res, TRUE);
    }
}

static void
ocitest_without_unix_path(void **state) {
    for (int i = 0; i < ARRAY_SIZE(testTypeJson); ++i) {
        bool res = rewriteOpenContainersConfigTest(i, NULL);
        if (res != TRUE) {
            scope_fprintf(scope_stderr, "Error with test: id=%d name=%s\n", i, testTypeJson[i]);
        }
        assert_int_equal(res, TRUE);
    }
}

int
main(int argc, char* argv[])
{
    scope_printf("running %s\n", argv[0]);
    if (testDirPath(dirPath, argv[0])) {
        return EXIT_FAILURE;
    }

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ocitest_with_unix_path),
        cmocka_unit_test(ocitest_without_unix_path),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
