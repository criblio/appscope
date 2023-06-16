#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include "test.h"
#include "scopestdlib.h"

#ifndef FALSE
#define FALSE   0
#endif
#ifndef TRUE
#define TRUE   1
#endif

static char dirPath[PATH_MAX];

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
    * /<dir>/appscope/test/linux/cfgutilsfiltertest
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

// This function reflects the logic provided by rewriteOpenContainersConfig
static bool
rewriteOpenContainersConfigTest(int id) {
    char inPath [PATH_MAX] = {0};
    char outPath [PATH_MAX] = {0};
    scope_snprintf(inPath, PATH_MAX, "%s/data/oci/oci%din.json", dirPath, id);
    scope_snprintf(outPath, PATH_MAX, "%s/data/oci/oci%dout.json", dirPath, id);

    bool res = FALSE;
    struct stat fileStat;
    if (scope_stat(inPath, &fileStat) == -1) {
        assert_non_null(NULL);
        return res;
    }

    FILE *fp = scope_fopen(inPath, "r");
    if (!fp) {
        assert_non_null(NULL);
        return res;
    }

    /*
    * Read the file contents into a string
    */
    char *buf = (char *)scope_malloc(fileStat.st_size);
    if (!buf) {
        scope_fclose(fp);
        assert_non_null(NULL);
        return res;
    }

    scope_fread(buf, sizeof(char), fileStat.st_size, fp);
    scope_fclose(fp);


    cJSON *json = cJSON_Parse(buf);
    if (!json) {
        assert_non_null(NULL);
        return res;
    }
    scope_free(buf);
    char *jsonStr = NULL;

    // Handle the process
    cJSON *procNode = cJSON_GetObjectItemCaseSensitive(json, "process");
    if (!procNode) {
        procNode = cJSON_CreateObject();
        if (!procNode) {
            assert_non_null(NULL);
            goto exit;
        }
        cJSON_AddItemToObject(json, "process", procNode);
    }
    cJSON *envNodeArr = cJSON_GetObjectItemCaseSensitive(procNode, "env");
    if (envNodeArr) {
        bool ldPreloadPresent = FALSE;
        // Iterate over environment string array
        size_t envSize = cJSON_GetArraySize(envNodeArr);
        for (int i = 0; i < envSize ;++i) {
            cJSON *item = cJSON_GetArrayItem(envNodeArr, i);
            char *strItem = cJSON_GetStringValue(item);

            if (scope_strncmp("LD_PRELOAD=", strItem, C_STRLEN("LD_PRELOAD=")) == 0) {
                size_t itemLen = scope_strlen(strItem);
                size_t newLdprelLen = itemLen + C_STRLEN("/opt/libscope.so:");
                char *newLdPreloadLib = scope_calloc(1, newLdprelLen);
                if (!newLdPreloadLib) {
                    assert_non_null(NULL);
                    goto exit;
                }
                scope_strncpy(newLdPreloadLib, "LD_PRELOAD=/opt/libscope.so:", C_STRLEN("LD_PRELOAD=/opt/libscope.so:"));
                scope_strcat(newLdPreloadLib, strItem + C_STRLEN("LD_PRELOAD="));
                cJSON *newLdPreloadLibObj = cJSON_CreateString(newLdPreloadLib);
                if (!newLdPreloadLibObj) {
                    scope_free(newLdPreloadLib);
                    assert_non_null(NULL);
                    goto exit;
                }
                cJSON_ReplaceItemInArray(envNodeArr, i, newLdPreloadLibObj);
                scope_free(newLdPreloadLib);

                cJSON *scopeEnvNode = cJSON_CreateString("SCOPE_SETUP_DONE=true");
                if (!scopeEnvNode) {
                    assert_non_null(NULL);
                    goto exit;
                }
                cJSON_AddItemToArray(envNodeArr, scopeEnvNode);
                ldPreloadPresent = TRUE;
                break;
            } else if (scope_strncmp("SCOPE_SETUP_DONE=true", strItem, C_STRLEN("SCOPE_SETUP_DONE=true")) == 0) {
                // we are done here
                res = TRUE;
                goto exit;
            }
        }


        // There was no LD_PRELOAD in environment variables
        if (ldPreloadPresent == FALSE) {
            const char *const envItems[2] =
            {
                "LD_PRELOAD=/opt/libscope.so",
                "SCOPE_SETUP_DONE=true"
            };
            for (int i = 0; i < 2 ;++i) {
                cJSON *scopeEnvNode = cJSON_CreateString(envItems[i]);
                if (!scopeEnvNode) {
                    assert_non_null(NULL);
                    goto exit;
                }
                cJSON_AddItemToArray(envNodeArr, scopeEnvNode);
            }
        }
    } else {
        const char * envItems[2] =
        {
            "LD_PRELOAD=/opt/libscope.so",
            "SCOPE_SETUP_DONE=true"
        };
        envNodeArr = cJSON_CreateStringArray(envItems, 2);
        if (!envNodeArr) {
            assert_non_null(NULL);
            goto exit;
        }
        cJSON_AddItemToObject(procNode, "env", envNodeArr);
    }

    cJSON *mountNodeArr = cJSON_GetObjectItemCaseSensitive(json, "mounts");
    if (!mountNodeArr) {
        mountNodeArr = cJSON_CreateArray();
        if (!mountNodeArr) {
            assert_non_null(NULL);
            goto exit;
        }
        cJSON_AddItemToObject(json, "mounts", mountNodeArr);
    }

    cJSON *mountNode = cJSON_CreateObject();
    if (!mountNode) {
        assert_non_null(NULL);
        goto exit;
    }

    if (!cJSON_AddStringToObjLN(mountNode, "destination", "/opt/scope")) {
        cJSON_Delete(mountNode);
        assert_non_null(NULL);
        goto exit;
    }

    if (!cJSON_AddStringToObjLN(mountNode, "type", "bind")) {
        cJSON_Delete(mountNode);
        assert_non_null(NULL);
        goto exit;
    }

    if (!cJSON_AddStringToObjLN(mountNode, "source", "/tmp/appscope/dev/scope")) {
        cJSON_Delete(mountNode);
        assert_non_null(NULL);
        goto exit;
    }

    const char *optItems[2] =
    {
        "rbind",
        "rprivate"
    };

    cJSON *optNodeArr = cJSON_CreateStringArray(optItems, 2);
    if (!optNodeArr) {
        cJSON_Delete(mountNode);
        assert_non_null(NULL);
        goto exit;
    }
    cJSON_AddItemToObject(mountNode, "options", optNodeArr);
    cJSON_AddItemToArray(mountNodeArr, mountNode);

    cJSON *hooksNode = cJSON_GetObjectItemCaseSensitive(json, "hooks");
    if (!hooksNode) {
        hooksNode = cJSON_CreateObject();
        if (!hooksNode) {
            assert_non_null(NULL);
            goto exit;
        }
        cJSON_AddItemToObject(json, "hooks", hooksNode);
    }

    cJSON *startContainerNodeArr = cJSON_GetObjectItemCaseSensitive(hooksNode, "startContainer");
    if (!startContainerNodeArr) {
        startContainerNodeArr = cJSON_CreateArray();
        if (!startContainerNodeArr) {
            assert_non_null(NULL);
            goto exit;
        }
        cJSON_AddItemToObject(hooksNode, "startContainer", startContainerNodeArr);
    }

    cJSON *startContainerNode = cJSON_CreateObject();
    if (!startContainerNode) {
        assert_non_null(NULL);
        goto exit;
    }

    if (!cJSON_AddStringToObjLN(startContainerNode, "path",  "/opt/scope")) {
        cJSON_Delete(startContainerNode);
        assert_non_null(NULL);
        goto exit;
    }

    const char *argsItems[3] =
    {
        "/opt/scope",
        "extract",
        "/opt"
    };
    cJSON *argsNodeArr = cJSON_CreateStringArray(argsItems, 3);
    if (!argsNodeArr) {
        cJSON_Delete(startContainerNode);
        assert_non_null(NULL);
        goto exit;
    }
    cJSON_AddItemToObject(startContainerNode, "args", argsNodeArr);
    cJSON_AddItemToArray(startContainerNodeArr, startContainerNode);

    jsonStr = cJSON_Print(json);

    // Overwrite the file
    int fdOut = scope_open(outPath, O_RDONLY);
    if (fdOut == -1) {
        cJSON_free(jsonStr);
        goto exit;
    }

    struct stat stOut;
    if (scope_fstat(fdOut, &stOut) == -1) {
        cJSON_free(jsonStr);
        scope_close(fdOut);
        goto exit;
    }


    void *fdOutMap = scope_mmap(NULL, stOut.st_size, PROT_READ, MAP_PRIVATE, fdOut, 0);
    if (fdOutMap == MAP_FAILED) {
        cJSON_free(jsonStr);
        scope_close(fdOut);
        goto exit;
    }

    if (memcmp(fdOutMap, jsonStr, stOut.st_size) != 0) {
        assert_non_null(NULL);
    }
    scope_munmap(fdOutMap, stOut.st_size);

    cJSON_free(jsonStr);
    scope_close(fdOut);

    res = TRUE;
exit:
    cJSON_Delete(json);
    return res;
}

static void
ocitest_empty_json(void **state)
{
    bool res = rewriteOpenContainersConfigTest(0);
    assert_int_equal(res, TRUE);
}

static void
ocitest_process_only_json(void **state)
{
    bool res = rewriteOpenContainersConfigTest(1);
    assert_int_equal(res, TRUE);
}

static void
ocitest_process_env_present_preload(void **state)
{
    bool res = rewriteOpenContainersConfigTest(2);
    assert_int_equal(res, TRUE);
}

static void
ocitest_process_env_empty_preload(void **state)
{
    bool res = rewriteOpenContainersConfigTest(3);
    assert_int_equal(res, TRUE);
}

static void
ocitest_missing_hooks(void **state)
{
    bool res = rewriteOpenContainersConfigTest(4);
    assert_int_equal(res, TRUE);
}

static void
ocitest_hooks_incomplete(void **state)
{
    bool res = rewriteOpenContainersConfigTest(5);
    assert_int_equal(res, TRUE);
}

static void
ocitest_hooks_complete(void **state)
{
    bool res = rewriteOpenContainersConfigTest(6);
    assert_int_equal(res, TRUE);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    if (testDirPath(dirPath, argv[0])) {
        return EXIT_FAILURE;
    }

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(ocitest_empty_json),
        cmocka_unit_test(ocitest_process_only_json),
        cmocka_unit_test(ocitest_process_env_present_preload),
        cmocka_unit_test(ocitest_process_env_empty_preload),
        cmocka_unit_test(ocitest_missing_hooks),
        cmocka_unit_test(ocitest_hooks_incomplete),
        cmocka_unit_test(ocitest_hooks_complete),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
