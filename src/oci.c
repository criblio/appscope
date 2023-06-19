#define _GNU_SOURCE

#include "oci.h"
#include "cJSON.h"


/*
 * Read the OCI configuration into memory
 *
 * Returns the modified data which should be freed with scope_free, in case of failure returns NULL
 */
void *
ociReadCfgIntoMem(const char *cfgPath) {
    void *buf = NULL;
    struct stat fileStat;

    if (scope_stat(cfgPath, &fileStat) == -1) {
        return buf;
    }

    FILE *fp = scope_fopen(cfgPath, "r");
    if (!fp) {
        return buf;
    }

    buf = (char *)scope_malloc(fileStat.st_size);
    if (!buf) {
        goto close_file;
    }

    size_t ret = scope_fread(buf, sizeof(char), fileStat.st_size, fp);
    if (ret != fileStat.st_size ) {
        scope_free(buf);
        buf = NULL;
        goto close_file;
    }

close_file:

    scope_fclose(fp);

    return buf;
}

/*
 * Modify the OCI configuration for the given container.
 * A path to the container specific the location of the configuration file.
 *
 * Please look into opencontainers Linux runtime-spec for details about the exact JSON struct.
 * The following changes will be performed:
 * - Add a mount points
 *   * `appscope` directory will be mounted from the host "/usr/lib/appscope/" into the container: "/usr/lib/appscope/"
 *   * UNIX socket directory will be mounted from the host into the container the path to UNIX socket will be read from
 *   host based on value in the filter file
 *
 * - Extend Environment variables
 *   * `LD_PRELOAD` will contain the following entry `/opt/appscope/libscope.so`
 *   * `SCOPE_SETUP_DONE=true` mark that configuration was processed
 *
 * - Add prestart hook
 *   execute scope extract operation to ensure using library with proper loader reference (musl/glibc)
 * 
 * Returns the modified data which should be freed with scope_free, in case of failure returns NULL
 */
char *
ociModifyCfg(const void *cfgMem, const char *scopePath) {

    cJSON *json = cJSON_Parse(cfgMem);
    if (json == NULL) {
        goto exit;
    }

    /*
    * Handle process environment variables
    *
    "env":[
         "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
         "HOSTNAME=6735578591bb",
         "TERM=xterm",
         "LD_PRELOAD=/opt/appscope/libscope.so",
         "SCOPE_SETUP_DONE=true"
      ],
    */
    cJSON *procNode = cJSON_GetObjectItemCaseSensitive(json, "process");
    if (!procNode) {
        procNode = cJSON_CreateObject();
        if (!procNode) {
            cJSON_Delete(json);
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
                size_t newLdprelLen = itemLen + C_STRLEN("/opt/appscope/libscope.so:");
                char *newLdPreloadLib = scope_calloc(1, newLdprelLen);
                if (!newLdPreloadLib) {
                    cJSON_Delete(json);
                    goto exit;
                }
                scope_strncpy(newLdPreloadLib, "LD_PRELOAD=/opt/appscope/libscope.so:", C_STRLEN("LD_PRELOAD=/opt/appscope/libscope.so:"));
                scope_strcat(newLdPreloadLib, strItem + C_STRLEN("LD_PRELOAD="));
                cJSON *newLdPreloadLibObj = cJSON_CreateString(newLdPreloadLib);
                if (!newLdPreloadLibObj) {
                    scope_free(newLdPreloadLib);
                    cJSON_Delete(json);
                    goto exit;
                }
                cJSON_ReplaceItemInArray(envNodeArr, i, newLdPreloadLibObj);
                scope_free(newLdPreloadLib);

                cJSON *scopeEnvNode = cJSON_CreateString("SCOPE_SETUP_DONE=true");
                if (!scopeEnvNode) {
                    cJSON_Delete(json);
                    goto exit;
                }
                cJSON_AddItemToArray(envNodeArr, scopeEnvNode);
                ldPreloadPresent = TRUE;
                break;
            } else if (scope_strncmp("SCOPE_SETUP_DONE=true", strItem, C_STRLEN("SCOPE_SETUP_DONE=true")) == 0) {
                // we are done here
                cJSON_Delete(json);
                goto exit;
            }
        }


        // There was no LD_PRELOAD in environment variables
        if (ldPreloadPresent == FALSE) {
            const char *const envItems[2] =
            {
                "LD_PRELOAD=/opt/appscope/libscope.so",
                "SCOPE_SETUP_DONE=true"
            };
            for (int i = 0; i < ARRAY_SIZE(envItems) ;++i) {
                cJSON *scopeEnvNode = cJSON_CreateString(envItems[i]);
                if (!scopeEnvNode) {
                    cJSON_Delete(json);
                    goto exit;
                }
                cJSON_AddItemToArray(envNodeArr, scopeEnvNode);
            }
        }
    } else {
        const char * envItems[2] =
        {
            "LD_PRELOAD=/opt/appscope/libscope.so",
            "SCOPE_SETUP_DONE=true"
        };
        envNodeArr = cJSON_CreateStringArray(envItems, ARRAY_SIZE(envItems));
        if (!envNodeArr) {
            cJSON_Delete(json);
            goto exit;
        }
        cJSON_AddItemToObject(procNode, "env", envNodeArr);
    }

    /*
    * Handle process mounts for library and filter file and socket
    *
    "mounts":[
      {
         "destination":"/proc",
         "type":"proc",
         "source":"proc",
         "options":[
            "nosuid",
            "noexec",
            "nodev"
         ]
      },
      ...
      {
         "destination":"/usr/lib/appscope/",
         "type":"bind",
         "source":"/usr/lib/appscope/",
         "options":[
            "rbind",
            "rprivate"
         ]
      },
      {
         "destination":"/var/run/appscope/",
         "type":"bind",
         "source":"/var/run/appscope/",
         "options":[
            "rbind",
            "rprivate"
         ]
      }
    */

    const char *mountPath[2] =
    {
        "/usr/lib/appscope/",
        "/var/run/appscope/"
    };

    for (int i = 0; i < ARRAY_SIZE(mountPath); ++i ) {
        cJSON *mountNodeArr = cJSON_GetObjectItemCaseSensitive(json, "mounts");
        if (!mountNodeArr) {
            mountNodeArr = cJSON_CreateArray();
            if (!mountNodeArr) {
                cJSON_Delete(json);
                goto exit;
            }
            cJSON_AddItemToObject(json, "mounts", mountNodeArr);
        }

        cJSON *mountNode = cJSON_CreateObject();
        if (!mountNode) {
            cJSON_Delete(json);
            goto exit;
        }

        if (!cJSON_AddStringToObjLN(mountNode, "destination", mountPath[i])) {
            cJSON_Delete(mountNode);
            cJSON_Delete(json);
            goto exit;
        }

        if (!cJSON_AddStringToObjLN(mountNode, "type", "bind")) {
            cJSON_Delete(mountNode);
            cJSON_Delete(json);
            goto exit;
        }

        if (!cJSON_AddStringToObjLN(mountNode, "source", mountPath[i])) {
            cJSON_Delete(mountNode);
            cJSON_Delete(json);
            goto exit;
        }

        const char *optItems[2] =
        {
            "rbind",
            "rprivate"
        };

        cJSON *optNodeArr = cJSON_CreateStringArray(optItems, ARRAY_SIZE(optItems));
        if (!optNodeArr) {
            cJSON_Delete(mountNode);
            cJSON_Delete(json);
            goto exit;
        }
        cJSON_AddItemToObject(mountNode, "options", optNodeArr);
        cJSON_AddItemToArray(mountNodeArr, mountNode);
    }

    /*
    * Handle startContainer hooks process
    *
   "hooks":{
      "prestart":[
         {
            "path":"/proc/1513/exe",
            "args":[
               "libnetwork-setkey",
               "-exec-root=/var/run/docker",
               "6735578591bb3c5aebc91e5c702470c52d2c10cea52e4836604bf5a4a6c0f2eb",
               "ec7e49ffc98c"
            ]
         }
      ],
      "startContainer":[
         {
            "path":"/usr/lib/appscope/<version>/scope"
            "args":[
               "/usr/lib/appscope/<version>/scope",
               "extract",
               "-p",
               "/opt/appscope",
            ]
         },
       ]
    */
    cJSON *hooksNode = cJSON_GetObjectItemCaseSensitive(json, "hooks");
    if (!hooksNode) {
        hooksNode = cJSON_CreateObject();
        if (!hooksNode) {
            cJSON_Delete(json);
            goto exit;
        }
        cJSON_AddItemToObject(json, "hooks", hooksNode);
    }

    cJSON *startContainerNodeArr = cJSON_GetObjectItemCaseSensitive(hooksNode, "startContainer");
    if (!startContainerNodeArr) {
        startContainerNodeArr = cJSON_CreateArray();
        if (!startContainerNodeArr) {
            cJSON_Delete(json);
            goto exit;
        }
        cJSON_AddItemToObject(hooksNode, "startContainer", startContainerNodeArr);
    }

    cJSON *startContainerNode = cJSON_CreateObject();
    if (!startContainerNode) {
        cJSON_Delete(json);
        goto exit;
    }

    if (!cJSON_AddStringToObjLN(startContainerNode, "path",  scopePath)) {
        cJSON_Delete(startContainerNode);
        cJSON_Delete(json);
        goto exit;
    }

    const char *argsItems[4] =
    {
        scopePath,
        "extract",
        "-p",
        "/opt/appscope"
    };
    cJSON *argsNodeArr = cJSON_CreateStringArray(argsItems, ARRAY_SIZE(argsItems));
    if (!argsNodeArr) {
        cJSON_Delete(startContainerNode);
        cJSON_Delete(json);
        goto exit;
    }
    cJSON_AddItemToObject(startContainerNode, "args", argsNodeArr);
    cJSON_AddItemToArray(startContainerNodeArr, startContainerNode);

    char *jsonStr = cJSON_PrintUnformatted(json);
    cJSON_Delete(json);

    return jsonStr;

exit:
    return NULL;
}

/*
 * Write the OCI configuration into the specified file
 * 
 * Returns TRUE in case of success, FALSE otherwise
 */
bool
ociWriteConfig(const char *path, const char *cfg) {
    FILE *fp = scope_fopen(path, "w");
    if (fp == NULL) {
        return FALSE;
    }

    scope_fprintf(fp, "%s\n", cfg);

    scope_fclose(fp);
    return TRUE;
}

