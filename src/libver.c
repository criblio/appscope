#define _GNU_SOURCE

#include <errno.h>

#include "libver.h"
#include "scopestdlib.h"

/*
 * Returns normalized version string
 * - "dev" for unofficial release
 * - "%d.%d.%d" for official release (e.g. "1.3.0" for "v1.3.0")
 */
const char *
libverNormalizedVersion(const char* version) {
    // Treat the version which not begins with v as an unofficial
    // TODO: be more restricted here
    if ((version == NULL) || (*version != 'v')) {
        return "dev";
    }

    ++version;
    return version;
}

 /*
 * Verify if following absolute path points to directory
 * Returns operation status
 */
static mkdir_status_t
checkIfDirExists(const char* absDirPath) {
    struct stat st = {0};
    if (!scope_stat(absDirPath, &st)) {
        if (S_ISDIR(st.st_mode)) {
            return MKDIR_STATUS_EXISTS;
        }
        return MKDIR_STATUS_NOT_ABSOLUTE_DIR;
    }
    // stat fails
    return MKDIR_STATUS_OTHER_ISSUE;
}

 /*
 * Create a directory in following absolute path creating any intermediate directories as necessary
 * Returns operation status
 */
mkdir_status_t
libverMkdirNested(const char* absDirPath) {
    int mkdirRes = -1;
    /* Operate only on absolute path */
    if (absDirPath == NULL || *absDirPath != '/') {
        return MKDIR_STATUS_NOT_ABSOLUTE_DIR;
    }

    mkdir_status_t res = checkIfDirExists(absDirPath);

    /* exit if path exists */
    if (res != MKDIR_STATUS_OTHER_ISSUE) {
        return res;
    }

    char* tempPath = scope_strdup(absDirPath);
    if (tempPath == NULL) {
        goto end;
    }

    /* traverse the full path */
    for (char* p = tempPath + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';
            scope_errno = 0;
            mkdirRes = scope_mkdir(tempPath, 0755);
            /* scope_mkdir fails with error other than directory exists */
            if (mkdirRes && (scope_errno != EEXIST)) {
                 goto end;
            }
            *p = '/';
        }
    }

    /* last element */
    scope_errno = 0;
    mkdirRes = scope_mkdir(tempPath, 0755);
    if (mkdirRes && (scope_errno != EEXIST)) {
        goto end;
    }

    res = MKDIR_STATUS_CREATED;

end:
    scope_free(tempPath);
    return res;
}
