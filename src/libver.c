#define _GNU_SOURCE

#include <errno.h>

#include "libver.h"
#include "scopestdlib.h"
#include "scopetypes.h"

/*
 * Returns normalized version string
 * - "dev" for unofficial release
 * - "%d.%d.%d" for official release (e.g. "1.3.0" for "v1.3.0")
 */
const char *
libverNormalizedVersion(const char *version) {

    if ((version == NULL) || (*version != 'v')) {
        return "dev";
    }
    ++version;
    size_t versionSize = scope_strlen(version);

    for (int i = 0; i < versionSize; ++i) {
        // Only digit and "." are accepted
        if ((scope_isdigit(version[i]) == 0) && version[i] != '.' ) {
            return "dev"; 
        }
        if (i == 0 || i == versionSize) {
            // First and last character must be number
            if (scope_isdigit(version[i]) == 0) {
                return "dev";
            }
        }
    }
    return version;
}
