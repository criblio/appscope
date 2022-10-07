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
