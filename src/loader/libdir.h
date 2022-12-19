/**
 * Cribl AppScope - Library Directory Interface
 *
 * See docs/STARTUP.md
 */

#ifndef _SCOPE_LIBDIR_H
#define _SCOPE_LIBDIR_H 1

#include <linux/limits.h>
#include <sys/stat.h>
#include <stdbool.h>
#include "scopetypes.h"

// File types
typedef enum {
    LIBRARY_FILE, // libscope.so
} libdirfile_t;

typedef enum {
    MKDIR_STATUS_CREATED = 0,           // Path was created
    MKDIR_STATUS_EXISTS = 1,            // Path already points to existing directory
    MKDIR_STATUS_ERR_PERM_ISSUE = 2,    // Error: Path already points to existing directory but user can not create file there
    MKDIR_STATUS_ERR_NOT_ABS_DIR = 3,   // Error: Path does not points to a directory
    MKDIR_STATUS_ERR_OTHER = 4,         // Error: Other
} mkdir_status_t;

mkdir_status_t libdirCreateDirIfMissing(const char *, mode_t, uid_t, gid_t);
int libdirSetLibraryBase(const char *);                                      // Override default library base search dir i.e. /tmp
int libdirExtract(libdirfile_t, uid_t, gid_t);                               // Extracts file to default path
const char *libdirGetPath(libdirfile_t);                                     // Get full path to existing file
int libdirSaveLibraryFile(const char *, bool, mode_t, uid_t, gid_t);         // Save libscope.so to specified path overwrite

// Unit Test helper
int libdirInitTest(const char *, const char *, const char *); // Override defaults

#endif // _SCOPE_LIBDIR_H
