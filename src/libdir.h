/**
 * Cribl AppScope - Library Directory Interface
 *
 * See docs/STARTUP.md
 */

#ifndef _SCOPE_LIBDIR_H
#define _SCOPE_LIBDIR_H 1

// File types
typedef enum {
    LIBRARY_FILE, // libscope.so
    LOADER_FILE   // ldscopedyn
} file_t;

typedef enum {
    MKDIR_STATUS_CREATED = 0,           // Path was created
    MKDIR_STATUS_EXISTS = 1,            // Path already points to existing directory
    MKDIR_STATUS_ERR_PERM_ISSUE = 2,    // Error: Path already points to existing directory but user can not create file there
    MKDIR_STATUS_ERR_NOT_ABS_DIR = 3,   // Error: Path does not points to a directory
    MKDIR_STATUS_ERR_OTHER = 4,         // Error: Other
} mkdir_status_t;

mkdir_status_t libdirCreateDirIfMissing(const char *dir);
int libdirSetBase(file_t file, const char *base);         // Override default base dir i.e. /tmp
int libdirExtract(file_t file);                           // Extracts file to default path
const char *libdirGetPath(file_t file);                   // Get full path to existing file

#endif // _SCOPE_LIBDIR_H
