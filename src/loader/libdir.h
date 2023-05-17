#ifndef _SCOPE_LIBDIR_H
#define _SCOPE_LIBDIR_H 1

#include <linux/limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>

extern unsigned long g_libscopesz;
extern unsigned long g_scopedynsz;

// File types
typedef enum {
    LIBRARY_FILE, // libscope.so
    LOADER_FILE,  // scopedyn
} libdirfile_t;

typedef enum {
    MKDIR_STATUS_CREATED = 0,           // Path was created
    MKDIR_STATUS_EXISTS = 1,            // Path already points to existing directory
    MKDIR_STATUS_ERR_PERM_ISSUE = 2,    // Error: Path already points to existing directory but user can not create file there
    MKDIR_STATUS_ERR_NOT_ABS_DIR = 3,   // Error: Path does not points to a directory
    MKDIR_STATUS_ERR_OTHER = 4,         // Error: Other
} mkdir_status_t;

mkdir_status_t libdirCreateDirIfMissing(const char *, mode_t, uid_t, gid_t);
int libdirCreateFileIfMissing(unsigned char *, size_t, const char *, bool, mode_t, uid_t, gid_t);
int libdirSetLibraryBase(const char *);                                      // Override default library base search dir
int libdirExtract(unsigned char *, size_t, uid_t, gid_t);                    // Extracts libscope.so to default path
const char *libdirGetPath(void);                                             // Get full path to existing libscope.so
size_t getAsset(libdirfile_t, unsigned char **);

// Unit Test helper
int libdirInitTest(const char *, const char *, const char *); // Override defaults

#endif // _SCOPE_LIBDIR_H
