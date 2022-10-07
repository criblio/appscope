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

int         libdirSetBase(file_t file, const char *base);    // Override default base dir i.e. /tmp
int         libdirExtract(file_t file);                      // Extracts file to default path
const char* libdirGetPath(file_t file);                      // Get full path to existing file

#endif // _SCOPE_LIBDIR_H
