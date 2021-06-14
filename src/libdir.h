/**
 * Cribl AppScope - Library Directory Interface
 *
 * See docs/STARTUP.md
 */

#ifndef _SCOPE_LIBDIR_H
#define _SCOPE_LIBDIR_H 1

// The base directory is where we will create our Library Directory.
int         libdirSetBase(const char *basedir); // override default
const char* libdirGetBase();

// Our Library Directory is where we stash stuff
const char* libdirGetDir(); // directory name; i.e. "libscope-1.2.3"
const char* libdirGet();    // full path; i.e. "/tmp/libscope-1.2.3"
int         libdirClean();  // remove it and it's contents

// Put things in the Library Directory
int libdirExtractLoader();                              // $libdir/ldscopedyn
int libdirExtractLibrary();                             // $libdir/libscope.so
int libdirLinkLoader(const char *from, const char *to); // $libdir/from -> to

// Get paths of things in the Library Directory
const char* libdirGetLoader();  // extracts & returns "$libdir/ldscopedyn"
const char* libdirGetLibrary(); // extracts & returns "$libdir/libscope.so"

#endif // _SCOPE_LIBDIR_H
