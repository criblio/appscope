/**
 * Cribl AppScope - Library Directory Implementation
 */

#include "libdir.h"

#define _XOPEN_SOURCE 500 // for FTW
#define _GNU_SOURCE

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <linux/limits.h> // for PATH_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "scopestdlib.h"
#include "scopetypes.h" // for ROUND_UP()

#ifndef SCOPE_VER
#error "Missing SCOPE_VER"
#endif

#ifndef SCOPE_LDSCOPEDYN
#define SCOPE_LDSCOPEDYN "ldscopedyn"
#endif

#ifndef SCOPE_LIBSCOPE_SO
#define SCOPE_LIBSCOPE_SO "libscope.so"
#endif

#ifndef SCOPE_TEMP_BASE
#define SCOPE_TEMP_BASE "/tmp/appscope"
#endif

#ifndef SCOPE_INSTALL_BASE
#define SCOPE_INSTALL_BASE "/usr/lib/appscope"
#endif

// private global state
static struct {
    char ver[PATH_MAX];      // name of the subdirectory under base i.e. 1.0.0
    char lib_base[PATH_MAX]; // full path to the library base directory i.e. /tmp/appscope or /usr/lib/appscope
    char lib_path[PATH_MAX]; // full path to the library file
    char lib_dir[PATH_MAX];  // full path to the library dir
    char ld_base[PATH_MAX];  // full path to the loader base directory i.e. /tmp/appscope or /usr/lib/appscope
    char ld_path[PATH_MAX];  // full path to the loader file
    char ld_dir[PATH_MAX];   // full path to the loader dir
} g_libdir_info;

// We use `objcopy` in the Makefile to get `ldscopedyn` into an object we then
// link into the `ldscope` binary. These globals them point to the start and
// end of the binary bytes for those files.
extern unsigned char _binary_ldscopedyn_start;
extern unsigned char _binary_ldscopedyn_end;

// Same as above for `libscope.so`.
extern unsigned char _binary_libscope_so_start;
extern unsigned char _binary_libscope_so_end;

// Representation of the .note.gnu.build-id ELF segment 
typedef struct {
    Elf64_Nhdr nhdr;         // Note section header
    char       name[4];      // "GNU\0"
    char       build_id[0];  // build-id bytes, length is nhdr.n_descsz
} note_t;

// from https://github.com/mattst88/build-id/blob/master/build-id.c
#define ALIGN(val, align) (((val) + (align) - 1) & ~((align) - 1))

// ----------------------------------------------------------------------------
// Internal
// ----------------------------------------------------------------------------

static int
libdirExists(const char *path, int requireDir, int mode)
{
    struct stat s;
    if (scope_stat(path, &s)) {
        if (scope_errno != ENOENT) {
            scope_perror("stat() failed");
        }
        return 0;
    }

    if (requireDir && !S_ISDIR(s.st_mode)) {
        return 0; // FALSE
    }
    if (!requireDir && S_ISDIR(s.st_mode)) {
        return 0; // FALSE
    }

    return !scope_access(path, mode);
}

static int
libdirFileExists(const char *path, int mode)
{
    return libdirExists(path, 0, mode);
}

// - Sets global ver
static const char*
libdirGetVer()
{
    if (!g_libdir_info.ver[0]) {
        char *ver = SCOPE_VER;
        char *dash;
        size_t verlen;

        if (*ver == 'v') {
            ++ver;
        }

        if ((dash = scope_strchr(ver, '-'))) {
            verlen = dash - ver;
        } else {
            verlen = scope_strlen(ver);
        }

        if (verlen > PATH_MAX - 1) { // -1 for \0
            scope_fprintf(scope_stderr, "error: libdir too long\n");
            return 0;
        }

        scope_strncpy(g_libdir_info.ver, ver, verlen);
        g_libdir_info.ver[verlen] = '\0';
    }

    return g_libdir_info.ver;
}

static note_t*
libdirGetNote(file_t file)
{
    unsigned char *buf;
    if (file == LOADER_FILE) {
        buf = &_binary_ldscopedyn_start;
    } else {
        buf = &_binary_libscope_so_start;
    }

    Elf64_Ehdr* elf = (Elf64_Ehdr*) buf;
    Elf64_Phdr* hdr = (Elf64_Phdr*) (buf + elf->e_phoff);

    for (unsigned i = 0; i < elf->e_phnum; i++) {
        if (hdr[i].p_type != PT_NOTE) {
            continue;
        }

        note_t*   note = (note_t *)(buf + hdr[i].p_offset);
        Elf64_Off len = hdr[i].p_filesz;
        while (len >= sizeof(note_t)) {
            if (note->nhdr.n_type == NT_GNU_BUILD_ID &&
                note->nhdr.n_descsz != 0 &&
                note->nhdr.n_namesz == 4 &&
                scope_memcmp(note->name, "GNU", 4) == 0) {
                return note;
            }

            // TODO: This needs to be reviewed. It's from
            // https://github.com/mattst88/build-id/blob/master/build-id.c but
            // I'm not entirely sure what it's doing or why. --PDugas
            size_t offset = sizeof(Elf64_Nhdr) +
                            ALIGN(note->nhdr.n_namesz, 4) +
                            ALIGN(note->nhdr.n_descsz, 4);
            note = (note_t *)((char *)note + offset);
            len -= offset;
        }
    }

    return 0;
}

static int
libdirCheckNote(file_t file, const char *path) {
    note_t *note = libdirGetNote(file);
    if (!note) {
        // no note given to compare against so we're done.
        return -1;
    }

    // open & mmap the file to get its note
    int fd = scope_open(path, O_RDONLY);
    if (fd == -1) {
        scope_perror("open() failed");
        return -1;
    }

    struct stat s;
    if (scope_fstat(fd, &s) == -1) {
        scope_close(fd);
        scope_perror("fstat failed");
        return -1;
    }

    void* buf = scope_mmap(NULL, ROUND_UP(s.st_size, scope_sysconf(_SC_PAGESIZE)),
            PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_close(fd);
        scope_perror("scope_mmap() failed");
        return -1;
    }

    scope_close(fd);

    // compare the notes
    int cmp = -1;
    note_t* pathNote = libdirGetNote(file);
    if (pathNote) {
        if (note->nhdr.n_descsz == pathNote->nhdr.n_descsz) {
            cmp = scope_memcmp(note->build_id, pathNote->build_id, note->nhdr.n_descsz);
        }
    }

    scope_munmap(buf, s.st_size);

    return cmp; // 0 if notes match
}

// - Sets global path
int
libdirExtractFile(file_t file, const char* dir)
{
    int fd;
    char temp[PATH_MAX];
    unsigned char *start;
    unsigned char *end;
    char *tmp_path;
    char *path;
    char *filename;

    if (file == LOADER_FILE) {
        filename = SCOPE_LDSCOPEDYN;
        start = &_binary_ldscopedyn_start;
        end = &_binary_ldscopedyn_end;
        path = g_libdir_info.ld_path;
    } else {
        filename = SCOPE_LIBSCOPE_SO;
        start = &_binary_libscope_so_start;
        end = &_binary_libscope_so_end;
        path = g_libdir_info.lib_path;
    }

    int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s", dir, filename);
    if (pathLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (pathLen >= PATH_MAX) {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return -1;
    }


    int tempLen = scope_snprintf(temp, PATH_MAX, "%s.XXXXXX", tmp_path);
    if (tempLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf(0 failed.\n");
        return -1;
    }
    if (tempLen >= PATH_MAX) {
        scope_fprintf(scope_stderr, "error: extract temp too long.\n");
        return -1;
    }

    if ((fd = scope_mkstemp(temp)) < 1) {
        scope_unlink(temp);
        scope_perror("mkstemp() failed");
        return -1;
    }

    size_t len = end - start;
    if (scope_write(fd, start, len) != len) {
        scope_close(fd);
        scope_unlink(temp);
        scope_perror("write() failed");
        return -1;
    }

    // 0755
    if (scope_fchmod(fd, S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)) {
        scope_close(fd);
        scope_unlink(temp);
        scope_perror("fchmod() failed");
        return -1;
    }
    scope_close(fd);

    if (scope_rename(temp, tmp_path)) {
        scope_unlink(temp);
        scope_perror("rename() failed");
        return -1;
    }

    path = tmp_path;
    return 0;
}

// Create directory recursively if missing
// Return extract directory 
// - Sets global ver
// - Sets global base
// - Sets global dir
static char *
libdirCreateDirIfMissing(file_t file)
{
    char *ver;
    char *base;
    char *dir;
    if (file == LOADER_FILE) {
        ver = g_libdir_info.ver;
        base = g_libdir_info.ld_base;
        dir = g_libdir_info.ld_dir;
    } else {
        ver = g_libdir_info.ver;
        base = g_libdir_info.lib_base;
        dir = g_libdir_info.lib_dir;
    }

    ver = (char *)libdirGetVer();

    // stat /usr/lib/appscope/<ver>
    // if exists
    //   set base
    //   set dir
    //   return dir
    // 
    // if root, create /usr/lib/appscope/<ver>
    //   set base
    //   set dir
    //   return dir
    //
    // stat /tmp/appscope/<ver> 
    // if exists
    //   set base
    //   set dir
    //   return dir
    //
    // create /tmp/appscope/<ver>/
    //   set base
    //   set dir
    //   return dir

    // see libdirSetBase() for ideas
        
/* old code
    if (!libdirDirExists(libdir, R_OK|X_OK)) {
        if (scope_mkdir(libdir, S_IRWXU|S_IRWXG|S_IRWXO) == -1) {
            scope_perror("mkdir() failed");
            return -1;
        }
    }
*/

    return 0;
}

// ----------------------------------------------------------------------------
// External
// ----------------------------------------------------------------------------

// - Sets global base
int
libdirSetBase(file_t file, const char *base)
{
    g_libdir_info.ver[0] = 0;
    g_libdir_info.lib_base[0] = 0;
    g_libdir_info.lib_path[0] = 0;
    g_libdir_info.lib_dir[0] = 0;
    g_libdir_info.ld_base[0] = 0;
    g_libdir_info.ld_path[0] = 0;
    g_libdir_info.ld_dir[0] = 0;

    if (base) {
        if (scope_strlen(base) >= PATH_MAX) {
            scope_fprintf(scope_stderr, "error: libdir base path too long.\n");
            return -1;
        }
        if (file == LOADER_FILE) {
            scope_strncpy(g_libdir_info.ld_base, base, sizeof(g_libdir_info.ld_base));
            return 0;
        } else {
            scope_strncpy(g_libdir_info.lib_base, base, sizeof(g_libdir_info.lib_base));
            return 0;
        }
    }

    return -1;
}

// - Sets global path (if file found)
const char *
libdirGetPath(file_t file)
{
    const char *ver = libdirGetVer();
    char *tmp_path;
    char *path;
    char *base;
    char *filename;
    if (file == LOADER_FILE) {
        path = g_libdir_info.ld_path;
        base = g_libdir_info.ld_base;
        filename = SCOPE_LDSCOPEDYN;
    } else {
        path = g_libdir_info.lib_path;
        base = g_libdir_info.lib_base;
        filename = SCOPE_LIBSCOPE_SO;
    }
        
    if (path[0]) {
        return path;
    }

    // Check custom base first
    if (base) {
        int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", base, ver, filename);
        if (pathLen < 0) {
            scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
            return 0;
        }
        if (pathLen >= PATH_MAX) {
            scope_fprintf(scope_stderr, "error: path too long.\n");
            return 0;
        }
        if (!scope_access(tmp_path, R_OK)) {
            path = tmp_path;
            return path;
        }
    }

    // Check install base next
    int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", SCOPE_INSTALL_BASE, ver, filename);
    if (pathLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return 0;
    }
    if (pathLen >= PATH_MAX) {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return 0;
    }
    if (!scope_access(path, R_OK)) {
        path = tmp_path;
        return path;
    }

    // Check temp base next
    pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", SCOPE_TEMP_BASE, ver, filename);
    if (pathLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return 0;
    }
    if (pathLen >= PATH_MAX) {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return 0;
    }
    if (!scope_access(path, R_OK)) {
        path = tmp_path;
        return path;
    }

    return 0;
}

// Does not extract to a custom base
int
libdirExtract(file_t file)
{
    const char *path = libdirGetPath(file);
    if (path) {
        // file exists
        if (!libdirCheckNote(file, path)) {
            // note is ok
            return 0;
        }
    }
    
    char *dir = libdirCreateDirIfMissing(file);
    if (!dir) {
        return -1;
    }

    return libdirExtractFile(file, dir);
}

