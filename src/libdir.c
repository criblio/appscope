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

#ifndef SCOPE_LIBDIR_BASE
#define SCOPE_LIBDIR_BASE "/tmp"
#endif

#ifndef SCOPE_LIBDIR_PREFIX
#define SCOPE_LIBDIR_PREFIX "libscope-"
#endif

#ifndef SCOPE_LDSCOPEDYN
#define SCOPE_LDSCOPEDYN "ldscopedyn"
#endif

#ifndef SCOPE_LIBSCOPE_SO
#define SCOPE_LIBSCOPE_SO "libscope.so"
#endif

#ifndef SCOPE_INSTALL_BASE
#define SCOPE_INSTALL_BASE "/usr/lib/appscope"
#endif

// private global state
static struct {
    char version_dir[PATH_MAX];  // name of the subdirectory under base i.e. libscope-web
    char lib_base[PATH_MAX]; // full path to the library base directory i.e. /tmp or /usr/lib/appscope
    char lib_path[PATH_MAX]; // full path to the library file
    char ld_base[PATH_MAX]; // full path to the loader base directory i.e. /tmp or /usr/lib/appscope
    char ld_path[PATH_MAX]; // full path to the loader file
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
libdirDirExists(const char *path, int mode)
{
    return libdirExists(path, 1, mode);
}

static int
libdirFileExists(const char *path, int mode)
{
    return libdirExists(path, 0, mode);
}

static int
libdirRemove(const char* name, const struct stat *s, int type, struct FTW *ftw)
{
    if (scope_remove(name)) {
        scope_perror("remove() failed");
        return -1;
    }
    return 0;
}

static const char*
libdirGetBase(int file)
{
    if (file == LOADER_FILE) {
        if (g_libdir_info.ld_base[0])
            return g_libdir_info.ld_base;

        char install_path[] = SCOPE_INSTALL_BASE "/" SCOPE_LDSCOPEDYN;
        if (!scope_access(install_path, R_OK)) 
            return SCOPE_INSTALL_BASE;
    } else {
        if (g_libdir_info.lib_base[0])
            return g_libdir_info.lib_base;

        char install_path[] = SCOPE_INSTALL_BASE "/" SCOPE_LIBSCOPE_SO;
        if (!scope_access(install_path, R_OK)) 
            return SCOPE_INSTALL_BASE;
    }

    return SCOPE_LIBDIR_BASE;
}

static const char*
libdirGetDir()
{
    if (!g_libdir_info.version_dir[0]) {
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

        if (verlen > PATH_MAX - scope_strlen(SCOPE_LIBDIR_PREFIX) - 1) { // -1 for \0
            scope_fprintf(scope_stderr, "error: libdir too long\n");
            return 0;
        }

        scope_strncpy(g_libdir_info.version_dir, SCOPE_LIBDIR_PREFIX, sizeof(g_libdir_info.version_dir));
        scope_strncat(g_libdir_info.version_dir, ver, verlen);
    }

    return g_libdir_info.version_dir;
}

static note_t*
libdirGetNote(int file)
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

int
libdirExtractFileTo(int file, const char* path)
{
    unsigned char *start;
    unsigned char *end;

    if (file == LOADER_FILE) {
        start = &_binary_ldscopedyn_start;
        end = &_binary_ldscopedyn_end;
    } else {
        start = &_binary_libscope_so_start;
        end = &_binary_libscope_so_end;
    }

    int fd;
    char temp[PATH_MAX];

    int tempLen = scope_snprintf(temp, PATH_MAX, "%s.XXXXXX", path);
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

    if (scope_rename(temp, path)) {
        scope_unlink(temp);
        scope_perror("rename() failed");
        return -1;
    }
    return 0;
}

static int
libdirCreateDirsIfMissing(int file)
{
    // Create base dir if missing
    
    // Create version dir if missing
    

/*    const char *libdir = libDirGetBase(file) + libdirGetDir(file) ;

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

int
libdirSetBase(int file, const char *base)
{
    g_libdir_info.version_dir[0] = 0;
    g_libdir_info.lib_base[0] = 0;
    g_libdir_info.lib_path[0] = 0;
    g_libdir_info.ld_base[0] = 0;
    g_libdir_info.ld_path[0] = 0;

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

const char *
libdirGetPath(int file)
{
    if (file == LOADER_FILE) {
        if (!g_libdir_info.ld_path[0]) {
            int pathLen = scope_snprintf(g_libdir_info.ld_path, PATH_MAX, "%s/%s/%s", libdirGetBase(file), libdirGetDir(), SCOPE_LDSCOPEDYN);
            if (pathLen < 0) {
                scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
                return 0;
            }
            if (pathLen >= PATH_MAX) {
                scope_fprintf(scope_stderr, "error: loader path too long.\n");
                return 0;
            }
        }
        return &g_libdir_info.ld_path[0];
    } else {
        if (!g_libdir_info.lib_path[0]) {
            int pathLen = scope_snprintf(g_libdir_info.lib_path, PATH_MAX, "%s/%s/%s", libdirGetBase(file), libdirGetDir(), SCOPE_LIBSCOPE_SO);
            if (pathLen < 0) {
                scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
                return 0;
            }
            if (pathLen >= PATH_MAX) {
                scope_fprintf(scope_stderr, "error: library path too long.\n");
                return 0;
            }
        }
        return &g_libdir_info.lib_path[0];
    }
}

int
libdirClean()
{
    int err = 0;

    char *ld_dir;
    int pathLen = scope_snprintf(ld_dir, PATH_MAX, "%s/%s", libdirGetBase(LOADER_FILE), libdirGetDir());
    if (pathLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (nftw(ld_dir, libdirRemove, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS)) {
        scope_perror("ntfw() failed");
        err = -1;
    }

    char *lib_dir;
    pathLen = scope_snprintf(lib_dir, PATH_MAX, "%s/%s", libdirGetBase(LIBRARY_FILE), libdirGetDir());
    if (pathLen < 0) {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (nftw(lib_dir, libdirRemove, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS)) {
        scope_perror("ntfw() failed");
        err = -1;
    }

    return err;
}

int
libdirExtract(int file)
{
    unsigned char *start;
    unsigned char *end;

    if (file == LOADER_FILE) {
        start = &_binary_ldscopedyn_start;
        end = &_binary_ldscopedyn_end;
    } else {
        start = &_binary_libscope_so_start;
        end = &_binary_libscope_so_end;
    }

    int location = libdirCreateDirsIfMissing(file);
    if (location < 0) {
        return -1;
    }
        
    const char *path = libdirGetPath(file);
    if (libdirFileExists(path, R_OK|X_OK)) {
        // extracted file already exists

        note_t *note = libdirGetNote(file);
        if (!note) {
            // no note given to compare against so we're done.
            return 0;
        }

        // open & mmap the file to get its note
        int fd = scope_open(path, O_RDONLY);
        if (fd == -1) {
            scope_perror("open() failed");
            return 0;
        }

        struct stat s;
        if (scope_fstat(fd, &s) == -1) {
            scope_close(fd);
            scope_perror("fstat failed");
            return 0;
        }

        void* buf = scope_mmap(NULL, ROUND_UP(s.st_size, scope_sysconf(_SC_PAGESIZE)),
                PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
        if (buf == MAP_FAILED) {
            scope_close(fd);
            scope_perror("scope_mmap() failed");
            return 0;
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

        if (cmp == 0) {
            // notes match, don't re-extract
            return 0;
        }
    }

    return libdirExtractFileTo(file, path);
}

