/**
 * Cribl AppScope - Library Directory Implementation
 */

#include "libdir.h"

#define _XOPEN_SOURCE 500 // for FTW

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

// private global state
static struct {
    char base[PATH_MAX]; // full path to the base directory
    char dir[PATH_MAX];  // name of the subdirectory under base
    char path[PATH_MAX]; // full path to our libdir
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
    if (stat(path, &s)) {
        if (errno != ENOENT) {
            perror("stat() failed");
        }
        return 0;
    }

    if (requireDir && !S_ISDIR(s.st_mode)) {
        return 0; // FALSE
    }
    if (!requireDir && S_ISDIR(s.st_mode)) {
        return 0; // FALSE
    }

    return !access(path, mode);
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
libdirCreateIfMissing()
{
    const char *libdir = libdirGet();

    if (!libdirDirExists(libdir, R_OK|X_OK)) {
        if (mkdir(libdir, S_IRWXU|S_IRWXG|S_IRWXO) == -1) {
            perror("mkdir() failed");
            return -1;
        }
    }

    return 0;
}

static note_t*
libdirGetNote(void* buf)
{
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
                memcmp(note->name, "GNU", 4) == 0) {
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

static note_t*
libdirGetLoaderNote()
{
    return libdirGetNote(&_binary_ldscopedyn_start);
}

static note_t*
libdirGetLibraryNote()
{
    return libdirGetNote(&_binary_libscope_so_start);
}

static int
libdirExtract(const char *path, unsigned char *start, unsigned char *end, note_t* note)
{
    char temp[PATH_MAX];
    int fd;

    if (libdirCreateIfMissing()) {
        return -1;
    }

    if (libdirFileExists(path, R_OK|X_OK)) {
        // extracted file already exists

        if (!note) {
            // no note given to compare against so we're done.
            return 0;
        }

        // open & mmap the file to get its note
        int fd = open(path, O_RDONLY);
        if (fd == -1) {
            perror("open() failed");
            return 0;
        }

        struct stat s;
        if (fstat(fd, &s) == -1) {
            close(fd);
            perror("mmap() failed");
            return 0;
        }

        void* buf = mmap(NULL, ROUND_UP(s.st_size, sysconf(_SC_PAGESIZE)),
                PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
        if (buf == MAP_FAILED) {
            close(fd);
            perror("mmap() failed");
            return 0;
        }

        close(fd);

        // compare the notes
        int cmp = -1;
        note_t* pathNote = libdirGetNote(buf);
        if (pathNote) {
            if (note->nhdr.n_descsz == pathNote->nhdr.n_descsz) {
                cmp = memcmp(note->build_id, pathNote->build_id, note->nhdr.n_descsz);
            }
        }

        munmap(buf, s.st_size);

        if (cmp == 0) {
            // notes match, don't re-extract
            return 0;
        }
    }

    int tempLen = snprintf(temp, PATH_MAX, "%s.XXXXXX", path);
    if (tempLen < 0) {
        fprintf(stderr, "error: snprintf(0 failed.\n");
        return -1;
    }
    if (tempLen >= PATH_MAX) {
        fprintf(stderr, "error: extract temp too long.\n");
        return -1;
    }

    if ((fd = mkstemp(temp)) < 1) {
        unlink(temp);
        perror("mkstemp() failed");
        return -1;
    }

    size_t len = end - start;
    if (write(fd, start, len) != len) {
        close(fd);
        unlink(temp);
        perror("write() failed");
        return -1;
    }

    // 0755
    if (fchmod(fd, S_IRWXU|S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH)) {
        close(fd);
        unlink(temp);
        perror("fchmod() failed");
        return -1;
    }
    close(fd);

    if (rename(temp, path)) {
        unlink(temp);
        perror("rename() failed");
        return -1;
    }

    return 0;
}

static int
libdirRemove(const char* name, const struct stat *s, int type, struct FTW *ftw)
{
    if (remove(name)) {
        perror("remove() failed");
        return -1;
    }
    return 0;
}

// ----------------------------------------------------------------------------
// External
// ----------------------------------------------------------------------------

int
libdirSetBase(const char *base)
{
    g_libdir_info.base[0] = 0;
    g_libdir_info.dir[0] = 0;
    g_libdir_info.path[0] = 0;

    if (base) {
        if (strlen(base) >= PATH_MAX) {
            fprintf(stderr, "error: libdir base path too long.\n");
            return -1;
        }
        strncpy(g_libdir_info.base, base, sizeof(g_libdir_info.base));
    }

    return 0;
}

const char*
libdirGetBase()
{
    return g_libdir_info.base[0]
        ? g_libdir_info.base
        : SCOPE_LIBDIR_BASE;
}

const char*
libdirGetDir()
{
    if (!g_libdir_info.dir[0]) {
        char *ver = SCOPE_VER;
        char *dash;
        size_t verlen;

        if (*ver == 'v') {
            ++ver;
        }

        if ((dash = strchr(ver, '-'))) {
            verlen = dash - ver;
        } else {
            verlen = strlen(ver);
        }

        if (verlen > PATH_MAX - strlen(SCOPE_LIBDIR_PREFIX) - 1) { // -1 for \0
            fprintf(stderr, "error: libdir too long\n");
            return 0;
        }

        strncpy(g_libdir_info.dir, SCOPE_LIBDIR_PREFIX, sizeof(g_libdir_info.dir));
        strncat(g_libdir_info.dir, ver, verlen);
    }

    return g_libdir_info.dir;
}

const char*
libdirGet()
{
    if (!g_libdir_info.path[0]) {
        int pathLen = snprintf(g_libdir_info.path, PATH_MAX, "%s/%s", libdirGetBase(), libdirGetDir());
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return 0;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: libdir path too long.\n");
            return 0;
        }
    }

    return g_libdir_info.path;
}

int
libdirClean()
{
    if (nftw(libdirGet(), libdirRemove, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS)) {
        perror("ntfw() failed");
        return -1;
    }

    return 0;
}

int
libdirExtractLoader()
{
    return libdirExtract(libdirGetLoader(),
            &_binary_ldscopedyn_start,
            &_binary_ldscopedyn_end,
            libdirGetLoaderNote());
}

int
libdirExtractLibrary()
{
    return libdirExtract(libdirGetLibrary(),
            &_binary_libscope_so_start,
            &_binary_libscope_so_end,
            libdirGetLibraryNote());
}

const char *
libdirGetLoader()
{
    static char path[PATH_MAX];

    if (!path[0]) {
        int pathLen = snprintf(path, PATH_MAX, "%s/" SCOPE_LDSCOPEDYN, libdirGet());
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return 0;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: loader path too long.\n");
            return 0;
        }
    }

    return path;
}

const char *
libdirGetLibrary()
{
    static char path[PATH_MAX];

    if (!path[0]) {
        int pathLen = snprintf(path, PATH_MAX, "%s/" SCOPE_LIBSCOPE_SO, libdirGet());
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return 0;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: loader path too long.\n");
            return 0;
        }
    }

    return path;
}

// EOF
