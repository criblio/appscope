/**
 * Cribl AppScope - Library Directory Implementation
 */

#include "libdir.h"

#define _XOPEN_SOURCE 500 // for FTW

#include <errno.h>
#include <ftw.h>
#include <linux/limits.h> // for PATH_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
extern unsigned char _binary___bin_linux_ldscopedyn_start;
extern unsigned char _binary___bin_linux_ldscopedyn_end;

// Same as above for `libscope.so`.
extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

// ----------------------------------------------------------------------------
// Internal
// ----------------------------------------------------------------------------

static int
_libdirExists(const char *path)
{
    struct stat s;
    if (stat(path, &s)) {
        if (errno != ENOENT) {
            perror("error: stat() failed");
        }
        return 0;
    }

    return S_ISDIR(s.st_mode) && !access(path, R_OK|W_OK|X_OK);
}

static int
_libdirCreateIfMissing()
{
    const char *libdir = libdirGet();

    if (!_libdirExists(libdir)) {
        if (mkdir(libdir, S_IRWXU|S_IRWXG|S_IRWXO) == -1) {
            perror("error: mkdir() failed");
            return -1;
        }
    }

    return 0;
}

static int
_libdirExtract(const char *path, unsigned char *start, unsigned char *end)
{
    char temp[PATH_MAX];
    int fd;

    if (_libdirCreateIfMissing()) {
        return -1;
    }

    if (snprintf(temp, PATH_MAX, "%s.XXXXXX", path) >= PATH_MAX) {
        fprintf(stderr, "error: extract temp too long.\n");
        return -1;
    }

    if ((fd = mkstemp(temp)) < 1) {
        unlink(temp);
        perror("error: mkstemp() failed");
        return -1;
    }

    size_t len = end - start;
    if (write(fd, start, len) != len) {
        close(fd);
        unlink(temp);
        perror("error: write() failed");
        return -1;
    }

    if (fchmod(fd, S_IRWXU|S_IRWXG|S_IRWXO)) {
        close(fd);
        unlink(temp);
        perror("error: fchmod() failed");
        return -1;
    }
    close(fd);

    if (rename(temp, path)) {
        unlink(temp);
        perror("error: rename() failed");
        return -1;
    }

    return 0;
}

static int
_libdirRemove(const char* name, const struct stat *s, int type, struct FTW *ftw)
{
    if (remove(name)) {
        perror("error: remove() failed");
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
        strcpy(g_libdir_info.base, base);
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

        strcpy(g_libdir_info.dir, SCOPE_LIBDIR_PREFIX);
        strncat(g_libdir_info.dir, ver, verlen);
    }

    return g_libdir_info.dir;
}

const char*
libdirGet()
{
    if (!g_libdir_info.path[0]) {
        if (snprintf(g_libdir_info.path, PATH_MAX, "%s/%s", libdirGetBase(), libdirGetDir()) >= PATH_MAX) {
            fprintf(stderr, "error: libdir path too long.\n");
            return 0;
        }
    }

    return g_libdir_info.path;
}

int
libdirClean()
{
    if (nftw(libdirGet(), _libdirRemove, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS)) {
        perror("error: ntfw() failed");
        return -1;
    }

    return 0;
}

int
libdirExtractLauncher()
{
    return _libdirExtract(libdirGetLauncher(),
            &_binary___bin_linux_ldscopedyn_start,
            &_binary___bin_linux_ldscopedyn_end);
}

int
libdirExtractLibrary()
{
    return _libdirExtract(libdirGetLibrary(),
            &_binary___lib_linux_libscope_so_start,
            &_binary___lib_linux_libscope_so_end);
}

const char *
libdirGetLauncher()
{
    static char path[PATH_MAX];

    if (!path[0]) {
        if (snprintf(path, PATH_MAX, "%s/" SCOPE_LDSCOPEDYN, libdirGet()) >= PATH_MAX) {
            fprintf(stderr, "error: launcher path too long.\n");
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
        if (snprintf(path, PATH_MAX, "%s/" SCOPE_LIBSCOPE_SO, libdirGet()) >= PATH_MAX) {
            fprintf(stderr, "error: launcher path too long.\n");
            return 0;
        }
    }

    return path;
}

// EOF
