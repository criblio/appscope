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
#include "libver.h"

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
static struct
{
    char ver[PATH_MAX];      // name of the subdirectory under base i.e. 1.0.0
    char lib_base[PATH_MAX]; // full path to the library base directory i.e. /tmp/appscope or /usr/lib/appscope
    char lib_path[PATH_MAX]; // full path to the library file
    char ld_base[PATH_MAX];  // full path to the loader base directory i.e. /tmp/appscope or /usr/lib/appscope
    char ld_path[PATH_MAX];  // full path to the loader file
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
typedef struct
{
    Elf64_Nhdr nhdr;  // Note section header
    char name[4];     // "GNU\0"
    char build_id[0]; // build-id bytes, length is nhdr.n_descsz
} note_t;

// from https://github.com/mattst88/build-id/blob/master/build-id.c
#define ALIGN(val, align) (((val) + (align)-1) & ~((align)-1))

// ----------------------------------------------------------------------------
// Internal
// ----------------------------------------------------------------------------

// Get version directory name
// - Sets global ver
static const char *
libdirGetVer()
{
    if (!g_libdir_info.ver[0])
    {
        char *ver = SCOPE_VER;
        char *dash;
        size_t verlen;

        if (*ver == 'v')
        {
            ++ver;
        }

        if ((dash = scope_strchr(ver, '-')))
        {
            verlen = dash - ver;
        }
        else
        {
            verlen = scope_strlen(ver);
        }

        if (verlen > PATH_MAX - 1)
        { // -1 for \0
            scope_fprintf(scope_stderr, "error: libdir too long\n");
            return 0;
        }

        const char *version = libverNormalizedVersion(ver);

        scope_strncpy(g_libdir_info.ver, version, scope_strlen(version));
        g_libdir_info.ver[verlen] = '\0';
    }

    return g_libdir_info.ver;
}

static note_t *
libdirGetNote(file_t file)
{
    unsigned char *buf;
    if (file == LOADER_FILE)
    {
        buf = &_binary_ldscopedyn_start;
    }
    else
    {
        buf = &_binary_libscope_so_start;
    }

    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *hdr = (Elf64_Phdr *)(buf + elf->e_phoff);

    for (unsigned i = 0; i < elf->e_phnum; i++)
    {
        if (hdr[i].p_type != PT_NOTE)
        {
            continue;
        }

        note_t *note = (note_t *)(buf + hdr[i].p_offset);
        Elf64_Off len = hdr[i].p_filesz;
        while (len >= sizeof(note_t))
        {
            if (note->nhdr.n_type == NT_GNU_BUILD_ID &&
                note->nhdr.n_descsz != 0 &&
                note->nhdr.n_namesz == 4 &&
                scope_memcmp(note->name, "GNU", 4) == 0)
            {
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
libdirCheckNote(file_t file, const char *path)
{
    note_t *note = libdirGetNote(file);
    if (!note)
    {
        // no note given to compare against so we're done.
        return -1;
    }

    // open & mmap the file to get its note
    int fd = scope_open(path, O_RDONLY);
    if (fd == -1)
    {
        scope_perror("open() failed");
        return -1;
    }

    struct stat s;
    if (scope_fstat(fd, &s) == -1)
    {
        scope_close(fd);
        scope_perror("fstat failed");
        return -1;
    }

    void *buf = scope_mmap(NULL, ROUND_UP(s.st_size, scope_sysconf(_SC_PAGESIZE)),
                           PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED)
    {
        scope_close(fd);
        scope_perror("scope_mmap() failed");
        return -1;
    }

    scope_close(fd);

    // compare the notes
    int cmp = -1;
    note_t *pathNote = libdirGetNote(file);
    if (pathNote)
    {
        if (note->nhdr.n_descsz == pathNote->nhdr.n_descsz)
        {
            cmp = scope_memcmp(note->build_id, pathNote->build_id, note->nhdr.n_descsz);
        }
    }

    scope_munmap(buf, s.st_size);

    return cmp; // 0 if notes match
}

static int
libdirCreateFileIfMissing(file_t file, const char *path)
{
    // Check if file exists
    if (!scope_access(path, R_OK))
    {
        return 0; // File exists
    }

    int fd;
    char temp[PATH_MAX];
    unsigned char *start;
    unsigned char *end;

    if (file == LOADER_FILE)
    {
        start = &_binary_ldscopedyn_start;
        end = &_binary_ldscopedyn_end;
    }
    else
    {
        start = &_binary_libscope_so_start;
        end = &_binary_libscope_so_end;
    }

    // Write file
    int tempLen = scope_snprintf(temp, PATH_MAX, "%s.XXXXXX", path);
    if (tempLen < 0)
    {
        scope_fprintf(scope_stderr, "error: snprintf(0 failed.\n");
        return -1;
    }
    if (tempLen >= PATH_MAX)
    {
        scope_fprintf(scope_stderr, "error: extract temp too long.\n");
        return -1;
    }
    if ((fd = scope_mkstemp(temp)) < 1)
    {
        // No permission
        scope_unlink(temp);
        return -1;
    }
    size_t len = end - start;
    if (scope_write(fd, start, len) != len)
    {
        scope_close(fd);
        scope_unlink(temp);
        scope_perror("write() failed");
        return -1;
    }
    if (scope_fchmod(fd, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH))
    { // 0755
        scope_close(fd);
        scope_unlink(temp);
        scope_perror("fchmod() failed");
        return -1;
    }
    scope_close(fd);
    if (scope_rename(temp, path))
    {
        scope_unlink(temp);
        scope_perror("rename() failed");
        return -1;
    }

    return 0;
}

// Verify if following absolute path points to directory
// Returns operation status
static mkdir_status_t
libdirCheckIfDirExists(const char *absDirPath)
{
    struct stat st = {0};
    if (!scope_stat(absDirPath, &st)) {
        if (S_ISDIR(st.st_mode)) {      
            // Check for file creation abilities in directory  
            if (((st.st_uid == scope_getuid()) && (st.st_mode & S_IWUSR)) ||
                ((st.st_gid == scope_getgid()) && (st.st_mode & S_IWGRP)) ||
                (st.st_mode & S_IWOTH)) {
                return MKDIR_STATUS_EXISTS;
            }
            return MKDIR_STATUS_ERR_PERM_ISSUE;
        }
        return MKDIR_STATUS_ERR_NOT_ABS_DIR;
    }
    return MKDIR_STATUS_ERR_OTHER;
}

// ----------------------------------------------------------------------------
// External
// ----------------------------------------------------------------------------

// Create a directory in following absolute path creating any intermediate directories as necessary
// Returns operation status
mkdir_status_t
libdirCreateDirIfMissing(const char *dir) {
    int mkdirRes = -1;
    /* Operate only on absolute path */
    if (dir == NULL || *dir != '/') {
        return MKDIR_STATUS_ERR_NOT_ABS_DIR;
    }

    mkdir_status_t res = libdirCheckIfDirExists(dir);

    /* exit if path exists */
    if (res != MKDIR_STATUS_ERR_OTHER) {
        return res;
    }

    char *tempPath = scope_strdup(dir);
    if (tempPath == NULL) {
        goto end;
    }

    /* traverse the full path */
    for (char *p = tempPath + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';
            scope_errno = 0;
            mkdirRes = scope_mkdir(tempPath, 0755);
            /* scope_mkdir fails with error other than directory exists */
            if (mkdirRes && (scope_errno != EEXIST)) {
                goto end;
            }
            *p = '/';
        }
    }

    /* last element */
    scope_errno = 0;
    mkdirRes = scope_mkdir(tempPath, 0755);
    if (mkdirRes && (scope_errno != EEXIST)) {
        goto end;
    }

    res = MKDIR_STATUS_CREATED;

end:
    scope_free(tempPath);
    return res;
}

// Set custom base directory of library
// - Sets global base
int libdirSetBase(file_t file, const char *base)
{
    g_libdir_info.ver[0] = 0;
    g_libdir_info.lib_base[0] = 0;
    g_libdir_info.lib_path[0] = 0;
    g_libdir_info.ld_base[0] = 0;
    g_libdir_info.ld_path[0] = 0;

    if (base)
    {
        if (scope_strlen(base) >= PATH_MAX)
        {
            scope_fprintf(scope_stderr, "error: libdir base path too long.\n");
            return -1;
        }
        if (file == LOADER_FILE)
        {
            scope_strncpy(g_libdir_info.ld_base, base, sizeof(g_libdir_info.ld_base));
            return 0;
        }
        else
        {
            scope_strncpy(g_libdir_info.lib_base, base, sizeof(g_libdir_info.lib_base));
            return 0;
        }
    }

    return -1;
}

// Get path of existing file
// - Sets global path (if file found)
const char *
libdirGetPath(file_t file)
{
    const char *ver = libdirGetVer();
    char tmp_path[PATH_MAX];
    ;
    char *path;
    char *base;
    char *filename;
    if (file == LOADER_FILE)
    {
        path = g_libdir_info.ld_path;
        base = g_libdir_info.ld_base;
        filename = SCOPE_LDSCOPEDYN;
    }
    else
    {
        path = g_libdir_info.lib_path;
        base = g_libdir_info.lib_base;
        filename = SCOPE_LIBSCOPE_SO;
    }

    if (path[0])
    {
        return path;
    }

    // Check custom base first
    if (base[0])
    {
        int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", base, ver, filename);
        if (pathLen < 0)
        {
            scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
            return 0;
        }
        if (pathLen >= PATH_MAX)
        {
            scope_fprintf(scope_stderr, "error: path too long.\n");
            return 0;
        }
        if (!scope_access(tmp_path, R_OK))
        {
            scope_strncpy(path, tmp_path, PATH_MAX);
            return path;
        }
    }

    // Check install base next
    int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", SCOPE_INSTALL_BASE, ver, filename);
    if (pathLen < 0)
    {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return 0;
    }
    if (pathLen >= PATH_MAX)
    {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return 0;
    }
    if (!scope_access(tmp_path, R_OK))
    {
        scope_strncpy(path, tmp_path, PATH_MAX);
        return path;
    }

    // Check temp base next
    pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s/%s", SCOPE_TEMP_BASE, ver, filename);
    if (pathLen < 0)
    {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return 0;
    }
    if (pathLen >= PATH_MAX)
    {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return 0;
    }
    if (!scope_access(tmp_path, R_OK))
    {
        scope_strncpy(path, tmp_path, PATH_MAX);
        return path;
    }

    return 0;
}

// Save libscope.so in specified path.
// returns 0 if file was successfully created or if file already exists, -1 in case of failure
int
libdirSaveLibraryFile(const char *libraryPath) {
    return libdirCreateFileIfMissing(LIBRARY_FILE, libraryPath);
}

// Does not extract to a custom base
// - Sets global base
// - Sets global path
int libdirExtract(file_t file)
{
    const char *existing_path = libdirGetPath(file);
    if (existing_path)
    {
        // file exists
        if (!libdirCheckNote(file, existing_path))
        {
            // note is ok
            return 0;
        }
    }

    const char *ver = libdirGetVer();
    char tmp_dir[PATH_MAX];
    char tmp_path[PATH_MAX];
    char *filename;
    char *path;
    char *base;
    if (file == LOADER_FILE)
    {
        filename = SCOPE_LDSCOPEDYN;
        path = g_libdir_info.ld_path;
        base = g_libdir_info.ld_base;
    }
    else
    {
        filename = SCOPE_LIBSCOPE_SO;
        path = g_libdir_info.lib_path;
        base = g_libdir_info.lib_base;
    }

    int pathLen = scope_snprintf(tmp_dir, PATH_MAX, "%s/%s", SCOPE_INSTALL_BASE, ver);
    if (pathLen < 0)
    {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (pathLen >= PATH_MAX)
    {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return -1;
    }
    if (libdirCreateDirIfMissing(tmp_dir) <= MKDIR_STATUS_EXISTS)
    {
        int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s", tmp_dir, filename);
        if (pathLen < 0)
        {
            scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX)
        {
            scope_fprintf(scope_stderr, "error: path too long.\n");
            return -1;
        }
        if (!libdirCreateFileIfMissing(file, tmp_path))
        {
            scope_strncpy(path, tmp_path, PATH_MAX);
            scope_strncpy(base, SCOPE_INSTALL_BASE, PATH_MAX);
            return 0;
        }
    }

    pathLen = scope_snprintf(tmp_dir, PATH_MAX, "%s/%s", SCOPE_TEMP_BASE, ver);
    if (pathLen < 0)
    {
        scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (pathLen >= PATH_MAX)
    {
        scope_fprintf(scope_stderr, "error: path too long.\n");
        return -1;
    }
    if (libdirCreateDirIfMissing(tmp_dir) <= MKDIR_STATUS_EXISTS)
    {
        int pathLen = scope_snprintf(tmp_path, PATH_MAX, "%s/%s", tmp_dir, filename);
        if (pathLen < 0)
        {
            scope_fprintf(scope_stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX)
        {
            scope_fprintf(scope_stderr, "error: path too long.\n");
            return -1;
        }
        if (!libdirCreateFileIfMissing(file, tmp_path))
        {
            scope_strncpy(path, tmp_path, PATH_MAX);
            scope_strncpy(base, SCOPE_TEMP_BASE, PATH_MAX);
            return 0;
        }
    }

    return -1;
}
