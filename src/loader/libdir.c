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
#include "nsfile.h"
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

#define SCOPE_NAME_SIZE (16)

// private global state
static struct
{
    char ver[PATH_MAX];          // contains raw version
    char install_base[PATH_MAX]; // full path to the desired install base directory
    char tmp_base[PATH_MAX];     // full path to the desired tmp base directory
} g_libdir_info = {
    .ver = SCOPE_VER,                    // default version
    .install_base = "/usr/lib/appscope", // default install base
    .tmp_base = "/tmp/appscope",         // default tmp base
};

// internal state object structure
struct scope_obj_state{
    char binaryName[SCOPE_NAME_SIZE];    // name of the binary
    char binaryBasepath[PATH_MAX];       // full path to the actual binary base directory i.e. /tmp/appscope or /usr/lib/appscope
    char binaryPath[PATH_MAX];           // full path to the actual binary file i.e. /tmp/appscope/dev/libscope.so
};

// internal state for dynamic loader
static struct scope_obj_state ldscopeDynState = {
    .binaryName = SCOPE_LDSCOPEDYN,
};

// internal state for library
static struct scope_obj_state libscopeState = {
    .binaryName = SCOPE_LIBSCOPE_SO,
};

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

static struct scope_obj_state *
getObjState(libdirfile_t objFileType) {
    switch (objFileType) {
        case LOADER_FILE:
            return &ldscopeDynState;
        case LIBRARY_FILE:
            return &libscopeState;
    }
    // unreachable
    return NULL;
}


static note_t *
libdirGetNote(libdirfile_t objFileType) {
    unsigned char *buf;

    switch (objFileType) {
        case LOADER_FILE:
            buf = &_binary_ldscopedyn_start;
            break;
        case LIBRARY_FILE:
            buf = &_binary_libscope_so_start;
            break;
        default:
            // unreachable
            return NULL;
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
                memcmp(note->name, "GNU", 4) == 0)
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

    return NULL;
}

static int
libdirCheckNote(libdirfile_t objFileType, const char *path)
{
    note_t *note = libdirGetNote(objFileType);
    if (!note) {
        // no note given to compare against so we're done.
        return -1;
    }

    // open & mmap the file to get its note
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        perror("open() failed");
        return -1;
    }

    struct stat s;
    if (fstat(fd, &s) == -1) {
        close(fd);
        perror("fstat failed");
        return -1;
    }

    void *buf = mmap(NULL, ROUND_UP(s.st_size, sysconf(_SC_PAGESIZE)),
                           PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        close(fd);
        perror("mmap() failed");
        return -1;
    }

    close(fd);

    // compare the notes
    int cmp = -1;
    note_t *pathNote = libdirGetNote(objFileType);
    if (pathNote) {
        if (note->nhdr.n_descsz == pathNote->nhdr.n_descsz) {
            cmp = memcmp(note->build_id, pathNote->build_id, note->nhdr.n_descsz);
        }
    }

    munmap(buf, s.st_size);

    return cmp; // 0 if notes match
}

static int
libdirCreateFileIfMissing(libdirfile_t objFileType, const char *path, bool overwrite, mode_t mode, uid_t nsEuid, gid_t nsEgid)
{
    // Check if file exists
    if (!access(path, R_OK) && !overwrite) {
        return 0; // File exists
    }

    int fd;
    char temp[PATH_MAX];
    unsigned char *start;
    unsigned char *end;

    switch (objFileType) {
        case LOADER_FILE:
            start = &_binary_ldscopedyn_start;
            end = &_binary_ldscopedyn_end;
            break;
        case LIBRARY_FILE:
            start = &_binary_libscope_so_start;
            end = &_binary_libscope_so_end;
            break;
        default:
            // unreachable
            return -1;
    }

    // Write file
    int tempLen = snprintf(temp, PATH_MAX, "%s.XXXXXX", path);
    if (tempLen < 0) {
        fprintf(stderr, "error: snprintf(0 failed.\n");
        return -1;
    }
    if (tempLen >= PATH_MAX) {
        fprintf(stderr, "error: extract temp too long.\n");
        return -1;
    }

    uid_t currentEuid = geteuid();
    gid_t currentEgid = getegid();

    if ((fd = nsFileMksTemp(temp, nsEuid, nsEgid, currentEuid, currentEgid)) < 1) {
        // No permission
        unlink(temp);
        return -1;
    }
    size_t len = end - start;
    if (write(fd, start, len) != len) {
        close(fd);
        unlink(temp);
        perror("libdirCreateFileIfMissing: write() failed");
        return -1;
    }
    if (fchmod(fd, mode)) {
        close(fd);
        unlink(temp);
        perror("libdirCreateFileIfMissing: fchmod() failed");
        return -1;
    }

    close(fd);
    if (nsFileRename(temp, path, nsEuid, nsEgid, currentEuid, currentEgid)) {
        unlink(temp);
        perror("libdirCreateFileIfMissing: rename() failed");
        return -1;
    }

    return 0;
}

// Verify if following absolute path points to directory
// Returns operation status
static mkdir_status_t
libdirCheckIfDirExists(const char *absDirPath, uid_t uid, gid_t gid)
{
    struct stat st = {0};
    if (!stat(absDirPath, &st)) {
        if (S_ISDIR(st.st_mode)) {      
            // Check for file creation abilities in directory  
            if (((st.st_uid == uid) && (st.st_mode & S_IWUSR)) ||
                ((st.st_gid == gid) && (st.st_mode & S_IWGRP)) ||
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

// Override default values (function is used only for unit test)
int
libdirInitTest(const char *installBase, const char *tmpBase, const char *rawVersion) {
    memset(&g_libdir_info, 0, sizeof(g_libdir_info));
    memset(&libscopeState, 0, sizeof(libscopeState));
    memset(&ldscopeDynState, 0, sizeof(ldscopeDynState));
    strcpy(ldscopeDynState.binaryName, SCOPE_LDSCOPEDYN);
    strcpy(libscopeState.binaryName, SCOPE_LIBSCOPE_SO);
          
    if (installBase) {
        int len = strlen(installBase);
        if (len >= PATH_MAX) {
            fprintf(stderr, "error: installBase path too long.\n");
            return -1;
        }
        strncpy(g_libdir_info.install_base, installBase, len);
    } else {
        strcpy(g_libdir_info.install_base, "/usr/lib/appscope");
    }

    if (tmpBase) {
        int len = strlen(tmpBase);
        if (len >= PATH_MAX){
            fprintf(stderr, "error: tmpBase path too long.\n");
            return -1;
        }
        strncpy(g_libdir_info.tmp_base, tmpBase, len);
    } else {
        strcpy(g_libdir_info.tmp_base, "/tmp/appscope");
    }

    if (rawVersion) {
        int len = strlen(rawVersion);
        if (len >= PATH_MAX){
            fprintf(stderr, "error: rawVersion too long.\n");
            return -1;
        }
        strncpy(g_libdir_info.ver, rawVersion, len);
    } else {
        strcpy(g_libdir_info.ver, SCOPE_VER);
    }

    return 0;
}

// Create a directory in following absolute path creating any intermediate directories as necessary
// Returns operation status
mkdir_status_t
libdirCreateDirIfMissing(const char *dir, mode_t mode, uid_t nsEuid, gid_t nsEgid) {
    int mkdirRes = -1;
    /* Operate only on absolute path */
    if (dir == NULL || *dir != '/') {
        return MKDIR_STATUS_ERR_NOT_ABS_DIR;
    }

    mkdir_status_t res = libdirCheckIfDirExists(dir, nsEuid, nsEgid);

    /* exit if path exists */
    if (res != MKDIR_STATUS_ERR_OTHER) {
        return res;
    }

    char *tempPath = strdup(dir);
    if (tempPath == NULL) {
        goto end;
    }

    uid_t euid = geteuid();
    gid_t egid = getegid();

    /* traverse the full path */
    for (char *p = tempPath + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';
            errno = 0;

            struct stat st = {0};
            if (stat(tempPath, &st)) {
                mkdirRes = nsFileMkdir(tempPath, mode, nsEuid, nsEgid, euid, egid);
                if (!mkdirRes) {
                    /* We ensure that we setup correct mode regarding umask settings */
                    if (chmod(tempPath, mode)) {
                        goto end;
                    }
                } else {
                    /* nsFileMkdir fails */
                    goto end;
                }
            }

            *p = '/';
        }
    }
    struct stat st = {0};
    if (stat(tempPath, &st)) {
        /* if last element was not created in the loop above */
        mkdirRes = nsFileMkdir(tempPath, mode, nsEuid, nsEgid, euid, egid);
        if (mkdirRes) {
            goto end;
        }
    }

    /* We ensure that we setup correct mode regarding umask settings */
    if (chmod(tempPath, mode)) {
        goto end;
    }

    res = MKDIR_STATUS_CREATED;

end:
    free(tempPath);
    return res;
}

// Sets base_dir of the full path to the library
// The full path takes a following format:
//  <base_dir>/<version>/<library_name>
// The <version> and <library_name> is set internally by this function
// E.g:
//  for /usr/lib/appscope/dev/libscope.so:
//    - <base_dir> - "/usr/lib/appscope"
//    - <version> - "dev"
//    - <library_name> - "libscope.so"
//  for /tmp/appscope/1.2.0/libscope.so:
//    - <base_dir> - "/tmp"
//    - <version> - "1.2.0"
//    - <library_name> - "libscope.so"
// Returns 0 if the full path to a library is accessible
int
libdirSetLibraryBase(const char *base) {
    const char *normVer = libverNormalizedVersion(g_libdir_info.ver);
    char tmp_path[PATH_MAX] = {0};


    int pathLen = snprintf(tmp_path, PATH_MAX, "%s/%s/%s", base, normVer, SCOPE_LIBSCOPE_SO);
    if (pathLen < 0) {
        fprintf(stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (pathLen >= PATH_MAX) {
        fprintf(stderr, "error: path too long.\n");
        return -1;
    }

    if (!access(tmp_path, R_OK)) {
        strncpy(libscopeState.binaryBasepath, base, PATH_MAX);
        return 0;
    }

    return -1;
}


/*
* Retrieve the full absolute path of the specified binary (ldscopedyn/libscope.so).
* Returns path for the specified binary, NULL in case of failure.
*/
const char *
libdirGetPath(libdirfile_t file) {
    const char *normVer = libverNormalizedVersion(g_libdir_info.ver);

    struct scope_obj_state *state = getObjState(file);
    if (!state) {
        return NULL;
    }

    if (state->binaryPath[0]) {
        return state->binaryPath;
    }

    if (state->binaryBasepath[0]) {
        // Check custom base first
        char tmp_path[PATH_MAX] = {0};
        int pathLen = snprintf(tmp_path, PATH_MAX, "%s/%s/%s", state->binaryBasepath, normVer, state->binaryName);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return NULL;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return NULL;
        }

        if (!access(tmp_path, R_OK)) {
            strncpy(state->binaryPath, tmp_path, PATH_MAX);
            return state->binaryPath;
        }
    }

    if (g_libdir_info.install_base[0]) {
        // Check install base next
        char tmp_path[PATH_MAX] = {0};
        int pathLen = snprintf(tmp_path, PATH_MAX, "%s/%s/%s", g_libdir_info.install_base, normVer, state->binaryName);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return NULL;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return NULL;
        }
        if (!access(tmp_path, R_OK)) {
            strncpy(state->binaryPath, tmp_path, PATH_MAX);
            return state->binaryPath;
        }
    }

    if (g_libdir_info.tmp_base[0]) {
        // Check temp base next
        char tmp_path[PATH_MAX] = {0};
        int pathLen = snprintf(tmp_path, PATH_MAX, "%s/%s/%s", g_libdir_info.tmp_base, normVer, state->binaryName);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return NULL;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return NULL;
        }
        if (!access(tmp_path, R_OK)) {
            strncpy(state->binaryPath, tmp_path, PATH_MAX);
            return state->binaryPath;
        }
    }


    return NULL;
}

/*
* Save libscope.so with specified permissions and ownership in specified path.
* Returns 0 if file was successfully created or if file already exists, -1 in case of failure.
*/
int
libdirSaveLibraryFile(const char *libraryPath, bool overwrite, mode_t mode, uid_t uid, gid_t gid) {
    return libdirCreateFileIfMissing(LIBRARY_FILE, libraryPath, overwrite, mode, uid, gid);
}

/*
* Extract (physically create) specified binary file to the filesystem.
* The extraction will not be performed:
* - if the file is present and it is official version
* - if the custom path was specified before by `libdirSetLibraryBase`
* Returns 0 in case of success, other values in case of failure.
*/
int libdirExtract(libdirfile_t file, uid_t uid, gid_t gid) {
    const char *normVer = libverNormalizedVersion(g_libdir_info.ver);
    bool isDevVersion = libverIsNormVersionDev(normVer);
    const char *existing_path = libdirGetPath(file);

    /*
    * If note match to existing path do not try to overwrite it
    */
    if ((existing_path) && (!libdirCheckNote(file, existing_path))) {
        return 0;
    }

    char dir[PATH_MAX] = {0};
    char path[PATH_MAX]= {0};
    int  pathLen;

    struct scope_obj_state *state = getObjState(file);
    if (!state) {
        return -1;
    }

    /*
    * Try to use the install base only for official version
    */
    mode_t mode = 0755;
    if (isDevVersion == FALSE) {
        pathLen = snprintf(dir, PATH_MAX, "%s/%s", g_libdir_info.install_base, normVer);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return -1;
        }

        if (libdirCreateDirIfMissing(dir, mode, uid, gid) <= MKDIR_STATUS_EXISTS) {
            int pathLen = snprintf(path, PATH_MAX, "%s/%s", dir, state->binaryName);
            if (pathLen < 0) {
                fprintf(stderr, "error: snprintf() failed.\n");
                return -1;
            }
            if (pathLen >= PATH_MAX) {
                fprintf(stderr, "error: path too long.\n");
                return -1;
            }
            if (!libdirCreateFileIfMissing(file, path, isDevVersion, mode, uid, gid)) {
                strncpy(state->binaryPath, path, PATH_MAX);
                strncpy(state->binaryBasepath, g_libdir_info.install_base, PATH_MAX);
                return 0;
            }
        }
    }
    // Clean the buffers
    memset(path, 0, PATH_MAX);
    memset(dir, 0, PATH_MAX);
    mode = 0777;
    pathLen = snprintf(dir, PATH_MAX, "%s/%s", g_libdir_info.tmp_base, normVer);
    if (pathLen < 0) {
        fprintf(stderr, "error: snprintf() failed.\n");
        return -1;
    }
    if (pathLen >= PATH_MAX) {
        fprintf(stderr, "error: path too long.\n");
        return -1;
    }
    if (libdirCreateDirIfMissing(dir, mode, uid, gid) <= MKDIR_STATUS_EXISTS) {
        int pathLen = snprintf(path, PATH_MAX, "%s/%s", dir, state->binaryName);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return -1;
        }
        if (!libdirCreateFileIfMissing(file, path, isDevVersion, mode, uid, gid)) {
            strncpy(state->binaryPath, path, PATH_MAX);
            strncpy(state->binaryBasepath, g_libdir_info.tmp_base, PATH_MAX);
            return 0;
        }
    }

    return -1;
}
