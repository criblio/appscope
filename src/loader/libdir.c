#define _XOPEN_SOURCE 500 // for FTW
#define _GNU_SOURCE

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <linux/limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stddef.h>
#include <unistd.h>

#include "libdir.h"
#include "libver.h"
#include "loaderutils.h"
#include "loader.h"
#include "nsfile.h"
#include "patch.h"
#include "scopetypes.h"

#ifndef SCOPE_VER
#error "Missing SCOPE_VER"
#endif

#ifndef SCOPE_LIBSCOPE_SO
#define SCOPE_LIBSCOPE_SO "libscope.so"
#endif

#ifndef SCOPE_DYN_NAME
#define SCOPE_DYN_NAME "scopedyn"
#endif

#define SCOPE_NAME_SIZE (16)
#define LIBSCOPE "github.com/criblio/scope/run._buildLibscopeSo"
#define SCOPEDYN "github.com/criblio/scope/run._buildScopedyn"

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

// internal state for library
static struct scope_obj_state libscopeState = {
    .binaryName = SCOPE_LIBSCOPE_SO,
};

// internal state for loader
static struct scope_obj_state scopedynState = {
    .binaryName = SCOPE_DYN_NAME,
};

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
        case LIBRARY_FILE:
            return &libscopeState;
            break;
        case LOADER_FILE:
            return &scopedynState;
            break;
    }
    // unreachable
    return NULL;
}

size_t
getAsset(libdirfile_t objFileType, unsigned char **start)
{
    if (!start) return -1;

    size_t len = -1;
    uint64_t *libsym;
    unsigned char *libptr;
    elf_buf_t *ebuf = NULL;

    if ((ebuf = getElf("/proc/self/exe")) == NULL) {
        fprintf(stderr, "error: can't read /proc/self/exe\n");
        goto out;
    }

    switch (objFileType) {
        case LIBRARY_FILE:
            if ((libsym = getSymbol(ebuf->buf, LIBSCOPE))) {
                libptr = (unsigned char *)*libsym;
            } else {
                fprintf(stderr, "%s:%d no addr for _buildLibscopeSo\n", __FUNCTION__, __LINE__);
                goto out;
            }

            *start = (unsigned char *)libptr;
            len =  g_libscopesz;
            break;

        case LOADER_FILE:
            if ((libsym = getSymbol(ebuf->buf, SCOPEDYN))) {
                libptr = (unsigned char *)*libsym;
            } else {
                fprintf(stderr, "%s:%d no addr for _buildScopedyn\n", __FUNCTION__, __LINE__);
                goto out;
            }

            *start = (unsigned char *)libptr;
            len =  g_scopedynsz;
            break;

        default:
            break;
    }

  out:
    if (ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
    }

    return len;
}

int
libdirCreateSymLinkIfMissing(char *path, char *target, bool overwrite, mode_t mode, uid_t nsUid, gid_t nsGid)
{
    int ret;

    // Check if file exists
    if (!access(path, R_OK) && !overwrite) {
        return 0; // File exists
    }

    uid_t currentEuid = geteuid();
    gid_t currentEgid = getegid();
    
    ret = nsFileSymlink(target, path, nsUid, nsGid, currentEuid, currentEgid);
    if (ret) { 
        fprintf(stderr, "libdirCreateSymLinkIfMissing: symlink %s failed\n", path);
    }

    return ret;
}

/*
 * Getting objects bundled
 */
int
libdirCreateFileIfMissing(unsigned char *file, size_t file_len, const char *path, bool overwrite, mode_t mode, uid_t nsEuid, gid_t nsEgid) {
    // Check if file exists
    if (!access(path, R_OK) && !overwrite) {
        return 0; // File exists
    }

    int fd;
    char temp[PATH_MAX];
    unsigned char *start;
    size_t len;

    if (file) {
        start = file;
        len = file_len;
    } else {
        if ((len = getAsset(LIBRARY_FILE, &start)) == -1) {
            return -1;
        }
    }

    // Write file
    int tempLen = snprintf(temp, PATH_MAX, "%s.XXXXXX", path);
    if (tempLen < 0) {
        fprintf(stderr, "error: snprintf failed.\n");
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
* Retrieve the full absolute path of the specified binary libscope.so.
* Returns path for the specified binary, NULL in case of failure.
*/
const char *
libdirGetPath(void) {
    const char *normVer = libverNormalizedVersion(g_libdir_info.ver);

    struct scope_obj_state *state = getObjState(LIBRARY_FILE);
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

    const char *cribl_home = getenv("CRIBL_HOME");
    if (cribl_home) {
        char tmp_path[PATH_MAX] = {0};
        int pathLen = snprintf(tmp_path, PATH_MAX, "%s/appscope/%s/%s", cribl_home, normVer, state->binaryName);
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
        // Check tmp base next
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
* Extract (physically create) libscope.so to the filesystem.
* The extraction will not be performed:
* - if the file is present and it is official version
* - if the custom path was specified before by `libdirSetLibraryBase`
* Returns 0 in case of success, other values in case of failure.
*/
int libdirExtract(unsigned char *file, size_t file_len, uid_t nsUid, gid_t nsGid) {
    char path[PATH_MAX] = {0};
    char path_musl[PATH_MAX] = {0};
    char path_glibc[PATH_MAX] = {0};
    size_t pathlen = 0;
    char *target;
    mode_t mode = 0755;
    mkdir_status_t res;

    // Which version of AppScope are we dealing with (official or dev)
    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);
    bool overwrite = isDevVersion;

    // Create the destination directory if it does not exist
    
    // Try to create $CRIBL_HOME/appscope (if set)
    const char *cribl_home = getenv("CRIBL_HOME");
    if (cribl_home) {
        int pathLen = snprintf(path, PATH_MAX, "/%s/appscope/%s/", cribl_home, loaderVersion);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return -1;
        }
        res = libdirCreateDirIfMissing(path, mode, nsUid, nsGid);
    }

    // If CRIBL_HOME not defined, or there was an error, create usr/lib/appscope
    if (!cribl_home || res > MKDIR_STATUS_EXISTS) {
        memset(path, 0, PATH_MAX);
        int pathLen = snprintf(path, PATH_MAX, "/usr/lib/appscope/%s/", loaderVersion);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return -1;
        }
        res = libdirCreateDirIfMissing(path, mode, nsUid, nsGid);
    }

    // If all else fails, create /tmp/appscope
    if (res > MKDIR_STATUS_EXISTS) {
        mode = 0777;
        memset(path, 0, PATH_MAX);
        int pathLen = snprintf(path, PATH_MAX, "/tmp/appscope/%s/", loaderVersion);
        if (pathLen < 0) {
            fprintf(stderr, "error: snprintf() failed.\n");
            return -1;
        }
        if (pathLen >= PATH_MAX) {
            fprintf(stderr, "error: path too long.\n");
            return -1;
        }
        res = libdirCreateDirIfMissing(path, mode, nsUid, nsGid);
    }

    if (res > MKDIR_STATUS_EXISTS) {
        fprintf(stderr, "setupInstall: libdirCreateDirIfMissing failed\n");
        return -1;
    }

    // Create the libscope file if it does not exist; or needs to be overwritten
    
    pathlen = strlen(path);
    // Extract libscope.so.glibc (bundled libscope defaults to glibc loader)
    strncpy(path_glibc, path, pathlen);
    strncat(path_glibc, "libscope.so.glibc", sizeof(path_glibc) - 1);
    if (libdirCreateFileIfMissing(file, file_len, path_glibc, overwrite, mode, nsUid, nsGid)) {
        fprintf(stderr, "setupInstall: saving %s failed\n", path_glibc);
        return -1;
    }

    // Extract libscope.so.musl
    strncpy(path_musl, path, pathlen);
    strncat(path_musl, "libscope.so.musl", sizeof(path_musl) - 1);
    if (libdirCreateFileIfMissing(file, file_len, path_musl, overwrite, mode, nsUid, nsGid)) {
        fprintf(stderr, "setupInstall: saving %s failed\n", path);
        return -1;
    }

    // Patch the libscope.so.musl file for musl
    patch_status_t patch_res;
    if ((patch_res = patchLibrary(path_musl, TRUE)) == PATCH_FAILED) {
        fprintf(stderr, "setupInstall: patch %s failed\n", path_musl);
        return -1;
    }

    // Create symlink to appropriate version
    strncat(path, "libscope.so", sizeof(path) - 1);
    target = isMusl() ? path_musl : path_glibc;
    if (libdirCreateSymLinkIfMissing(path, target, overwrite, mode, nsUid, nsGid)) {
        fprintf(stderr, "setupInstall: symlink %s failed\n", path);
        return -1;
    }

    return 0;
}
