#define _GNU_SOURCE

#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include "libdir.h"
#include "loaderop.h"
#include "nsfile.h"

#include "scopestdlib.h"

#define EXE_TEST_FILE "/bin/cat"
#define LIBMUSL "musl"

static int g_debug = 0;

static void
setEnvVariable(char *env, char *value)
{
    char *cur_val = getenv(env);

    // If env is not set
    if (!cur_val) {
        if (setenv(env, value, 1)) {
            perror("setEnvVariable:setenv");
        }
        return;
    }

    // env is set. try to append
    char *new_val = NULL;
    if ((scope_asprintf(&new_val, "%s:%s", cur_val, value) == -1)) {
        scope_perror("setEnvVariable:asprintf");
        return;
    }

    if (g_debug) scope_printf("%s:%d %s to %s\n", __FUNCTION__, __LINE__, env, new_val);
    if (setenv(env, new_val, 1)) {
        perror("setEnvVariable:setenv");
    }

    if (new_val) scope_free(new_val);
}

/*
 * This code exists solely to support the ability to
 * build a libscope and an ldscope on a glibc distro
 * that will execute on both a glibc distro and a
 * musl distro.
 *
 * This code is used to create a static exec that will
 * execute on both glibc and musl distros.
 *
 * The process:
 * 1) extract the ldscopedyn dynamic exec and libscope.so
 *    dynamic lib from this object.
 * 2) open an executable file on the current FS and
 *    read the loader string from the .interp section.
 * 3) if it uses a musl ld.so then do musl
 * 4) for musl; create a dir and in that dir create a
 *    soft link to ld-musl.so from ld.linux.so (the
 *    glibc loader).
 * 5) for musl; create or add to the ld lib path
 *    env var to point to the dir created above.
 * 6) for musl; modify the loader string in .interp
 *    of ldscope to ld-musl.so.
 * 7) execve the extracted ldscopedyn passing args
 *    from this command line.
 */

static int
get_dir(const char *path, char *fres, size_t len) {
    DIR *dirp;
    struct dirent *entry;
    char *dcopy, *pcopy, *dname, *fname;
    int res = -1;

    if (!path || !fres || (len <= 0)) return res;

    pcopy = scope_strdup(path);
    dname = scope_dirname(pcopy);

    if ((dirp = scope_opendir(dname)) == NULL) {
        scope_perror("get_dir:opendir");
        if (pcopy) scope_free(pcopy);
        return res;
    }

    dcopy = scope_strdup(path);
    fname = scope_basename(dcopy);

    while ((entry = scope_readdir(dirp)) != NULL) {
        if ((entry->d_type != DT_DIR) &&
            (scope_strstr(entry->d_name, fname))) {
            scope_strncpy(fres, entry->d_name, len);
            res = 0;
            break;
        }
    }

    scope_closedir(dirp);
    if (pcopy) scope_free(pcopy);
    if (dcopy) scope_free(dcopy);
    return res;
}

static char *
loaderOpGetLoader(const char *exe) {
    int i, fd;
    struct stat sbuf;
    char *buf, *ldso = NULL;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return NULL;

    if ((fd = scope_open(exe, O_RDONLY)) == -1) {
        scope_perror("loaderOpGetLoader:open");
        return NULL;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("loaderOpGetLoader:fstat");
        scope_close(fd);
        return NULL;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("loaderOpGetLoader:scope_mmap");
        scope_close(fd);
        return NULL;
    }

    scope_close(fd);

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char * exld = (char *)&buf[phead[i].p_offset];
            if (g_debug) scope_printf("%s:%d exe ld.so: %s\n", __FUNCTION__, __LINE__, exld);

            ldso = scope_strdup(exld);

            break;
        }
    }

    scope_munmap(buf, sbuf.st_size);
    return ldso;
}

// modify NEEDED entries in libscope.so to avoid dependencies
static int
loaderOpSetLibrary(const char *libpath) {
    int i, fd, found, name;
    struct stat sbuf;
    char *buf;
    Elf64_Ehdr *elf;
    Elf64_Shdr *sections;
    Elf64_Dyn *dyn;
    const char *section_strtab = NULL;
    const char *strtab = NULL;
    const char *sec_name = NULL;

    if (libpath == NULL)
        return -1;

    if ((fd = scope_open(libpath, O_RDONLY)) == -1) {
        scope_perror("loaderOpSetLibrary:open");
        return -1;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("loaderOpSetLibrary:fstat");
        scope_close(fd);
        return -1;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("loaderOpSetLibrary:scope_mmap");
        scope_close(fd);
        return -1;
    }

    // get the elf header, section table and string table
    elf = (Elf64_Ehdr *)buf;

    if (elf->e_ident[EI_MAG0] != ELFMAG0
        || elf->e_ident[EI_MAG1] != ELFMAG1
        || elf->e_ident[EI_MAG2] != ELFMAG2
        || elf->e_ident[EI_MAG3] != ELFMAG3
        || elf->e_ident[EI_VERSION] != EV_CURRENT) {
        scope_fprintf(scope_stderr, "ERROR:%s: is not valid ELF file", libpath);
        scope_close(fd);
        scope_munmap(buf, sbuf.st_size);
        return -1;
    }

    sections = (Elf64_Shdr *)((char *)buf + elf->e_shoff);
    section_strtab = (char *)buf + sections[elf->e_shstrndx].sh_offset;
    found = name = 0;

    // locate the .dynstr section
    for (i = 0; i < elf->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        if (sections[i].sh_type == SHT_STRTAB && scope_strcmp(sec_name, ".dynstr") == 0) {
            strtab = (const char *)(buf + sections[i].sh_offset);
        }
    }

    if (strtab == NULL) {
        scope_fprintf(scope_stderr, "ERROR:%s: did not locate the .dynstr from %s", __FUNCTION__, libpath);
        scope_close(fd);
        scope_munmap(buf, sbuf.st_size);
        return -1;
    }

    // locate the .dynamic section
    for (i = 0; i < elf->e_shnum; i++) {
        if (sections[i].sh_type == SHT_DYNAMIC) {
            for (dyn = (Elf64_Dyn *)((char *)buf + sections[i].sh_offset); dyn != NULL && dyn->d_tag != DT_NULL; dyn++) {
                if (dyn->d_tag == DT_NEEDED) {
                    char *depstr = (char *)(strtab + dyn->d_un.d_val);
                    if (depstr && scope_strstr(depstr, "ld-linux")) {
                        char newdep[PATH_MAX];
                        size_t newdep_len;
                        if (get_dir("/lib/ld-musl", newdep, sizeof(newdep)) == -1) break;
                        newdep_len = scope_strlen(newdep);
                        if (scope_strlen(depstr) >= newdep_len) {
                            scope_strncpy(depstr, newdep, newdep_len + 1);
                            found = 1;
                            break;
                        }
                    }
                }
            }
        }
        if (found == 1) break;
    }

    if (found) {
        if (scope_close(fd) == -1) {
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        if ((fd = scope_open(libpath, O_RDWR)) == -1) {
            scope_perror("loaderOpSetLibrary:open write");
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        int rc = scope_write(fd, buf, sbuf.st_size);
        if (rc < sbuf.st_size) {
            scope_perror("loaderOpSetLibrary:write");
        }
    }

    scope_close(fd);
    scope_munmap(buf, sbuf.st_size);
    return (found - 1);
}

patch_status_t
loaderOpPatchLibrary(const char *so_path) {
    patch_status_t patch_res = PATCH_NO_OP;

    char *ldso = loaderOpGetLoader(EXE_TEST_FILE);
    if (ldso && scope_strstr(ldso, LIBMUSL) != NULL) {
        if (!loaderOpSetLibrary(so_path)) {
            patch_res = PATCH_SUCCESS;
        } else {
            patch_res = PATCH_FAILED;
        }
    }

    scope_free(ldso);

    return patch_res;
}

// modify the loader string in the .interp section of ldscope
static int
set_loader(char *exe)
{
    int i, fd, found, name;
    struct stat sbuf;
    char *buf;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return -1;

    if ((fd = scope_open(exe, O_RDONLY)) == -1) {
        scope_perror("set_loader:open");
        return -1;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("set_loader:fstat");
        scope_close(fd);
        return -1;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("set_loader:scope_mmap");
        scope_close(fd);
        return -1;
    }

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];
    found = name = 0;

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char *exld = (char *)&buf[phead[i].p_offset];
            DIR *dirp;
            struct dirent *entry;
            char dir[PATH_MAX];
            size_t dir_len;

            if (scope_strstr(exld, "ld-musl") != NULL) {
                scope_close(fd);
                scope_munmap(buf, sbuf.st_size);
                return 0;
            }

            scope_snprintf(dir, sizeof(dir), "/lib/");
            if ((dirp = scope_opendir(dir)) == NULL) {
                scope_perror("set_loader:opendir");
                break;
            }

            while ((entry = scope_readdir(dirp)) != NULL) {
                if ((entry->d_type != DT_DIR) &&
                    (scope_strstr(entry->d_name, "ld-musl"))) {
                    scope_strncat(dir, entry->d_name, scope_strlen(entry->d_name) + 1);
                    name = 1;
                    break;
                }
            }

            scope_closedir(dirp);
            dir_len = scope_strlen(dir);
            if (name && (scope_strlen(exld) >= dir_len)) {
                if (g_debug) scope_printf("%s:%d exe ld.so: %s to %s\n", __FUNCTION__, __LINE__, exld, dir);
                scope_strncpy(exld, dir, dir_len + 1);
                found = 1;
                break;
            }
        }
    }

    if (found) {
        if (scope_close(fd) == -1) {
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        if ((fd = scope_open(exe, O_RDWR)) == -1) {
            scope_perror("set_loader:open write");
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        int rc = scope_write(fd, buf, sbuf.st_size);
        if (rc < sbuf.st_size) {
            scope_perror("set_loader:write");
        }
    } else {
        scope_fprintf(scope_stderr, "WARNING: can't locate or set the loader string in %s\n", exe);
    }

    scope_close(fd);
    scope_munmap(buf, sbuf.st_size);
    return (found - 1);
}

static void
do_musl(char *exld, char *ldscope, uid_t nsUid, gid_t nsGid)
{
    int symlinkErr = 0;
    char *lpath = NULL;
    char *ldso = NULL;
    char *path;
    char dir[scope_strlen(ldscope) + 2];

    // always set the env var
    scope_strncpy(dir, ldscope, scope_strlen(ldscope) + 1);
    path = scope_dirname(dir);
    setEnvVariable(LD_LIB_ENV, path);

    // does a link to the musl ld.so exist?
    // if so, we assume the symlink exists as well.
    if ((ldso = loaderOpGetLoader(ldscope)) == NULL) return;

    // Avoid creating ld-musl-x86_64.so.1 -> /lib/ld-musl-x86_64.so.1
    if (scope_strstr(ldso, "musl")) return;

    if (scope_asprintf(&lpath, "%s/%s", path, scope_basename(ldso)) == -1) {
        scope_perror("do_musl:asprintf");
        if (ldso) scope_free(ldso);
        return;
    }

    // dir is expected to exist here, not creating one
    if ((nsFileSymlink((const char *)exld, lpath, nsUid, nsGid, scope_geteuid(), scope_getegid(), &symlinkErr) == -1) &&
        (symlinkErr != EEXIST)) {
        scope_perror("do_musl:symlink");
        if (ldso) scope_free(ldso);
        if (lpath) scope_free(lpath);
        return;
    }

    set_loader(ldscope);
    loaderOpSetLibrary(libdirGetPath(LIBRARY_FILE));

    if (ldso) scope_free(ldso);
    if (lpath) scope_free(lpath);
}


/*
 * Check for and handle the extra steps needed to run under musl libc.
 *
 * Returns 0 if musl was not detected and 1 if it was.
 */
int
loaderOpSetupLoader(char *ldscope, uid_t nsUid, gid_t nsGid)
{
    int ret = 0; // not musl

    char *ldso = NULL;

    if (((ldso = loaderOpGetLoader(EXE_TEST_FILE)) != NULL) &&
        (scope_strstr(ldso, LIBMUSL) != NULL)) {
            // we are using the musl ld.so
            do_musl(ldso, ldscope, nsUid, nsGid);
            ret = 1; // detected musl
    }

    if (ldso) scope_free(ldso);

    return ret;
}
