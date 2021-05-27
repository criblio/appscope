#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <elf.h>
#include <libgen.h>
#include <dirent.h>

#include "scopetypes.h"

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
 * 1) extract the ldscope dynamic exec from this object.
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
 * 7) execve the extracted ldscope passing args
 *    from this command line.
 */

#define EXE_TEST_FILE "/bin/cat"
#define DEFAULT_BIN_DIR "/tmp"
#define DEFAULT_BIN_FNAME "ldscopedyn"
#define LIBMUSL "musl"
#define ALWAYSEXTRACT 0
#define DEBUG 0

typedef struct libscope_info_t {
    char *path;
    int fd;
} libscope_info;

extern unsigned char _binary___bin_linux_ldscopedyn_start;
extern unsigned char _binary___bin_linux_ldscopedyn_end;

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
    if ((asprintf(&new_val, "%s:%s", cur_val, value) == -1)) {
        perror("setEnvVariable:asprintf");
        return;
    }

    if (g_debug) printf("%s:%d %s to %s\n", __FUNCTION__, __LINE__, env, new_val);
    if (setenv(env, new_val, 1)) {
        perror("setEnvVariable:setenv");
    }

    if (new_val) free(new_val);
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

    if ((fd = open(exe, O_RDONLY)) == -1) {
        perror("set_loader:open");
        return -1;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("set_loader:fstat");
        return -1;
    }

    buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        perror("set_loader:mmap");
        close(fd);
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
            char buf[PATH_MAX];

            if (strstr(exld, "ld-musl") != NULL) {
                close(fd);
                munmap(buf, sbuf.st_size);
                return 0;
            }

            snprintf(buf, sizeof(buf), "/lib/");
            if ((dirp = opendir(buf)) == NULL) {
                perror("set_loader:opendir");
                break;
            }

            while ((entry = readdir(dirp)) != NULL) {
                if ((entry->d_type != DT_DIR) &&
                    (strstr(entry->d_name, "ld-musl"))) {
                    strncat(buf, entry->d_name, strlen(entry->d_name));
                    name = 1;
                    break;
                }
            }

            closedir(dirp);

            if (name && (strlen(exld) > (strlen(buf) + 1))) {
                if (g_debug) printf("%s:%d exe ld.so: %s to %s\n", __FUNCTION__, __LINE__, exld, buf);
                strncpy(exld, buf, strlen(buf) + 1);
                found = 1;
                break;
            }
        }
    }

    if (found) {
        if (close(fd) == -1) {
            munmap(buf, sbuf.st_size);
            return -1;
        }

        if ((fd = open(exe, O_RDWR)) == -1) {
            perror("set_loader:open write");
            close(fd);
            munmap(buf, sbuf.st_size);
            return -1;
        }

        int rc = write(fd, buf, sbuf.st_size);
        if (rc < sbuf.st_size) {
            perror("set_loader:write");
        }
    } else {
        fprintf(stderr, "WARNING: can't locate or set the loader string in %s\n", exe);
    }

    close(fd);
    munmap(buf, sbuf.st_size);
    return (found - 1);
}

static char *
get_loader(char *exe)
{
    int i, fd;
    struct stat sbuf;
    char *buf, *ldso;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return NULL;

    if ((fd = open(exe, O_RDONLY)) == -1) {
        perror("get_loader:open");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("get_loader:fstat");
        return NULL;
    }

    buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
               PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        perror("get_loader:mmap");
        close(fd);
        return NULL;
    }

    close(fd);

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char * exld = (char *)&buf[phead[i].p_offset];
            if (g_debug) printf("%s:%d exe ld.so: %s\n", __FUNCTION__, __LINE__, exld);

            if ((ldso = calloc(1, strlen(exld) + 2)) != NULL) {
                strncpy(ldso, exld, strlen(exld));
            }

            break;
        }
    }

    munmap(buf, sbuf.st_size);
    return ldso;
}

static void
do_musl(char *exld, char *ldscope)
{
    char *lpath = NULL;
    char *ldso = NULL;
    char *path;
    char dir[strlen(ldscope) + 2];

    // alwyas set the env var
    strncpy(dir, ldscope, strlen(ldscope) + 1);
    path = dirname(dir);
    setEnvVariable(LD_LIB_ENV, path);

    // does a link to the musl ld.so exist?
    // if so, we assume the symlink exists as well.
    if ((ldso = get_loader(ldscope)) == NULL) return;

    if (asprintf(&lpath, "%s/%s", path, basename(ldso)) == -1) {
        perror("do_musl:asprintf");
        if (ldso) free(ldso);
        return;
    }

    // dir is expected to exist here, not creating one
    if ((symlink((const char *)exld, lpath) == -1) &&
        (errno != EEXIST)) {
        perror("do_musl:symlink");
        if (ldso) free(ldso);
        if (lpath) free(lpath);
        return;
    }

    set_loader(ldscope);

    if (ldso) free(ldso);
    if (lpath) free(lpath);
}

static void
release_bin(libscope_info *info) {
    if (!info) return;

    if (info->fd != -1) close(info->fd);
    if (info->path) free(info->path);
}

static int
extract_bin(libscope_info *info, unsigned char *start, unsigned char *end)
{
    if (!info || !start || !end || !info->path) return -1;

    struct stat sbuf;

    // if already extracted, don't do it again
    if (lstat(info->path, &sbuf) == 0) return 0;

    char *path;
    char dir[strlen(info->path)];

    strncpy(dir, info->path, strlen(info->path) + 1);
    path = dirname(dir);

    if (lstat(path, &sbuf) == -1) {
        if ((mkdir(path, S_IRWXU | S_IRWXG | S_IRWXO) == -1) &&
            (errno != EEXIST)) {
            perror("extract_bin:mkdir");
            return -1;
        }
    }

    info->fd = open(info->path, O_RDWR | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
    if (info->fd == -1) {
        perror("extract_bin:open");
        return -1;
    }

    size_t libsize = (size_t) (end - start);
    if (write(info->fd, start, libsize) != libsize) {
        perror("setup_libscope:write");
        return -1;
    }

    close(info->fd);

    return 1;
}

static int
setup_loader(char *exe, char *ldscope)
{
    char *ldso = NULL;

    if (((ldso = get_loader(exe)) != NULL) &&
        (strstr(ldso, LIBMUSL) != NULL)) {
            // we are using the musl ld.so
            do_musl(ldso, ldscope);
    }

    if (ldso) free(ldso);

    return 0;
}

#if ALWAYSEXTRACT > 0
static int
clean_extract(char *symlinkdir, char *verstr)
{
    char *unpath;
    DIR *dirp;
    struct dirent *entry;

    if (asprintf(&unpath, "%s/libscope-%s/",
                 (symlinkdir == NULL) ? DEFAULT_BIN_DIR : symlinkdir,
                 verstr) == -1) {
        perror("clean_extract:asprintf");
        if (unpath) free(unpath);
        return -1;
    }

    if ((dirp = opendir(unpath)) == NULL) {
        perror("clean_extract:opendir");
        if (unpath) free(unpath);
        return -1;
    }

    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type != DT_DIR) {
            char path[PATH_MAX];

            strncpy(path, unpath, strlen(unpath) + 1);
            strncat(path, entry->d_name, strlen(entry->d_name));

            if (unlink(path) == -1) {
                perror("clean_extract:unlink");
                printf("%s\n", entry->d_name);
                closedir(dirp);
                if (unpath) free(unpath);
                return -1;
            }
        }
    }

    unlinkat(AT_FDCWD, unpath, AT_REMOVEDIR);
    closedir(dirp);
    if (unpath) free(unpath);
    return 0;
}
#endif // ALWAYSEXTRACT

static void
usage(char *prog) {
  fprintf(stderr,"usage: %s [-h help] [-f sym link dir] executable\n", prog);
  exit(-1);
}

int
main(int argc, char **argv, char **env)
{
    int i, j, rc, optind = 1;
    libscope_info info;
    char *verstr;
    char *symlinkdir = NULL;
    char scopever[64];

    if (argc < 2) {
        usage(argv[0]);
    }

    if ((argc >= 3) && (strncmp(argv[1], "-f", 2) == 0)) {
        symlinkdir=strdup(argv[2]);
        optind = 3;
    }

#if DEBUG > 0
    g_debug = 1;
#endif

    if (g_debug) printf("Starting scope static...first extract\nsym link dir %s\noption index %d\n",
                        (symlinkdir == NULL) ? "/tmp" : symlinkdir, optind);

    strncpy(scopever, SCOPE_VER, strlen(SCOPE_VER) + 1);
    verstr = strtok(scopever, "-");
    if (asprintf(&info.path, "%s/libscope-%s/%s",
                 (symlinkdir == NULL) ? DEFAULT_BIN_DIR : symlinkdir,
                 verstr, DEFAULT_BIN_FNAME) == -1) {
        perror("ldscope:path");
        exit(EXIT_FAILURE);
    }

#if ALWAYSEXTRACT > 0
    clean_extract(symlinkdir, verstr);
#endif

    if ((rc = extract_bin(&info,
                          &_binary___bin_linux_ldscopedyn_start,
                          &_binary___bin_linux_ldscopedyn_end)) == -1) {
        release_bin(&info);
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (g_debug) printf("path to ldscope: %s %s\n", info.path, argv[optind]);

    // are we on glibc or musl?
    setup_loader(EXE_TEST_FILE, info.path);

    argv[0] = info.path;
    argv[1] = argv[optind];

    for (i = optind + 1, j = 2; i < argc; i++, j++) {
        argv[j] = argv[i];
    }

    argv[j] = NULL;

    execve(info.path, argv, environ);
    perror("execve");
    release_bin(&info);
    exit(EXIT_FAILURE);
}
