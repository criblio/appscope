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

#include "scopetypes.h"

#define EXE_TEST_FILE "/bin/cat"
#define DEFAULT_BIN_DIR "/tmp"
#define DEFAULT_BIN_FNAME "ldscopedyn"
#define LIBMUSL "musl"
#define LD_LIB_ENV "LD_LIBRARY_PATH"
#define ALWAYSEXTRACT 1

typedef struct libscope_info_t {
    char *path;
    char *shm_name;
    int fd;
    int use_memfd;
} libscope_info;

extern unsigned char _binary___bin_linux_ldscopedyn_start;
extern unsigned char _binary___bin_linux_ldscopedyn_end;

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

    printf("%s:%d %s to %s\n", __FUNCTION__, __LINE__, env, new_val);
    if (setenv(env, new_val, 1)) {
        perror("setEnvVariable:setenv");
    }

    if (new_val) free(new_val);
}

static void
setLdsoEnv(char *ldscope)
{
    char *path;
    char dir[strlen(ldscope)];

    strncpy(dir, ldscope, strlen(ldscope) + 1);
    path = dirname(dir);

    setEnvVariable(LD_LIB_ENV, path);
}

static char *
set_loader(char *exe)
{
    int i, fd;
    struct stat sbuf;
    char *buf;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return NULL;

    if ((fd = open(exe, O_RDWR)) == -1) {
        perror("set_loader:open");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("set_loader:fstat");
        return NULL;
    }

    buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        perror("set_loader:mmap");
        close(fd);
        return NULL;
    }

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char *exld = (char *)&buf[phead[i].p_vaddr];
            printf("%s:%d exe ld.so: %s\n", __FUNCTION__, __LINE__, exld);
            strncpy(exld, "/lib/ld-musl-x86_64.so.1", strlen("/lib/ld-musl-x86_64.so.1") + 1);
            break;
        }
    }

    int rc = write(fd, buf, sbuf.st_size);
    if (rc < sbuf.st_size) {
        perror("set_loader:write");
    }

    close(fd);
    munmap(buf, sbuf.st_size);
    return NULL;
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
            char *exld = (char *)&buf[phead[i].p_vaddr];
            printf("%s:%d exe ld.so: %s\n", __FUNCTION__, __LINE__, exld);

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
    char dir[strlen(ldscope)];

    // does a link to the musl ld.so exist?
    if ((ldso = get_loader(ldscope)) == NULL) return;

    strncpy(dir, ldscope, strlen(ldscope) + 1);
    path = dirname(dir);

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

    setEnvVariable(LD_LIB_ENV, path);

    if (ldso) free(ldso);
    if (lpath) free(lpath);
}

static void
release_bin(libscope_info *info) {
    if (!info) return;

    if (info->fd != -1) close(info->fd);
    if (info->shm_name) {
        if (info->fd != -1) close(info->fd);
        free(info->shm_name);
    }
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

    info->shm_name = NULL;
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

int
main(int argc, char **argv, char **env)
{
    int rc;
    libscope_info info;
    char *verstr;
    char scopever[64];

    printf("Starting scope static...first extract\n");

    strncpy(scopever, SCOPE_VER, strlen(SCOPE_VER) + 1);
    verstr = strtok(scopever, "-");
    if (asprintf(&info.path, "%s/libscope-%s/%s",
                 DEFAULT_BIN_DIR, verstr, DEFAULT_BIN_FNAME) == -1) {
        perror("ldscope:path");
        exit(EXIT_FAILURE);
    }

    if ((rc = extract_bin(&info,
                          &_binary___bin_linux_ldscopedyn_start,
                          &_binary___bin_linux_ldscopedyn_end)) == -1) {
        release_bin(&info);
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

#if ALWAYSEXTRACT == 1
    rc = 1;
#endif

    if (rc == 1) {
        // new extraction of the bin
        printf("path to ldscope: %s\n", info.path);
        setup_loader(EXE_TEST_FILE, info.path);
    } else {
        setLdsoEnv(info.path);
    }

    execve(info.path, argv, environ);
    perror("execve");
    release_bin(&info);
    exit(EXIT_FAILURE);
}
