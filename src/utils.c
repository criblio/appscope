#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/utsname.h>
#include <sys/mman.h>
#include <elf.h>
#include <libgen.h>

#include "utils.h"
#include "fn.h"
#include "dbg.h"
#include "runtimecfg.h"

#define LIBMUSL "musl"
#define LD_LIB_ENV "LD_LIBRARY_PATH"

rtconfig g_cfg = {0};

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

    //setEnvVariable(LD_LIB_ENV, path);
    if (setenv(LD_LIB_ENV, path, 1) == -1) {
        perror("do_musl:setenv");
    }

    if (ldso) free(ldso);
    if (lpath) free(lpath);
}

unsigned int
strToVal(enum_map_t map[], const char *str)
{
    enum_map_t *m;
    for (m=map; m->str; m++) {
        if (!strcmp(str, m->str)) return m->val;
    }
    return -1;
}

const char *
valToStr(enum_map_t map[], unsigned int val)
{
    enum_map_t *m;
    for (m=map; m->str; m++) {
        if (val == m->val) return m->str;
    }
    return NULL;
}

int
checkEnv(char *env, char *val)
{
    char *estr;
    if (((estr = getenv(env)) != NULL) &&
       (strncmp(estr, val, strlen(estr)) == 0)) {
        return TRUE;
    }
    return FALSE;
}

void
setPidEnv(int pid)
{
    char val[32];
    int returnval = snprintf(val, sizeof(val), "%d", pid);
    if (returnval >= sizeof(val) || returnval == -1) {
        DBG("returnval = %d", returnval);
        return;
    }

    if (!g_fn.setenv || g_fn.setenv(SCOPE_PID_ENV, val, 1) == -1) {
        DBG("g_fn.setenv=%p, SCOPE_PID_ENV=%s, val=%s",
             g_fn.setenv, SCOPE_PID_ENV, val);
    }
}

#ifdef __MACOS__
char *
getpath(const char *cmd)
{
    return NULL;
}

#else

// This tests to see if cmd can be resolved to an executable file.
// If so it will return a malloc'd buffer containing an absolute path,
// otherwise it will return NULL.
char *
getpath(const char *cmd)
{
    char *path_env = NULL;
    char *ret_val = NULL;
    struct stat buf;

    if (!cmd) goto out;

    // an absolute path was specified for cmd.
    if (cmd[0] == '/') {
        //  If we can resolve it, use it.
        if (!stat(cmd, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = strdup(cmd);
        }
        goto out;
    }

    // a relative path was specified for cmd.
    if (strchr(cmd, '/')) {
        char *cur_dir = get_current_dir_name();
        if (!cur_dir) goto out;

        char *path = NULL;
        if (asprintf(&path, "%s/%s", cur_dir, cmd) > 0) {
            // If we can resolve it, use it
            if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
                ret_val = path;
            } else {
                free(path);
            }
        }
        free(cur_dir);
        goto out;
    }

    // try the current dir
    char *path = NULL;
    if (asprintf(&path, "./%s", cmd) > 0) {
        if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = path;
            goto out;
        } else {
            free(path);
        }
    }

    // try to resolve the cmd from PATH env variable
    char *path_env_ptr = getenv("PATH");
    if (!path_env_ptr) goto out;
    path_env = strdup(path_env_ptr); // create a copy for strtok below
    if (!path_env) goto out;

    char *saveptr = NULL;
    char *strtok_path = strtok_r(path_env, ":", &saveptr);
    if (!strtok_path) goto out;

    do {
        char *path = NULL;
        if (asprintf(&path, "%s/%s", strtok_path, cmd) < 0) {
            break;
        }
        if ((stat(path, &buf) == -1) ||    // path doesn't exist
            (!S_ISREG(buf.st_mode)) ||     // path isn't a file
            ((buf.st_mode & 0111) == 0)) { // path isn't executable

            free(path);
            continue;
        }

        // We found the cmd, and it's an executable file
        ret_val = path;
        break;

    } while ((strtok_path = strtok_r(NULL, ":", &saveptr)));

out:
    if (path_env) free(path_env);
    return ret_val;
}
#endif //__MACOS__


int
startsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    return (strncmp(string, substring, strlen(substring)) == 0);
}

int
endsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    int stringlen = strlen(string);
    int sublen = strlen(substring);
    return (sublen <= stringlen) &&
       ((strncmp(&string[stringlen-sublen], substring, sublen)) == 0);
}

int
sigSafeNanosleep(const struct timespec *req)
{
    if (!g_fn.nanosleep) {
        DBG(NULL);
        return -1;
    }

    struct timespec time = *req;
    int rv;

    // If we're interrupted, sleep again for whatever time remains
    do {
        rv = g_fn.nanosleep(&time, &time);
    } while (rv && (errno == EINTR));

    return rv;
}

void
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
    if (!strstr(cur_val, value)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s", cur_val, value) == -1)) {
            perror("setEnvVariable:asprintf");
            return;
        }

        if (setenv(env, new_val, 1)) {
            perror("setEnvVariable:setenv");
        }

        if (new_val) free(new_val);
    }
}

void
setLdsoEnv(char *ldscope)
{
    char *path;
    char dir[strlen(ldscope)];

    strncpy(dir, ldscope, strlen(ldscope) + 1);
    path = dirname(dir);

    //setEnvVariable(LD_LIB_ENV, path);
    if (setenv(LD_LIB_ENV, path, 1) == -1) {
        perror("setLdsoEnv:setenv");
    }
}

void
release_bin(libscope_info *info) {
    if (!info) return;

    if (info->fd != -1) close(info->fd);
    if (info->shm_name) {
        if (info->fd != -1) close(info->fd);
        free(info->shm_name);
    }
    if (info->path) free(info->path);
}

int
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

int
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
