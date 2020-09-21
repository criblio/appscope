/*
 * Load and run static executables
 *
 * objcopy -I binary -O elf64-x86-64 -B i386 ./lib/linux/libscope.so ./lib/linux/libscope.o
 * gcc -Wall -g src/scope.c -ldl -lrt -o scope ./lib/linux/libscope.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include "fn.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include <limits.h>
#include <errno.h>

#define DEVMODE 0
#define __NR_memfd_create   319
#define _MFD_CLOEXEC		0x0001U
#define SHM_NAME            "libscope"
#define PARENT_PROC_NAME "start_scope"
#define GO_ENV_VAR "GODEBUG"
#define GO_ENV_VALUE "http2server"

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

typedef struct {
    char *path;
    char *shm_name;
    int fd;
    int use_memfd;
} libscope_info_t;

// Wrapper to call memfd_create syscall
static inline int _memfd_create(const char *name, unsigned int flags) {
	return syscall(__NR_memfd_create, name, flags);
}

static void
usage(char *prog) {
  fprintf(stderr,"usage: %s static-executable args-for-executable\n", prog);
  exit(-1);
}

static int
is_static(char *buf)
{
    int i;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_DYNAMIC) || (phead[i].p_type == PT_INTERP)) {
            return 0;
        }
    }

    return 1;
}

static int
app_type(char *buf, const uint32_t sh_type, const char *sh_name)
{
    int i = 0;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        //printf("section %s type = %d \n", sec_name, sections[i].sh_type);
        if (sections[i].sh_type == sh_type && strcmp(sec_name, sh_name) == 0) {
            return 1;
        }
    }
    return 0;
}

/**
 * Checks if kernel version is >= 3.17
 */
static int
check_kernel_version(void)
{
    struct utsname buffer;
    char *token;
    char *separator = ".";
    int val;

    if (uname(&buffer)) {
        return 0;
    }
    token = strtok(buffer.release, separator);
    val = atoi(token);
    if (val < 3) {
        return 0;
    } else if (val > 3){
        return 1;
    }

    token = strtok(NULL, separator);
    val = atoi(token);
    return (val < 17) ? 0 : 1;
}

static void
release_libscope(libscope_info_t **info_ptr) {
    if (!info_ptr || !*info_ptr) return;

    libscope_info_t *info = *info_ptr;

    if (info->fd != -1) close(info->fd);
    if (info->shm_name) {
        if (info->fd != -1) shm_unlink(info->shm_name);
        free(info->shm_name);
    }
    if (info->path) free(info->path);
    free(info);
    *info_ptr = NULL;
}

static libscope_info_t *
setup_libscope()
{
    libscope_info_t *info = NULL;
    int everything_successful = FALSE;

    if (!(info = calloc(1, sizeof(libscope_info_t)))) {
        perror("setup_libscope:calloc");
        goto err;
    }

    info->fd = -1;
    info->use_memfd = check_kernel_version();
    
    if (info->use_memfd) {
        info->fd = _memfd_create(SHM_NAME, _MFD_CLOEXEC);
    } else {
        if (asprintf(&info->shm_name, "%s%i", SHM_NAME, getpid()) == -1) {
            perror("setup_libscope:shm_name");
            info->shm_name = NULL; // failure leaves info->shm_name undefined
            goto err;
        }
        info->fd = shm_open(info->shm_name, O_RDWR | O_CREAT, S_IRWXU);
    }
    if (info->fd == -1) {
        perror(info->use_memfd ? "setup_libscope:memfd_create" : "setup_libscope:shm_open");
        goto err;
    }
    
    size_t libsize = (size_t) (&_binary___lib_linux_libscope_so_end - &_binary___lib_linux_libscope_so_start);
    if (write(info->fd, &_binary___lib_linux_libscope_so_start, libsize) != libsize) {
        perror("setup_libscope:write");
        goto err;
    }

    int rv;
    if (info->use_memfd) {
        rv = asprintf(&info->path, "/proc/%i/fd/%i", getpid(), info->fd);
    } else {
        rv = asprintf(&info->path, "/dev/shm/%s", info->shm_name);
    }
    if (rv == -1) {
        perror("setup_libscope:path");
        info->path = NULL; // failure leaves info->path undefined
        goto err;
    }

/*
 * DEVMODE is here only to help with gdb. The debugger has
 * a problem reading symbols from a /proc pathname.
 * This is expected to be enabled only by developers and
 * only when using the debugger.
 */
#if DEVMODE == 1
    asprintf(&info->path, "./lib/linux/libscope.so");
    printf("LD_PRELOAD=%s\n", info->path);
#endif

    everything_successful = TRUE;

err:
    if (!everything_successful) release_libscope(&info);
    return info;
}

// If possible, we want to set GODEBUG=http2server=0
// This tells go not to upgrade servers to http2, which allows
// our http1 protocol capture stuff to do it's thing.
// We consider this temporary, because when we support http2
// it will not be necessary.
static void
setGoHttpEnvVariable(void)
{
    char *cur_val = getenv(GO_ENV_VAR);

    // If GODEBUG isn't set, try to set it to http2server=0
    if (!cur_val) {
        if (setenv(GO_ENV_VAR, GO_ENV_VALUE "=0", 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        return;
    }

    // GODEBUG is set.
    // If http2server wasn't specified, let's append ",http2server=0"
    if (!strstr(cur_val, GO_ENV_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_VALUE) == -1)) {
            perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) free(new_val);
    }
}

int
main(int argc, char **argv, char **env)
{
    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    int status;
    pid_t pid;
    void *handle = NULL;
    libscope_info_t *info;

    // Use dlsym to get addresses for everything in g_fn
    initFn();

    //check command line arguments 
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }
    info = setup_libscope();
    if (!info) {
        fprintf(stderr, "%s:%d ERROR: unable to set up libscope\n", __FUNCTION__, __LINE__);
        exit(EXIT_FAILURE);
    }

    ebuf = getElf(argv[1]);

    if ((ebuf != NULL) && (app_type(ebuf->buf, SHT_NOTE, ".note.go.buildid") != 0)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }

        setGoHttpEnvVariable();

    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);
        
        if (setenv("LD_PRELOAD", info->path, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }
        
        pid = fork();
        if (pid == -1) {
            perror("fork");
            goto err;
        } else if (pid > 0) {
            int ret;
            do {
                ret = waitpid(pid, &status, 0);
            } while (ret == -1 && errno == EINTR);

            release_libscope(&info);
            exit(status);
        } else {
            execve(argv[1], &argv[1], environ);
            perror("execve");
            goto err;
        }
    }

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    // Static executable path
    if ((handle = dlopen(info->path, RTLD_LAZY)) == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }
    release_libscope(&info);

    sys_exec(ebuf, argv[1], argc, argv, env);

    return 0;
err:
    release_libscope(&info);
    if (ebuf) free(ebuf);
    exit(EXIT_FAILURE);
}
