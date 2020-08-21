/*
 * Load and run static executables
 *
 * objcopy -I binary -O elf64-x86-64 -B i386 ./lib/linux/libscope.so ./lib/linux/libscope.o
 * gcc -Wall -g src/scope.c -ldl -o scope ./lib/linux/libscope.o
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

#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))
#define __NR_memfd_create   319
#define _MFD_CLOEXEC		0x0001U
#define SHM_NAME            "libscope"

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
get_file_size(char *path)
{
    int fd;
    struct stat sbuf;

    if ((fd = open(path, O_RDONLY)) == -1) {
        perror("get_file_size:open");
        return -1;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("get_file_size:fstat");
        return -1;
    }

    if (close(fd) == -1) {
        perror("get_file_size:close");
        return -1;        
    }

    return sbuf.st_size;
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

static void
dump_elf(char *buf, size_t len)
{
    if (!buf) return;

    if (munmap(buf, len) == -1) {
        perror("dump_elf: munmap");
    }
}

static char *
get_elf(char *path)
{
    int fd;
    char *buf;
    Elf64_Ehdr *elf;
    struct stat sbuf;

    if ((fd = open(path, O_RDONLY)) == -1) {
        perror("get_elf:open");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("get_elf:fstat");
        return NULL;        
    }

    if ((buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                    PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
        perror("get_elf:mmap");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("get_elf:close");
        dump_elf(buf, sbuf.st_size);
        return NULL;        
    }

    elf = (Elf64_Ehdr *)buf;
    if((elf->e_ident[EI_MAG0] != 0x7f) ||
       strncmp((char *)&elf->e_ident[EI_MAG1], "ELF", 3) ||
       (elf->e_ident[EI_CLASS] != ELFCLASS64) ||
       (elf->e_ident[EI_DATA] != ELFDATA2LSB) ||
       (elf->e_machine != EM_X86_64)) {
        printf("%s:%d ERROR: %s is not a viable ELF file\n", __FUNCTION__, __LINE__, path);
        dump_elf(buf, sbuf.st_size);
        return NULL;
    }

    if (elf->e_type != ET_EXEC) {
        printf("%s:%d %s is not a static executable\n", __FUNCTION__, __LINE__, path);
        dump_elf(buf, sbuf.st_size);
        return NULL;
    }

    return buf;
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

    if (uname(&buffer) == 1) {
        perror("check_kernel_version:uname");
        return -1;
    }
    token = strtok(buffer.release, separator);
    val = atoi(token);
    if (val < 3) {
        return 0;
    } else if (val > 3){
        return 1;
    }

    token = strtok(NULL, separator);
    return atoi(token) < 17 ? 0 : 1;
}

static libscope_info_t *
setup_libscope()
{
    int fd;
    size_t libsize;
    char shm_name[255];
    libscope_info_t *info;
    int use_memfd;

    if ((use_memfd = check_kernel_version()) == -1) {
        return NULL;
    }
    
    if (use_memfd) {
        fd = _memfd_create(SHM_NAME, _MFD_CLOEXEC);
    } else {
        if (snprintf(shm_name, sizeof(shm_name), "%s%i", SHM_NAME, getpid()) == -1) {
            return NULL;
        }
        fd = shm_open(shm_name, O_RDWR | O_CREAT, S_IRWXU);
    }
    if (fd == -1) {
        perror(use_memfd ? "setup_libscope:memfd_create" : "setup_libscope:shm_open");
        return NULL;
    }
    
    libsize = (size_t) (&_binary___lib_linux_libscope_so_end - &_binary___lib_linux_libscope_so_start);
    if (write(fd, &_binary___lib_linux_libscope_so_start, libsize) != libsize) {
        perror("setup_libscope:write");
        goto err;
    }

    if ((info = calloc(1, sizeof(libscope_info_t))) == NULL ) {
        perror("setup_libscope:calloc");
        goto err;
    }

    info->fd = fd;
    info->use_memfd = use_memfd;

    if (use_memfd) {
        if (asprintf(&info->path, "/proc/%i/fd/%i", getpid(), fd) == -1) {
            goto err;
        }
    } else {
        if (asprintf(&info->path, "/dev/shm/%s", shm_name) == -1) {
            goto err;
        }
        if (asprintf(&info->shm_name, "%s", shm_name) == -1) {
            goto err;
        }
    }

    return info;
err: 
    close(fd);
    if (info) {
        if (info->path) free(info->path);
        if (info->shm_name) free(info->shm_name);
        free(info);
    }
    return NULL;
}

static void
release_libscope(libscope_info_t **info) {
    if (!info || !*info) return;
    close((*info)->fd);
    if ((*info)->shm_name) {
        shm_unlink((*info)->shm_name);
        free((*info)->shm_name);
    }
    if ((*info)->path) free((*info)->path);
    free(*info);
    *info = NULL;
}

int
main(int argc, char **argv, char **env)
{
    int flen;
    char *buf;
    int (*sys_exec)(const char *, const char *, int, char **, char **);
    void *handle;
    libscope_info_t *info;

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

    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, argv[1]);

    if (((buf = get_elf(argv[1])) == NULL) ||
        (!is_static(buf))) {
        printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if ((flen = get_file_size(argv[1])) == -1) {
            fprintf(stderr, "%s:%d ERROR: file size\n", __FUNCTION__, __LINE__);
            goto err;
        }

        dump_elf(buf, flen);
        
        if (setenv("LD_PRELOAD", info->path, 0) == -1) {
            perror("setenv");
            goto err;
        }
        printf("LD_PRELOAD=%s\n", info->path);
        
        pid_t pid = fork();
        if (pid == -1) {
            perror("fork");
            goto err;
        } else if (pid > 0) {
            int status;
            waitpid(pid, &status, 0);
            release_libscope(&info);
            exit(status);
        } else {
            execve(argv[1], &argv[1], environ);
        }
    }

    handle = dlopen(info->path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }
    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        fprintf(stderr, "%s\n", dlerror());
        goto err;
    }
    release_libscope(&info);

    sys_exec(buf, argv[1], argc, argv, env);

    return 0;
err:
    release_libscope(&info);
    exit(EXIT_FAILURE);
}
