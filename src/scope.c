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

#define DEVMODE 0
#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))
#define __NR_memfd_create   319
#define _MFD_CLOEXEC		0x0001U
#define SHM_NAME            "libscope"
#define TRUE 1
#define FALSE 0

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

typedef struct elf_buf_t {
    char *buf;
    int len;
} elf_buf;

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

static void
dump_elf(char *buf, size_t len)
{
    if (!buf) return;

    if (munmap(buf, len) == -1) {
        perror("dump_elf: munmap");
    }
}

static elf_buf *
get_elf(char *path)
{
    int fd;
    elf_buf *ebuf;
    Elf64_Ehdr *elf;
    struct stat sbuf;

    if ((ebuf = calloc(1, sizeof(struct elf_buf_t))) == NULL) {
        perror("calloc:get_elf");
        return NULL;
    }

    if ((fd = open(path, O_RDONLY)) == -1) {
        perror("get_elf:open");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("get_elf:fstat");
        return NULL;        
    }

    ebuf->len = sbuf.st_size;

    if ((ebuf->buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                          PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
        perror("get_elf:mmap");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("get_elf:close");
        dump_elf(ebuf->buf, sbuf.st_size);
        if (ebuf) free(ebuf);
        return NULL;        
    }

    elf = (Elf64_Ehdr *)ebuf->buf;
    if((elf->e_ident[EI_MAG0] != 0x7f) ||
       strncmp((char *)&elf->e_ident[EI_MAG1], "ELF", 3) ||
       (elf->e_ident[EI_CLASS] != ELFCLASS64) ||
       (elf->e_ident[EI_DATA] != ELFDATA2LSB) ||
       (elf->e_machine != EM_X86_64)) {
        //printf("%s:%d ERROR: %s is not a viable ELF file\n", __FUNCTION__, __LINE__, path);
        dump_elf(ebuf->buf, sbuf.st_size);
        if (ebuf) free(ebuf);
        return NULL;
    }

    if (elf->e_type != ET_EXEC) {
        //printf("%s:%d %s is not a static executable\n", __FUNCTION__, __LINE__, path);
        dump_elf(ebuf->buf, sbuf.st_size);
        if (ebuf) free(ebuf);
        return NULL;
    }

    return ebuf;
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

#if DEVMODE == 1
    snprintf(info->path, sizeof(info->path), "./lib/linux/libscope.so");
    printf("LD_PRELOAD=%s\n", info->path);
    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, argv[1]);
#endif

    everything_successful = TRUE;

err:
    if (!everything_successful) release_libscope(&info);
    return info;
}

int
main(int argc, char **argv, char **env)
{
    elf_buf *ebuf;
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

    ebuf = get_elf(argv[1]);

    if ((ebuf != NULL) && (app_type(ebuf->buf, SHT_NOTE, ".note.go.buildid") != 0)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }
    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        //printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if (ebuf) dump_elf(ebuf->buf, ebuf->len);
        
        if (setenv("LD_PRELOAD", info->path, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }
        
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

    // Static executable path
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

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    sys_exec(ebuf->buf, argv[1], argc, argv, env);

    return 0;
err:
    release_libscope(&info);
    if (ebuf) free(ebuf);
    exit(EXIT_FAILURE);
}
