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
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>

#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

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
        perror("open:get_elf");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("fstat:get_elf");
        return NULL;        
    }

    if ((buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                    PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
        perror("mmap:get_elf");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("close:get_elf");
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


int
main(int argc, char **argv, char **env)
{
    int flen;
    char *buf;
    char path[1024];
    int fd;
    int (*sys_exec)(const char *, const char *, int, char **, char **);
    void *handle;
    size_t libsize;
    
    //check command line arguments 
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    fd = memfd_create("", 0);
    if (fd == -1) {
        perror("memfd_create");
        exit(EXIT_FAILURE);
    }
    libsize = (size_t) (&_binary___lib_linux_libscope_so_end - &_binary___lib_linux_libscope_so_start);
    if (write(fd, &_binary___lib_linux_libscope_so_start, libsize) != libsize) {
        perror("write");
        close(fd);
        exit(EXIT_FAILURE);
    }
    
    snprintf(path, sizeof(path), "/proc/%i/fd/%i", getpid(), fd);
    printf("LD_PRELOAD=%s\n", path);

    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, argv[1]);

    if (((buf = get_elf(argv[1])) == NULL) ||
        (!is_static(buf))) {
        printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if ((flen = get_file_size(argv[1])) == -1) {
            printf("%s:%d ERROR: file size\n", __FUNCTION__, __LINE__);
            exit(EXIT_FAILURE);
        }

        dump_elf(buf, flen);
        
        if (setenv("LD_PRELOAD", path, 0) == -1) {
            perror("setenv");
            exit(EXIT_FAILURE);
        }
        
        pid_t pid = fork();
        if (pid == -1) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid > 0) {
            int status;
            waitpid(pid, &status, 0);
            close(fd);
            exit(status);
        } else  {
            execve(argv[1], &argv[1], environ);
        }
    }

    handle = dlopen(path, RTLD_LAZY);
    if (!handle) {
        perror("dlopen");
        exit(EXIT_FAILURE);
    }
    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        perror("dlsym");
        exit(EXIT_FAILURE);
    }
    close(fd);

    sys_exec(buf, argv[1], argc, argv, env);

    return 0;
}
