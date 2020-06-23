/*
 * Load and run static executables
 *
 * gcc -Wall -g src/scope.c -L./lib/linux -lscope -o scope
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>

#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))

extern char **environ;
extern int sys_exec(const char *, const char *, int, char **, char **);

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
    
    //check command line arguments 
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, argv[1]);

    if (((buf = get_elf(argv[1])) == NULL) ||
        (!is_static(buf))) {
        printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if (setenv("LD_PRELOAD", "libscope.so", 0) == -1) {
            perror("setenv");
        }

        if (fork() == 0) {
            execve(argv[1], &argv[1], environ);
            perror("execve");
            exit(-1);
        }

        if ((flen = get_file_size(argv[1])) == -1) {
            printf("%s:%d ERROR: file size\n", __FUNCTION__, __LINE__);
            exit(-1);
        }

        dump_elf(buf, flen);

        exit(0);
    }

    sys_exec(buf, argv[1], argc, argv, env);

    return 0;
}
