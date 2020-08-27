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

#define DEVMODE 0
#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))

extern unsigned char _binary___lib_linux_libscope_so_start;
extern unsigned char _binary___lib_linux_libscope_so_end;

typedef struct elf_buf_t {
    char *buf;
    int len;
} elf_buf;

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
        perror("open:get_elf");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("fstat:get_elf");
        return NULL;        
    }

    ebuf->len = sbuf.st_size;

    if ((ebuf->buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                          PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
        perror("mmap:get_elf");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("close:get_elf");
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
        printf("%s:%d ERROR: %s is not a viable ELF file\n", __FUNCTION__, __LINE__, path);
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


int
main(int argc, char **argv, char **env)
{
    elf_buf *ebuf;
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

#if DEVMODE == 1
    snprintf(path, sizeof(path), "./lib/linux/libscope.so");
    printf("LD_PRELOAD=%s\n", path);
    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, argv[1]);
#else
    snprintf(path, sizeof(path), "/proc/%i/fd/%i", getpid(), fd);
#endif

    ebuf = get_elf(argv[1]);

    if ((ebuf != NULL) && (app_type(ebuf->buf, SHT_NOTE, ".note.go.buildid") != 0)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            if (ebuf) free(ebuf);
            exit(EXIT_FAILURE);
        }
    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            if (ebuf) free(ebuf);
            exit(EXIT_FAILURE);
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        //printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if (ebuf) dump_elf(ebuf->buf, ebuf->len);
        
        if (setenv("LD_PRELOAD", path, 0) == -1) {
            perror("setenv");
            if (ebuf) free(ebuf);
            exit(EXIT_FAILURE);
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            if (ebuf) free(ebuf);
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

    // Static executable path
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

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        exit(EXIT_FAILURE);
    }

    sys_exec(ebuf->buf, argv[1], argc, argv, env);

    return 0;
}
