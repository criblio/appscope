/*
 * Experimenting with loading static executables
 *
 * gcc -Wall -g test/manual/loadStatic.c -o lds
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <elf.h>

#define USE_MMAP 1
#define ROUND_DOWN(num, unit) ((num) & ~((unit) - 1))
#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))

extern char **environ;

void
usage(char *prog) {
  fprintf(stderr,"usage: %s [-v] -f static-executable\n", prog);
  exit(-1);
}

static int
osGetPageProt(uint64_t addr)
{
    int prot = -1;
    size_t len = 0;
    char *buf = NULL;

    FILE *fstream = fopen("/proc/self/maps", "r");
    if (fstream == NULL) return -1;

    while (getline(&buf, &len, fstream) != -1) {
        char *end = NULL;
        errno = 0;
        uint64_t addr1 = strtoull(buf, &end, 0x10);
        if ((addr1 == 0) || (errno != 0)) {
            if (buf) free(buf);
            fclose(fstream);
            return -1;
        }

        uint64_t addr2 = strtoull(end + 1, &end, 0x10);
        if ((addr2 == 0) || (errno != 0)) {
            if (buf) free(buf);
            fclose(fstream);
            return -1;
        }

        printf("addr 0x%lux addr1 0x%lux addr2 0x%lux\n", addr, addr1, addr2);

        if ((addr >= addr1) && (addr <= addr2)) {
            char *perms = end + 1;
            printf("matched 0x%lx to 0x%lx-0x%lx\n\t%c%c%c",
                   addr, addr1, addr2, perms[0], perms[1], perms[2]);

            prot = 0;
            prot |= perms[0] == 'r' ? PROT_READ : 0;
            prot |= perms[1] == 'w' ? PROT_WRITE : 0;
            prot |= perms[2] == 'x' ? PROT_EXEC : 0;
            if (buf) free(buf);
            break;
        }

        if (buf) {
            free(buf);
            buf = NULL;
        }

        len = 0;
    }

    fclose(fstream);
    return prot;
}

static void
dump_elf(char *buf, size_t len)
{
    if (!buf) return;
#if USE_MMAP
    if (munmap(buf, len) == -1) {
        perror("dump_elf: munmap");
    }
#else
    free(buf);
#endif
}
static int
map_segment(char *buf, Elf64_Phdr *phead, int map)
{
    int prot;
    int pgsz = sysconf(_SC_PAGESIZE);
    void *addr;
    
    if ((phead->p_vaddr == 0) || (phead->p_memsz <= 0) || (phead->p_flags == 0)) return -1;

    prot = 0;
    prot |= (phead->p_flags & PF_R) ? PROT_READ : 0;
    prot |= (phead->p_flags & PF_W) ? PROT_WRITE : 0;
    prot |= (phead->p_flags & PF_X) ? PROT_EXEC : 0;

    printf("%s:%d vaddr 0x%lx size 0x%lx\n", __FUNCTION__, __LINE__, phead->p_vaddr, (size_t)phead->p_memsz);
    if (map == 1) {
        if ((addr = mmap((void *)ROUND_DOWN(phead->p_vaddr, pgsz),
                         (size_t)phead->p_memsz, prot | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                         -1, (off_t)NULL)) == MAP_FAILED) {
            perror("load_segment:mmap");
            return -1;
        }
    } else {
        addr = (void *)phead->p_vaddr;
        prot = osGetPageProt((uint64_t)addr);
        if (mprotect((void *)ROUND_DOWN(phead->p_vaddr, pgsz), phead->p_memsz, prot | PROT_WRITE) == -1) {
            perror("load_segment:mprotect");
            return -1;
        }
    }

    memmove(addr, &buf[phead->p_offset], phead->p_memsz);

    if (((prot & PROT_WRITE) == 0) && (mprotect((void *)ROUND_DOWN(phead->p_vaddr, pgsz), phead->p_memsz, prot) == -1)) {
        perror("load_segment:mprotect");
        return -1;
    }

    return 0;
}

static int
load_elf(char *buf)
{
    int i;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];
    Elf64_Half pnum = elf->e_phnum;

    for (i = 0; i < pnum; i++) {
        if (phead[i].p_type == PT_LOAD) {
            map_segment(buf, &phead[i], 1);
        } else {
            map_segment(buf, &phead[i], 0);
        }
    }

    return 0;
}

static uint64_t
get_symbol(char *buf, char *sname)
{
    int i, nsyms = 0;
    uint64_t symaddr = 0;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    Elf64_Sym *symtab = NULL;
    const char *section_strtab = NULL;
    const char *strtab = NULL;
    const char *sec_name = NULL;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)((char *)buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;
    
    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;

        if (sections[i].sh_type == SHT_SYMTAB) {
            symtab = (Elf64_Sym *)((char *)buf + sections[i].sh_offset);
            nsyms = sections[i].sh_size / sections[i].sh_entsize;
        } else if (sections[i].sh_type == SHT_STRTAB && strcmp(sec_name, ".strtab") == 0) {
            strtab = (const char *)(buf + sections[i].sh_offset);
        }

        if ((strtab != NULL) && (symtab != NULL)) break;
        printf("section %s = 0x%lx, size = 0x%lx\n",
               section_strtab + sections[i].sh_name,
               sections[i].sh_addr,
               sections[i].sh_size);
    }

    for (i=0; i < nsyms; i++) {
        if (strcmp(sname, strtab + symtab[i].st_name) == 0) {
            symaddr = symtab[i].st_value;
            printf("symbol found %s = 0x%08lx\n", strtab + symtab[i].st_name, symtab[i].st_value);
            break;
        }
    }

  return symaddr;
}
#if 0
static int
set_vars(char *buf, char **vars)
{
    int i;
    char *sym;

    if (!vars || !*vars) return -1;

    for (i = 0; *vars; i++) {
        if ((sym = (char *)get_symbol(buf, vars[i])) == 0) return -1;
    }

    return 0;
}
#endif

static int
copy_strings(int argc, char **argv)
{
    int i;
    char *evsrc = __environ;
    char *evdst = get_symbol("__environ");

    if (!evsrc || !evdst) return -1;
    
    for (i = 0; evsrc; i++) {
        
    }

    return 0;
}
   
//The first six integer or pointer arguments are passed in registers RDI, RSI, RDX, RCX, R8, R9
static int
set_go(Elf64_Ehdr *ehdr, int argc, char **argv)
{
    //uint64_t val = (uint64_t)(argv) + (sizeof(uint64_t) * argc);
    //uint64_t val2 = (uint64_t)(argv) + (sizeof(uint64_t) * (argc + 1));
    size_t val = sizeof(uint64_t) * argc;
    size_t val2 = sizeof(uint64_t) * (argc + 1);
    uint64_t res;
    //char *sfunc = (char *)ehdr->e_entry;
    //int (*statexec)() = (int (*)())ehdr->e_entry;

    // src,dst
    //asm volatile ("movl (%0), %%eax" :: "r" (argc));
    //asm volatile ("movw %0, %%ds" :: "r" (ctxt->ds));
    //asm volatile("movl (%1),%%eax" : "=a" (val) : "r" (pos));

    //asm volatile ("push %rsi");
    //asm volatile ("push %rdx");
    
    //     "orw $2, %%ax \n"
    // "mov %1, %%rax \n"
    //     "push %%rax \n"
    //     : "%rax"         /* clobbered register */


    //__asm__ volatile ("mov %1, 0(%%rsp) \n" : "=r"(res) : "r"(argc));


    __asm__ volatile (
    "mov %1, 0(%%rsp) \n"
    : "=r"(res) /* output */
    : "r"(argc)      /* input */
    );

    __asm__ volatile (
    "mov %1, 8(%%rsp) \n"
    : "=r"(res) /* output */
    : "r"(argv)      /* input */
    );

    __asm__ volatile (
    "mov %%rsp, %%rax \n"
    "add %1, %%rax \n"
    "movl $0, 0(%%rax) \n"
    : "=r"(res) /* output */
    : "r"(val)      /* input */
    : "%rax"         /* clobbered register */
    );

    __asm__ volatile (
    "mov %%rsp, %%rax \n"
    "add %1, %%rax \n"
    "mov %2, 0(%%rax) \n"
    : "=r"(res) /* output */
    : "r"(val2), "r"(*environ)      /* input */
    : "%rax"         /* clobbered register */
    );

    //statexec();
    //__asm__ volatile ("jmp %0 \n" : : "r"((uint64_t)ehdr->e_entry));
    __asm__ volatile (
    "mov %1, %%rax \n"
    "jmp *%%rax \n"
    : "=r"(res) /* output */
    : "r"(ehdr->e_entry)
    : "%rax"         /* clobbered register */
    );

    return 0;
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
#if USE_MMAP
    if ((buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                    PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
        perror("mmap:get_elf");
        return NULL;
    }

#else
    if ((buf = calloc(1, sbuf.st_size)) == NULL) {
        perror("calloc:get_elf");
        return NULL;        
    }

    if (read(fd, buf, sbuf.st_size) <= 0) {
        perror("read:get_elf");
        dump_elf(buf, sbuf.st_size);
        return NULL;        
    }
#endif
    if (close(fd) == -1) {
        perror("close:get_elf");
        dump_elf(buf, sbuf.st_size);
        return NULL;        
    }

    elf = (Elf64_Ehdr *)buf;
    if((elf->e_ident[0] != 0x7f) || strncmp((char *)&elf->e_ident[1], "ELF", 3)) {
        printf("%s:%d ERROR: %s is not an ELF file\n", __FUNCTION__, __LINE__, path);
        dump_elf(buf, sbuf.st_size);
        return NULL;
    }

    if (elf->e_type != ET_EXEC) {
        printf("%s:%d ERROR: %s is not a static executable\n", __FUNCTION__, __LINE__, path);
        dump_elf(buf, sbuf.st_size);
        return NULL;
    }

    return buf;

}

int
main(int argc, char **argv)
{
    int opt, verbose = 0;
    char *path, *buf;
    Elf64_Ehdr *ehdr;
    //int (*statexec)(int, char **);

    printf("Starting interpose test\n");

    //check command line arguments 
    if (argc != 2) {
        usage(argv[0]);
        exit(1);
    }
    
    while ((opt = getopt(argc, argv, "vhf:")) > 0) {
        switch (opt) {
        case 'v': verbose++; break;
        case 'f': path = optarg; break;
        case 'h': default: usage(argv[0]); break;
        }
    }
    
    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, path);

    if ((buf = get_elf(path)) == NULL) {
        printf("%s:%d ERROR: read_file\n", __FUNCTION__, __LINE__);
        exit(-1);
    }

    ehdr = (Elf64_Ehdr *)buf;
    printf("%s:%d type: %d\n", __FUNCTION__, __LINE__, ehdr->e_type);

    load_elf(buf);
    printf("%s:%d sym @ 0x%lx\n", __FUNCTION__, __LINE__, get_symbol(buf, "__environ"));

    set_go(ehdr, argc, argv);

    //statexec = (int (*)(int, char **))ehdr->e_entry;
    //statexec(0, NULL);
    //dump_elf(buf, sbuf.st_size);
    return 0;
}
