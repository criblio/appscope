/*
 * Experimenting with loading static executables
 *
 * gcc -Wall -g test/manual/loadStatic.c -o lds
 */

#include <alloca.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

#define USE_MMAP  1
#define HEAP_SIZE (size_t)(200 * 1024)
// 1Mb + an 8kb guard
#define STACK_SIZE            (size_t)(1024 * 1024) + (8 * 1024)
#define ROUND_DOWN(num, unit) ((num) & ~((unit)-1))
#define ROUND_UP(num, unit)   (((num) + (unit)-1) & ~((unit)-1))
#define AUX_ENT(id, val)     \
    do {                     \
        elf_info[i++] = id;  \
        elf_info[i++] = val; \
    } while (0)

static char *g_exec_file;

void
usage(char *prog)
{
    fprintf(stderr, "usage: %s static-executable args-for-executable\n", prog);
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
    if (!buf)
        return;
#if USE_MMAP
    if (munmap(buf, len) == -1) {
        perror("dump_elf: munmap");
    }
#else
    free(buf);
#endif
}

static int
load_sections(char *buf, char *addr, size_t mlen)
{
    int i;
    Elf64_Word type;
    Elf64_Xword flags, len;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *sec_name;
    const char *section_strtab = NULL;
    char *laddr;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)((char *)buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        flags = sections[i].sh_flags;
        len = sections[i].sh_size;
        type = sections[i].sh_type;
        laddr = (char *)sections[i].sh_addr;

        if ((laddr >= addr) && (laddr <= (addr + mlen))) {
            sec_name = section_strtab + sections[i].sh_name;

            if ((type != SHT_NOBITS) && ((flags & SHF_ALLOC) || (flags & SHF_EXECINSTR))) {

                memmove(laddr, &buf[sections[i].sh_offset], len);
            } else if (type == SHT_NOBITS) {
                memset(laddr, 0, len);
            }

            printf("%s:%d %s addr %p - %p\n", __FUNCTION__, __LINE__, sec_name, laddr, laddr + len);
        }
    }

    return 0;
}

static Elf64_Addr
map_segment(char *buf, Elf64_Phdr *phead)
{
    int prot;
    int pgsz = sysconf(_SC_PAGESIZE);
    void *addr;
    char *laddr;

    if ((phead->p_vaddr == 0) || (phead->p_memsz <= 0) || (phead->p_flags == 0))
        return -1;

    prot = 0;
    prot |= (phead->p_flags & PF_R) ? PROT_READ : 0;
    prot |= (phead->p_flags & PF_W) ? PROT_WRITE : 0;
    prot |= (phead->p_flags & PF_X) ? PROT_EXEC : 0;

    laddr = (char *)ROUND_DOWN(phead->p_vaddr, pgsz);

    printf("%s:%d vaddr 0x%lx size 0x%lx\n", __FUNCTION__, __LINE__, phead->p_vaddr, (size_t)phead->p_memsz);

    if ((addr = mmap(laddr, (size_t)phead->p_memsz, prot | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, (off_t)NULL)) == MAP_FAILED) {
        perror("load_segment:mmap");
        return -1;
    }

    load_sections(buf, (char *)phead->p_vaddr, (size_t)phead->p_memsz);

    if (((prot & PROT_WRITE) == 0) && (mprotect(laddr, phead->p_memsz, prot) == -1)) {
        perror("load_segment:mprotect");
        return -1;
    }

    laddr = addr + phead->p_memsz;
    return (Elf64_Addr)ROUND_UP((Elf64_Addr)laddr, pgsz);
}

static Elf64_Addr
load_elf(char *buf)
{
    int i;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];
    Elf64_Half pnum = elf->e_phnum;
    Elf64_Addr endaddr = 0;

    for (i = 0; i < pnum; i++) {
        if (phead[i].p_type == PT_LOAD) {
            endaddr = map_segment(buf, &phead[i]);
        }
    }

    return endaddr;
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

        if ((strtab != NULL) && (symtab != NULL))
            break;
        printf("section %s type = %d flags = 0x%lx addr = 0x%lx-0x%lx, size = 0x%lx off = 0x%lx\n", sec_name, sections[i].sh_type, sections[i].sh_flags, sections[i].sh_addr,
               sections[i].sh_addr + sections[i].sh_size, sections[i].sh_size, sections[i].sh_offset);
    }

    for (i = 0; i < nsyms; i++) {
        if (strcmp(sname, strtab + symtab[i].st_name) == 0) {
            symaddr = symtab[i].st_value;
            printf("symbol found %s = 0x%08lx\n", strtab + symtab[i].st_name, symtab[i].st_value);
            break;
        }
    }

    return symaddr;
}

static int
unmap_all(char *buf)
{
    // wait on this until we are in the library
    return 0;

    Elf64_Half phnum;
    int flen, i;
    int pgsz = sysconf(_SC_PAGESIZE);
    ;
    Elf64_Phdr *phead;

    if ((flen = get_file_size(g_exec_file)) == -1) {
        printf("%s:%d ERROR: file size\n", __FUNCTION__, __LINE__);
        return -1;
    }

    dump_elf(buf, flen);

    if ((phead = (Elf64_Phdr *)getauxval(AT_PHDR)) == 0) {
        perror("unmap_all: getauxval");
        return -1;
    }

    if ((phnum = (Elf64_Half)getauxval(AT_PHNUM)) == 0) {
        perror("unmap_all: getauxval");
        return -1;
    }

    for (i = 0; i < phnum; i++) {
        if (phead[i].p_type == PT_LOAD) {
            if (munmap((void *)ROUND_DOWN(phead[i].p_vaddr, pgsz), phead[i].p_memsz) == -1) {
                perror("unmap_all: munmap");
                return -1;
            }
        }
    }

    return 0;
}

static int
copy_strings(char *buf, uint64_t sp, int argc, char **argv, char **env)
{
    int i;
    unsigned long auxindex;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;
    char **spp = (char **)sp;
    uint64_t cnt = (uint64_t)argc - 1;
    Elf64_Addr *elf_info;

    if (!buf || !spp || !argv || !*argv || !env || !*env)
        return -1;

    elf = (Elf64_Ehdr *)buf;
    phead = (Elf64_Phdr *)&buf[elf->e_phoff];

    // do argc
    *spp++ = (char *)cnt;

    // do argv; start at argv[1] to get the app's args
    for (i = 0; i < (argc - 1); i++) {
        if (&argv[i + 1] && spp) {
            *spp++ = argv[i + 1];
        } else {
            printf("%s:%d ERROR: arg entry is not correct\n", __FUNCTION__, __LINE__);
            return -1;
        }
    }

    // end of argv
    *spp++ = NULL;

    // do env
    for (i = 0; env[i]; i++) {
        if ((&env[i]) && spp) {
            *spp++ = env[i];
        } else {
            printf("%s:%d ERROR: environ string is not correct\n", __FUNCTION__, __LINE__);
            return -1;
        }
    }

    // end of env
    *spp++ = NULL;

    elf_info = (Elf64_Addr *)spp;
    memset(elf_info, 0, sizeof(Elf64_Addr) * (AT_EXECFN + 1) * 2);

    for (auxindex = 1, i = 0; auxindex < 32; auxindex++) {
        unsigned long val;

        switch (auxindex) {
            case AT_PHDR:
                AUX_ENT(auxindex, (Elf64_Addr)phead);
                break;

            case AT_PHNUM:
                AUX_ENT(auxindex, elf->e_phnum);
                break;

            case AT_BASE:
                AUX_ENT(auxindex, -1);
                break;

            case AT_ENTRY:
                AUX_ENT(auxindex, elf->e_entry);
                break;

            case AT_EXECFN:
                AUX_ENT(auxindex, (unsigned long)g_exec_file);
                break;

            case AT_EXECFD:
                // AUX_ENT(auxindex, );
                break;

            default:
                if ((val = getauxval(auxindex)) != 0) {
                    AUX_ENT(auxindex, val);
                }
        }
    }

    return 0;
}

// The first six integer or pointer arguments are passed in registers RDI, RSI, RDX, RCX, R8, R9
static int
set_go(char *buf, int argc, char **argv, char **env, Elf64_Addr laddr)
{
    // int pgsz = sysconf(_SC_PAGESIZE);
    uint64_t res;
    char *sp, *heap;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;

    // create a heap (void *)ROUND_UP(laddr + pgsz, pgsz)  | MAP_FIXED
    if ((heap = mmap(NULL, HEAP_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, (off_t)NULL)) == MAP_FAILED) {
        perror("set_go:mmap");
        return -1;
    }

    // create a stack (void *)ROUND_UP(laddr + pgsz + HEAP_SIZE, pgsz)  | MAP_FIXED
    if ((sp = mmap(NULL, STACK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN, -1, (off_t)NULL)) == MAP_FAILED) {
        perror("set_go:mmap");
        return -1;
    }

    // build the stack and change the heap
    copy_strings(buf, (uint64_t)sp, argc, argv, env);

    unmap_all(buf);
    brk(heap);
#if 0
    if (prctl(PR_SET_MM, PR_SET_MM_START_STACK, (unsigned long)sp,
              (unsigned long)0, (unsigned long)0) == -1) {
        perror("set_go:prctl:PR_SET_MM_START_STACK");
    }

    if (prctl(PR_SET_MM, PR_SET_MM_BRK, (unsigned long)heap,
              (unsigned long)0, (unsigned long)0) == -1) {
        perror("set_go:prctl:PR_SET_MM_BRK");
    }
#endif
    __asm__ volatile("mov %1, %%r12 \n"
                     "mov %2, %%rsp \n"
                     "jmp *%%r12 \n"
                     : "=r"(res) /* output */
                     : "r"(ehdr->e_entry), "r"(sp)
                     : "%rax" /* clobbered register */
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
    if ((buf = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)), PROT_READ, MAP_PRIVATE, fd, (off_t)NULL)) == MAP_FAILED) {
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
    if ((elf->e_ident[EI_MAG0] != 0x7f) || strncmp((char *)&elf->e_ident[EI_MAG1], "ELF", 3) || (elf->e_ident[EI_CLASS] != ELFCLASS64) || (elf->e_ident[EI_DATA] != ELFDATA2LSB) ||
        (elf->e_machine != EM_X86_64)) {
        printf("%s:%d ERROR: %s is not a viable ELF file\n", __FUNCTION__, __LINE__, path);
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
main(int argc, char **argv, char **env)
{
    int flen;
    char *buf;
    Elf64_Ehdr *ehdr;
    Elf64_Addr lastaddr;

    printf("Starting interpose test\n");

    // check command line arguments
    if (argc < 2) {
        usage(argv[0]);
        exit(1);
    }

    g_exec_file = argv[1];
    printf("%s:%d loading %s\n", __FUNCTION__, __LINE__, g_exec_file);

    if ((buf = get_elf(g_exec_file)) == NULL) {
        printf("%s:%d ERROR: read_file\n", __FUNCTION__, __LINE__);
        exit(-1);
    }

    if (!is_static(buf)) {
        printf("%s:%d not a static file\n", __FUNCTION__, __LINE__);

        if (fork() == 0) {
            execve(g_exec_file, argv, env);
            perror("execve");
            exit(-1);
        }

        exit(0);
    }

    ehdr = (Elf64_Ehdr *)buf;
    printf("%s:%d type: %d\n", __FUNCTION__, __LINE__, ehdr->e_type);

    lastaddr = load_elf(buf);
    printf("%s:%d last mapped addr 0x%lx\n", __FUNCTION__, __LINE__, lastaddr);
    printf("%s:%d sym @ 0x%lx\n", __FUNCTION__, __LINE__, get_symbol(buf, "__environ"));

    set_go(buf, argc, argv, env, lastaddr);

    if ((flen = get_file_size(g_exec_file)) == -1) {
        printf("%s:%d ERROR: file size\n", __FUNCTION__, __LINE__);
        exit(-1);
    }

    dump_elf(buf, flen);
    return 0;
}
