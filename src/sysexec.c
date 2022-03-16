#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <elf.h>
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <pthread.h>
#include <dlfcn.h>

#include "fn.h"
#include "dbg.h"
#include "os.h"
#include "com.h"
#include "state.h"
#include "gocontext.h"

#define SYSPRINT_CONSOLE 0
#define PRINT_BUF_SIZE 1024
#define HEAP_SIZE (size_t)(500 * 1024)
// 1Mb + an 8kb guard
#define STACK_SIZE (size_t)(1024 * 1024) + (8 * 1024)

#define AUX_ENT(id, val)                        \
	do { \
		*elf_info = (Elf64_Addr)id; \
        elf_info++;     \
		*elf_info = (Elf64_Addr)val; \
        elf_info++;     \
	} while (0)

#define EXPORTON __attribute__((visibility("default")))

uint64_t scope_stack;
unsigned long scope_fs;

void
sysprint(const char* fmt, ...)
{
    // Create the string
    char str[PRINT_BUF_SIZE];

    if (fmt) {
        va_list args;
        va_start(args, fmt);
        int rv = vsnprintf(str, PRINT_BUF_SIZE, fmt, args);
        va_end(args);
        if (rv == -1) return;
    }

    // Output the string
#if SYSPRINT_CONSOLE > 0
    printf("%s", str);
#endif
    scopeLog(CFG_LOG_DEBUG, "%s", str);
}

static int
get_file_size(const char *path)
{
    int fd;
    struct stat sbuf;

    if (g_fn.open && (fd = g_fn.open(path, O_RDONLY)) == -1) {
        scopeLogError("ERROR: get_file_size:open");
        return -1;
    }

    if (fstat(fd, &sbuf) == -1) {
        scopeLogError("ERROR: get_file_size:fstat");
        return -1;
    }

    if (g_fn.close && g_fn.close(fd) == -1) {
        scopeLogError("ERROR: get_file_size:close");
        return -1;        
    }

    return sbuf.st_size;
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
            //sysprint("load_sections: laddr = %p len = 0x%lx end = %p section: %s\n", laddr, len, laddr + len, sec_name);
            if ((type != SHT_NOBITS) && ((flags & SHF_ALLOC) || (flags & SHF_EXECINSTR))) {
                memmove(laddr, &buf[sections[i].sh_offset], len);
            } else if (type == SHT_NOBITS) {
                memset(laddr, 0, len);
            }

            sysprint("%s:%d %s addr %p - %p\n",
                     __FUNCTION__, __LINE__, sec_name, laddr, laddr + len);
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
    unsigned long lsize;

    if ((phead->p_vaddr == 0) || (phead->p_memsz <= 0) || (phead->p_flags == 0)) return -1;

    prot = 0;
    prot |= (phead->p_flags & PF_R) ? PROT_READ : 0;
    prot |= (phead->p_flags & PF_W) ? PROT_WRITE : 0;
    prot |= (phead->p_flags & PF_X) ? PROT_EXEC : 0;

    laddr = (char *)ROUND_DOWN(phead->p_vaddr, pgsz);
    lsize = phead->p_memsz + ((char*)phead->p_vaddr - laddr);


    sysprint("%s:%d vaddr 0x%lx size 0x%lx\n",
             __FUNCTION__, __LINE__, phead->p_vaddr, (size_t)phead->p_memsz);

    if ((addr = mmap(laddr, ROUND_UP((size_t)lsize, phead->p_align),
                     prot | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                     -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLogError("ERROR: load_segment:mmap");
        return -1;
    }

    if (laddr != addr) {
        scopeLogError("ERROR: load_segment:mmap:laddr mismatch");
        return -1;
    }

    load_sections(buf, (char *)phead->p_vaddr, (size_t)lsize);

    if (((prot & PROT_WRITE) == 0) && (mprotect(laddr, lsize, prot) == -1)) {
        scopeLogError("ERROR: load_segment:mprotect");
        return -1;
    }

    laddr = addr + lsize;
    return (Elf64_Addr)ROUND_UP((Elf64_Addr)laddr, pgsz);
}

static Elf64_Addr
load_elf(char *buf)
{
    int i;
    int pgsz = sysconf(_SC_PAGESIZE);
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];
    Elf64_Half pnum = elf->e_phnum;
    Elf64_Half phsize = elf->e_phentsize;
    void *pheadaddr;

    if ((pheadaddr = mmap(NULL, ROUND_UP((size_t)(pnum * phsize), pgsz),
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLogError("ERROR: load_elf:mmap");
        return (Elf64_Addr)NULL;
    }

    memmove(pheadaddr, phead, (size_t)(pnum * phsize));

    for (i = 0; i < pnum; i++) {
        if (phead[i].p_type == PT_LOAD) {
            map_segment(buf, &phead[i]);
        }
    }

    return (Elf64_Addr)pheadaddr;
}

static int
unmap_all(char *buf, const char **argv)
{
    int arg = 1;
    if (is_static(buf)) arg = 0;

    int flen;
    if ((flen = get_file_size(argv[arg])) == -1) {
        scopeLogError("ERROR:unmap_all: file size");
        return -1;
    }

    freeElf(buf, flen);
    return 0;
}

static int
copy_strings(char *buf, uint64_t sp, int argc, const char **argv, const char **env, Elf64_Addr phaddr)
{
    int i;
    Elf64_auxv_t *auxv;
    char **astart;
    Elf64_Ehdr *elf;
    char **spp = (char **)sp;
    uint64_t cnt = (uint64_t)argc;
    Elf64_Addr *elf_info;
    
    if (!buf || !spp || !argv || !*argv || !env || !*env) return -1;

    elf = (Elf64_Ehdr *)buf;
    
    // do argc
    *spp++ = (char *)cnt;
    
    // do argv
    for (i = 0; i < argc; i++) {
        if (&argv[i] && spp) {
            *spp++ = (char *)argv[i];
        } else {
            scopeLogError("ERROR:copy_strings: arg entry is not correct");
            return -1;
        }
    }

    // end of argv
    *spp++ = NULL;
    
    /* do env
     * Note that the env array on the stack are pointers to strings.
     * We are pointing to the strings provided from the executable, main,
     * re-using them for the new app we are starting.
     *
     * We are using 2 different values for env. 
     * First, is the environ variable from libc. Any call to setenv updates
     * environ and not env. Therefore, to ensure we set env for the new
     * app being started with any additional variables since main started
     * we need environ and not env.
     *
     * Second, we use env that was passed from main because it is the
     * pointer to what's on the stack. We need to locate the aux vectors
     * on the stack. They exist immediately after the env pointers.
     * Therefore, we need to start from the env on the stack in order to 
     * locate aux vectors.
    */
    for (i = 0; environ[i]; i++) {
        if ((&environ[i]) && spp) {
            *spp++ = (char *)environ[i];
        } else {
            scopeLogError("ERROR:copy_strings: environ string is not correct");
            return -1;
        }
    }

    // end of env
    *spp++ = NULL;

    // This is the destination for the new auxv array
    // The AUX_ENT macro uses elf_info
    elf_info = (Elf64_Addr *)spp;
    memset(elf_info, 0, sizeof(Elf64_Addr) * ((AT_EXECFN + 1) * 2));

    /*
     * There is an auxv vector that defines a TLS section from the elf image.
     * It's defined as AT_BASE or PT_TLS.
     * The aux type of AT_BASE and PT_TLS both define the same auxv entry.
     * If a static go exec on musl lib attempts to use the AT_BASE/PT_TLS auxv entry
     * as a pointer to TLS it segfaults. Setting an AT_BASE auxv entry to be ignored
     * allows both gnu and musl go static execs to function as expected.
     */

    // This is the source of the existing auxv array that is to be copied
    // auxv entries start right after env is null
    astart = (char **)env;
    while(*astart++ != NULL);

    for (auxv = (Elf64_auxv_t *)astart; auxv->a_type != AT_NULL; auxv++) {
        switch ((unsigned long)auxv->a_type) {
        case AT_PHDR:
            AUX_ENT(auxv->a_type, (Elf64_Addr)phaddr);
            break;

        case AT_PHNUM:
            AUX_ENT(auxv->a_type, elf->e_phnum);
            break;

        case AT_BASE:
            AUX_ENT(-1, -1);
            break;

        case AT_ENTRY:
            AUX_ENT(auxv->a_type, elf->e_entry);
            break;

        case AT_EXECFN:
            AUX_ENT(auxv->a_type, (unsigned long)argv[0]);
            break;

        default:
            AUX_ENT(auxv->a_type, auxv->a_un.a_val);
            break;
        }
    }

    AUX_ENT(0, 0);
    return 0;
}
   
//The first six integer or pointer arguments are passed in registers RDI, RSI, RDX, RCX, R8, R9
static int
set_go(char *buf, int argc, const char **argv, const char **env, Elf64_Addr phaddr)
{
    uint64_t res = 0;
    char *sp;
    Elf64_Addr start;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    char *rtld_fini = NULL;

    // create a stack (void *)ROUND_UP(laddr + pgsz + HEAP_SIZE, pgsz)  | MAP_FIXED
    if ((sp = mmap(NULL, STACK_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN,
                   -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLogError("set_go:mmap");
        return -1;
    }

    // build the stack
    copy_strings(buf, (uint64_t)sp, argc, argv, env, phaddr);
    start = ehdr->e_entry;

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&scope_fs) == -1) {
        scopeLogError("set_go:arch_prctl");
        return -1;
    }

    unmap_all(buf, argv);

    __asm__ volatile (
        "lea scope_stack(%%rip), %%r11 \n"
        "mov %%rsp, (%%r11)  \n"
        "mov %1, %%r11 \n"
        "mov %2, %%rsp \n"
        "mov %3, %%rdx \n"
        "jmp *%%r11 \n"
        : "=r"(res)                   //output
        : "r"(start), "r"(sp), "r"(rtld_fini)
        : "%r11"                      //clobbered register
    );

    return 0;
}

EXPORTON int
sys_exec(elf_buf_t *ebuf, const char *path, int argc, const char **argv, const char **env)
{
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)ebuf->buf;
    Elf64_Addr phaddr;

    if (!ebuf || !path || !argv || (argc < 1)) return -1;

    scopeLog(CFG_LOG_DEBUG, "fd:%d sys_exec type:", ehdr->e_type);

    phaddr = load_elf((char *)ebuf->buf);

    // TODO: are we loading a Go app or a glibc app?
    initGoHook(ebuf);

    set_go((char *)ebuf->buf, argc, argv, env, phaddr);

    return 0;
}
