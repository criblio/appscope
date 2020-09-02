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
uint64_t *g_currsheap = NULL;
uint64_t *g_heapend = NULL;

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
    scopeLog(str, -1, CFG_LOG_DEBUG);
}

#if 0
static int
get_file_size(const char *path)
{
    int fd;
    struct stat sbuf;

    if ((fd = open(path, O_RDONLY)) == -1) {
        scopeLog("ERROR: get_file_size:open", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (fstat(fd, &sbuf) == -1) {
        scopeLog("ERROR: get_file_size:fstat", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (close(fd) == -1) {
        scopeLog("EROR: get_file_size:close", -1, CFG_LOG_ERROR);
        return -1;        
    }

    return sbuf.st_size;
}
#endif

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

    if ((phead->p_vaddr == 0) || (phead->p_memsz <= 0) || (phead->p_flags == 0)) return -1;

    prot = 0;
    prot |= (phead->p_flags & PF_R) ? PROT_READ : 0;
    prot |= (phead->p_flags & PF_W) ? PROT_WRITE : 0;
    prot |= (phead->p_flags & PF_X) ? PROT_EXEC : 0;

    laddr = (char *)ROUND_DOWN(phead->p_vaddr, pgsz);

    sysprint("%s:%d vaddr 0x%lx size 0x%lx\n",
             __FUNCTION__, __LINE__, phead->p_vaddr, (size_t)phead->p_memsz);

    if ((addr = mmap(laddr, ROUND_UP((size_t)phead->p_memsz, phead->p_align),
                     prot | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                     -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLog("ERROR: load_segment:mmap", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (laddr != addr) {
        scopeLog("ERROR: load_segment:mmap:laddr mismatch", -1, CFG_LOG_ERROR);
        return -1;
    }

    load_sections(buf, (char *)phead->p_vaddr, (size_t)phead->p_memsz);

    if (((prot & PROT_WRITE) == 0) && (mprotect(laddr, phead->p_memsz, prot) == -1)) {
        scopeLog("ERROR: load_segment:mprotect", -1, CFG_LOG_ERROR);
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

static int
unmap_all(char *buf, const char **argv)
{
#if 0
    Elf64_Half phnum;
    int flen, i;
    int pgsz = sysconf(_SC_PAGESIZE);
    Elf64_Phdr *phead;
    
    if ((flen = get_file_size(argv[1])) == -1) {
        scopeLog("ERROR:unmap_all: file size", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (munmap(buf, flen) == -1) {
        scopeLog("ERROR: unmap_all: munmap(1)", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((phead = (Elf64_Phdr *)getauxval(AT_PHDR)) == 0) {
        scopeLog("ERROR: unmap_all: getauxval", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((phnum = (Elf64_Half)getauxval(AT_PHNUM)) == 0) {
        scopeLog("ERROR: unmap_all: getauxval", -1, CFG_LOG_ERROR);
        return -1;
    }

    for (i = 0; i < phnum; i++) {
        if (phead[i].p_type == PT_LOAD) {
            if (munmap((void *)ROUND_DOWN(phead[i].p_vaddr, pgsz), phead[i].p_memsz) == -1) {
                scopeLog("ERROR: unmap_all: munmap(2)", -1, CFG_LOG_ERROR);
                return -1;
            }
        }

    }
#endif
    return 0;
}

static int
copy_strings(char *buf, uint64_t sp, int argc, const char **argv, const char **env)
{
    int i;
    Elf64_auxv_t *auxv;
    char **astart;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;
    char **spp = (char **)sp;
    uint64_t cnt = (uint64_t)argc - 1;
    Elf64_Addr *elf_info;
    
    if (!buf || !spp || !argv || !*argv || !env || !*env) return -1;

    elf = (Elf64_Ehdr *)buf;
    phead = (Elf64_Phdr *)&buf[elf->e_phoff];
    
    // do argc
    *spp++ = (char *)cnt;
    
    // do argv; start at argv[1] to get the app's args
    for (i = 0; i < (argc - 1); i++) {
        if (&argv[i + 1] && spp) {
            *spp++ = (char *)argv[i + 1];
        } else {
            scopeLog("ERROR:copy_strings: arg entry is not correct", -1, CFG_LOG_ERROR);
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
            scopeLog("ERROR:copy_strings: environ string is not correct", -1, CFG_LOG_ERROR);
            return -1;
        }
    }

    // end of env
    *spp++ = NULL;

    elf_info = (Elf64_Addr *)spp;
    memset(elf_info, 0, sizeof(Elf64_Addr) * ((AT_EXECFN + 1) * 2));

    // auxv entries start right after env is null
    astart = (char **)env;    // leaving env in place for debugging
    while(*astart++ != NULL);

    for (auxv = (Elf64_auxv_t *)astart; auxv->a_type != AT_NULL; auxv++) {
        switch ((unsigned long)auxv->a_type) {
        case AT_PHDR:
            AUX_ENT(auxv->a_type, (Elf64_Addr)phead);
            break;

        case AT_PHNUM:
            AUX_ENT(auxv->a_type, elf->e_phnum);
            break;

        case AT_BASE:
            AUX_ENT(auxv->a_type, -1);
            break;

        case AT_ENTRY:
            AUX_ENT(auxv->a_type, elf->e_entry);
            break;

        case AT_EXECFN:
            AUX_ENT(auxv->a_type, (unsigned long)argv[1]);
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
set_go(char *buf, int argc, const char **argv, const char **env, Elf64_Addr laddr)
{
    //int pgsz = sysconf(_SC_PAGESIZE);
    uint64_t res = 0;
    char *sp;
    Elf64_Addr start;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;

    // create a heap (void *)ROUND_UP(laddr + pgsz, pgsz)  | MAP_FIXED
    if ((g_currsheap = mmap(NULL, HEAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS,
                            -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLog("set_go:mmap", -1, CFG_LOG_ERROR);
        return -1;
    }

    g_heapend = g_currsheap + HEAP_SIZE;

    // create a stack (void *)ROUND_UP(laddr + pgsz + HEAP_SIZE, pgsz)  | MAP_FIXED
    if ((sp = mmap(NULL, STACK_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN,
                   -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLog("set_go:mmap", -1, CFG_LOG_ERROR);
        return -1;
    }

    // build the stack
    copy_strings(buf, (uint64_t)sp, argc, argv, env);
    start = ehdr->e_entry;

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&scope_fs) == -1) {
        scopeLog("set_go:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    unmap_all(buf, argv);

#if 0
    rc = syscall(SYS_brk, heap);
    printf("%s:%d rc 0x%lx heap %p\n", __FUNCTION__, __LINE__, rc, heap);
    //if (brk(heap) == -1) scopeLog("set_go:brk", -1, CFG_LOG_ERROR);

    if (prctl(PR_SET_MM, PR_SET_MM_START_STACK, (unsigned long)sp,
              (unsigned long)0, (unsigned long)0) == -1) {
        scopeLog("set_go:prctl:PR_SET_MM_START_STACK", -1, CFG_LOG_ERROR);
    } else {
        printf("%s:%d\n", __FUNCTION__, __LINE__);
    }

    if (prctl(PR_SET_MM, PR_SET_MM_BRK, (unsigned long)heap,
              (unsigned long)0, (unsigned long)0) == -1) {
        scopeLog("set_go:prctl:PR_SET_MM_BRK", -1, CFG_LOG_ERROR);
    } else {
        printf("%s:%d heap %p sbrk %p\n", __FUNCTION__, __LINE__, heap, sbrk(0));
    }
#endif    

    __asm__ volatile (
        "lea scope_stack(%%rip), %%r11 \n"
        "mov %%rsp, (%%r11)  \n"
        "mov %1, %%r11 \n"
        "mov %2, %%rsp \n"
        "jmp *%%r11 \n"
        : "=r"(res)                   //output
        : "r"(start), "r"(sp)
        : "%r11"                      //clobbered register
    );

    return 0;
}

EXPORTON int
sys_exec(elf_buf_t *ebuf, const char *path, int argc, const char **argv, const char **env)
{
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)ebuf->buf;
    Elf64_Addr lastaddr;

    if (!ebuf || !path || !argv || (argc < 1)) return -1;

    scopeLog("sys_exec type:", ehdr->e_type, CFG_LOG_DEBUG);

    lastaddr = load_elf((char *)ebuf->buf);

    // TODO: are we loading a Go app or a glibc app?
    initGoHook(ebuf);

    threadNow(0);

    set_go((char *)ebuf->buf, argc, argv, env, lastaddr);

    return 0;
}
