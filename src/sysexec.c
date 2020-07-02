#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>
#include <sys/auxv.h>
#include <sys/syscall.h>
#include <asm/prctl.h>
#include <sys/prctl.h>

#include "dbg.h"
#include "os.h"
#include "com.h"
#include "state.h"

#define HEAP_SIZE (size_t)(200 * 1024)
// 1Mb + an 8kb guard
#define STACK_SIZE (size_t)(1024 * 1024) + (8 * 1024)
#define AUX_ENT(id, val) \
	do { \
		elf_info[i++] = id; \
		elf_info[i++] = val; \
	} while (0)
#define EXPORTON __attribute__((visibility("default")))

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

static void *
getSymbol(const char *buf, char *sname)
{
    int i, nsyms = 0;
    Elf64_Addr symaddr = 0;
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
        /*printf("section %s type = %d flags = 0x%lx addr = 0x%lx-0x%lx, size = 0x%lx off = 0x%lx\n",
               sec_name,
               sections[i].sh_type,
               sections[i].sh_flags,
               sections[i].sh_addr,
               sections[i].sh_addr + sections[i].sh_size,
               sections[i].sh_size,
               sections[i].sh_offset);*/
    }

    for (i=0; i < nsyms; i++) {
        if (strcmp(sname, strtab + symtab[i].st_name) == 0) {
            symaddr = symtab[i].st_value;
            printf("symbol found %s = 0x%08lx\n", strtab + symtab[i].st_name, symtab[i].st_value);
            break;
        }
    }

    return (void *)symaddr;
}

ssize_t (*go_syscall_write)(int, const void *, size_t);
int (*go_syscall_open)(const char *, int, int, mode_t);
int (*runtime_lock)(void);
int (*runtime_unlock)(void);
int (*runtime_schedEnableUser)(void);
int (*runtime_entersyscall)(void);
int (*runtime_exitsyscall)(void);
int (*runtime_morestack_noctxt)(void);
int (*go2go)();
uint64_t scope_stack;
uint64_t go_stack;
uint64_t go_stack_in, go_stack_out;
uint64_t go_params[10];
uint64_t go_results[4];
uint64_t go_stack_saves[32];
unsigned long scope_fs, go_fs;
int go_what;

extern int go_start();
extern int go_end();
extern int go_hook_write();
extern int go_hook_open();
extern int go_hook_entersyscall();
extern int go_hook_exitsyscall();
extern int go_hook_morestack_noctxt();
extern void threadNow(int);
extern int arch_prctl(int, unsigned long);
#if 0
static int res;
go_hook_write() {
__asm__ volatile  (
    "mov %1, %%rax \n"
    "jmp *%%rax \n"
    : "=r"(res) /* output */
    : "r"(go_syscall_write)
    : "%rax"         /* clobbered register */
    );
}
#endif
#if 1
EXPORTON int
go_hook_test(void)
{
    int rc = 0;
    int fd;
    size_t len;
    const void *buf;
    uint64_t initialTime;
    char path[PATH_MAX];

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("set_go:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    puts("from Scope");
    //calloc(1, 1024);

    switch (go_what) {
    case 1:
        fd = (int)go_params[0];
        buf = (void *)go_params[1];
        len = (size_t)go_params[2];
        initialTime = getTime();

        if ((len <= 0) || (len >= PATH_MAX)) break;

        if (buf) {
            memmove(path, buf, len);
            path[len] = '\0';
        } else {
            scopeLog("go_hook_test:Open: null pathname", -1, CFG_LOG_ERROR);
            break;
        }

        doWrite(fd, initialTime, 1, path, len, "write", BUF, 0);
        break;

    case 2:
        if ((fd = (int)go_results[0]) == -1) break;
        buf = (char *)go_params[1];
        len = (size_t)go_params[2];

        if ((len <= 0) || (len >= PATH_MAX)) break;

        if (buf) {
            memmove(path, buf, len);
            path[len] = '\0';
        } else {
            scopeLog("go_hook_test:Open: null pathname", -1, CFG_LOG_ERROR);
            break;
        }

        doOpen(fd, path, FD, "open");
        break;

    default:
        scopeLog("go_hook_test:bad go_what", -1, CFG_LOG_ERROR);
        break;
    }



    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_hook_test:arch_prctl", -1, CFG_LOG_ERROR);
        // what to do here?
        return -1;
    }

    return rc;
}
#endif

static void
initGoHook(const char *buf)
{
    int rc;
    funchook_t *funchook;

    funchook = funchook_create();

    if (logLevel(g_log) <= CFG_LOG_DEBUG) {
        // TODO: add some mechanism to get the config'd log file path
        funchook_set_debug_file(DEFAULT_LOG_PATH);
    }

    if ((go_syscall_write = getSymbol(buf, "syscall.write")) != 0) {
        go_syscall_write += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_write, go_hook_write);
    }

    if ((go_syscall_open = getSymbol(buf, "syscall.openat")) != 0) {
        go_syscall_open += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_open, go_hook_open);
    }
/*
    if ((runtime_entersyscall = getSymbol(buf, "runtime.entersyscall")) != 0) {
        rc = funchook_prepare(funchook, (void**)&runtime_entersyscall, go_hook_entersyscall);
    }

    if ((runtime_exitsyscall = getSymbol(buf, "runtime.exitsyscall")) != 0) {
        rc = funchook_prepare(funchook, (void**)&runtime_exitsyscall, go_hook_exitsyscall);
    }

    if ((runtime_morestack_noctxt = getSymbol(buf, "runtime.morestack_noctxt")) != 0) {
        rc = funchook_prepare(funchook, (void**)&runtime_morestack_noctxt, go_hook_morestack_noctxt);
    }
*/
    //DEBUG
    runtime_lock = getSymbol(buf, "runtime.lock");
    runtime_unlock = getSymbol(buf, "runtime.unlock");
    runtime_schedEnableUser = getSymbol(buf, "runtime.schedEnableUser");
    getSymbol(buf, "os.(*File).Write");

    // hook a few Go funcs
    rc = funchook_install(funchook, 0);
    if (rc != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "ERROR: failed to install SSL_read hook. (%s)\n",
                funchook_error_message(funchook));
        scopeLog(msg, -1, CFG_LOG_ERROR);
        return;
    }
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
    char msg[1024];
    
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

            if (strcmp(sec_name, ".text") == 0) {
                go2go = (int (*)())laddr + len + 8;
                memmove(laddr + len + 8, go_start, go_end - go_start);
            }

            snprintf(msg, sizeof(msg), "%s:%d %s addr %p - %p\n",
                     __FUNCTION__, __LINE__, sec_name, laddr, laddr + len);
            scopeLog(msg, -1, CFG_LOG_DEBUG);
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
    char msg[1024];    

    if ((phead->p_vaddr == 0) || (phead->p_memsz <= 0) || (phead->p_flags == 0)) return -1;

    prot = 0;
    prot |= (phead->p_flags & PF_R) ? PROT_READ : 0;
    prot |= (phead->p_flags & PF_W) ? PROT_WRITE : 0;
    prot |= (phead->p_flags & PF_X) ? PROT_EXEC : 0;

    laddr = (char *)ROUND_DOWN(phead->p_vaddr, pgsz);
    
    snprintf(msg, sizeof(msg), "%s:%d vaddr 0x%lx size 0x%lx\n",
             __FUNCTION__, __LINE__, phead->p_vaddr, (size_t)phead->p_memsz);
    scopeLog(msg, -1, CFG_LOG_DEBUG);

    if ((addr = mmap(laddr, (size_t)phead->p_memsz,
                     prot | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                     -1, (off_t)NULL)) == MAP_FAILED) {
        scopeLog("ERROR: load_segment:mmap", -1, CFG_LOG_ERROR);
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
    int pgsz = sysconf(_SC_PAGESIZE);;
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
    unsigned long auxindex;
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
    
    // do env
    for (i = 0; env[i]; i++) {
        if ((&env[i]) && spp) {
            *spp++ = (char *)env[i];
        } else {
            scopeLog("ERROR:copy_strings: environ string is not correct", -1, CFG_LOG_ERROR);
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
            AUX_ENT(auxindex, (unsigned long)argv[1]);
            break;

        case AT_EXECFD:
            //AUX_ENT(auxindex, );
            break;

        default:
            if ((val = getauxval(auxindex)) != 0) {
                AUX_ENT(auxindex, val);
            }
        }
    }

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
        : "=r"(res) /* output */
        : "r"(start), "r"(sp)
        : "%r11"         /* clobbered register */
    );

    return 0;
}

EXPORTON int
sys_exec(const char *buf, const char *path, int argc, const char **argv, const char **env)
{
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    Elf64_Addr lastaddr;

    if (!buf || !path || !argv || !env || (argc < 1)) return -1;

    scopeLog("sys_exec type:", ehdr->e_type, CFG_LOG_DEBUG);

    lastaddr = load_elf((char *)buf);
    initGoHook(buf);

    threadNow(0);

    set_go((char *)buf, argc, argv, env, lastaddr);

    return 0;
}
