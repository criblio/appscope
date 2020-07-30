#define _GNU_SOURCE
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
#include <signal.h>
#include <pthread.h>

#include "dbg.h"
#include "os.h"
#include "com.h"
#include "state.h"
#include "gocontext.h"

#define ENABLE_SIGNAL_MASKING_IN_SYSEXEC 1
#define ENABLE_CAS_IN_SYSEXEC 1

#ifdef ENABLE_CAS_IN_SYSEXEC
#include "atomic.h"
#else
// This disables the CAS spinlocks by default.  Aint nobody got time for that.
bool
atomicCasU64(uint64_t* ptr, uint64_t oldval, uint64_t newval)
{
    return TRUE;
}
#endif

#define HEAP_SIZE (size_t)(500 * 1024)
// 1Mb + an 8kb guard
#define STACK_SIZE (size_t)(1024 * 1024) + (8 * 1024)
#define SCOPE_STACK_SIZE (size_t)(64 * 1024)
#define AUX_ENT(id, val) \
	do { \
		elf_info[i++] = id; \
		elf_info[i++] = val; \
	} while (0)
#define EXPORTON __attribute__((visibility("default")))

static __thread char *tstack;
static __thread char *gstack;
uint64_t g_glibc_guard = 0LL;
void (*go_runtime_cgocall)(void);
uint64_t scope_stack;
unsigned long scope_fs, go_fs;
uint64_t *g_currsheap = NULL;
uint64_t *g_heapend = NULL;
bool g_ongostack = FALSE;

extern void threadNow(int);
extern int arch_prctl(int, unsigned long);

// Define the interposed function seta
typedef void (*assembly_fn)(void);
extern void go_hook_write(void);
extern void go_hook_open(void);
extern void go_hook_socket(void);
extern void go_hook_accept4(void);
extern void go_hook_read(void);
extern void go_hook_close(void);
extern void go_hook_tls_read(void);
extern void go_hook_tls_write(void);

typedef struct {
    // These are constants at build time
    char *   func_name;    // name of go function
    uint64_t byte_off;     // patch offset into go function
    void *   assembly_fn;  // scope handler function (in assembly)

    // This is set at runtime.
    void *   return_addr;  // addr of where in go to resume after patch
} tap_t;

tap_t g_go_tap[] = {
    {"syscall.write",   0x9d, go_hook_write,   NULL},
    {"syscall.openat", 0x101, go_hook_open,    NULL},
    {"syscall.socket",  0x85, go_hook_socket,  NULL},
    {"syscall.accept4", 0xc1, go_hook_accept4, NULL},
    {"syscall.read",    0x9d, go_hook_read,    NULL},
    {"syscall.Close",   0x6a, go_hook_close,   NULL},
    {"net/http.(*connReader).Read",           0x183, go_hook_tls_read,   NULL},
    {"net/http.checkConnErrorWriter.Write",   0x93,  go_hook_tls_write,  NULL},
    {"TAP_TABLE_END",    0x0, NULL, NULL}
};

static void *
return_addr(assembly_fn fn)
{
    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {
        if (tap->assembly_fn == fn) return tap->return_addr;
    }

    scopeLog("FATAL ERROR: no return addr", -1, CFG_LOG_ERROR);
    exit(-1);
}

void
sysprint(const char* fmt, ...)
{
    return; // Comment this out if you want output.  =)

    // Create the string
    char* str = NULL;
    if (fmt) {
        va_list args;
        va_start(args, fmt);
        int rv = vasprintf(&str, fmt, args);
        va_end(args);
        if (rv == -1) {
            if (str) free(str);
            str = NULL;
        }
    }
    // Output the string
    if (str) {
        printf("%s", str);
        free(str);
    }
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

int
switch_tcb_to_glibc(sigset_t *mask)
{
// This disables the signal masking by default.
#ifdef ENABLE_SIGNAL_MASKING_IN_SYSEXEC
    if ((sigfillset(mask) == -1) || (sigprocmask(SIG_SETMASK, mask, NULL) == -1)) {
        scopeLog("blocking signals", -1, CFG_LOG_ERROR);
        return 0;
    }
#endif

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("arch_prctl get go", -1, CFG_LOG_ERROR);
        return 0;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("arch_prctl set scope", -1, CFG_LOG_ERROR);
        return 0;
    }
    return -1;
}

int
switch_tcb_to_go(sigset_t *mask)
{
    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("arch_prctl restore go ", -1, CFG_LOG_ERROR);
        return 0;
    }

// This disables the signal masking by default.
#ifdef ENABLE_SIGNAL_MASKING_IN_SYSEXEC
    if ((sigemptyset(mask) == -1) || (sigprocmask(SIG_SETMASK, mask, NULL) == -1)) {
        scopeLog("unblocking signals", -1, CFG_LOG_ERROR);
        return 0;
    }
#endif
    return -1;
}

inline static void *
go_switch(char *stackptr, void *cfunc, void *gfunc)
{
    sigset_t mask;
    uint64_t rc;

    // Switch the TCB and stack from Go to libc
    // Beginning of critical section.
    while (!atomicCasU64(&g_glibc_guard, 0ULL, 1ULL)) ;
    if (!switch_tcb_to_glibc(&mask)) goto out;

    if (tstack == NULL) {
        if ((tstack = mmap(NULL, SCOPE_STACK_SIZE,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, (off_t)NULL)) == MAP_FAILED) {
            scopeLog("go_write:mmap", -1, CFG_LOG_ERROR);
            exit(-1);
        }
        tstack += SCOPE_STACK_SIZE;
        printf("Scope: write stack %p\n", tstack);
    }

    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        "mov %3, %%rdi  \n"
        "callq *%4  \n"
        : "=r"(rc)                        // output
        : "m"(tstack), "m"(gstack),       // input
          "m"(stackptr), "r"(cfunc)
        :                                // clobbered register
        );

    // Switch TCB and stack back to Go
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(rc)                        // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    if (!switch_tcb_to_go(&mask)) goto out;
out:
    // End of critical section.
    atomicCasU64(&g_glibc_guard, 1ULL, 0ULL);
    return return_addr(gfunc);
}

static void
c_write(char *stackptr)
{
    char * stackaddr = stackptr + 0x50;
    uint64_t fd  = *(uint64_t *)(stackaddr + 0x8);
    uint64_t buf = *(uint64_t *)(stackaddr + 0x10);
    uint64_t rc =  *(uint64_t *)(stackaddr + 0x28);
    uint64_t initialTime = getTime();

    sysprint("Scope: write fd %ld rc %ld buf 0x%lx\n", fd, rc, buf);
    doWrite(fd, initialTime, (rc != -1), (char *)buf, rc, "go_write", BUF, 0);
}

EXPORTON void *
go_write(char *stackptr)
{
    return go_switch(stackptr, c_write, go_hook_write);
}

static void
c_open(char *stackptr)
{
    char *path = NULL;
    char * stackaddr = stackptr + 0x78;
    uint64_t fd  = *((uint64_t *)(stackaddr + 0x30));
    uint64_t buf = *((uint64_t *)(stackaddr + 0x10));
    uint64_t len = *((uint64_t *)(stackaddr + 0x18));

    if ((len <= 0) || (len >= PATH_MAX)) return;

    sysprint("Scope: open of %ld\n", fd);
    if (buf) {
        if ((path = calloc(1, len+1)) == NULL) return;
        memmove(path, (char *)buf, len);
        path[len] = '\0';
    } else {
        scopeLog("ERROR:go_open: null pathname", -1, CFG_LOG_ERROR);
        puts("Scope:ERROR:open:no path");
        if (path) free(path);
        return;
    }

    doOpen(fd, path, FD, "open");

    if (path) free(path);
}

EXPORTON void *
go_open(char *stackptr)
{
    return go_switch(stackptr, c_open, go_hook_open);
}

static void
c_close(char *stackptr)
{
    char * stackaddr = stackptr + 0x40;
    uint64_t fd  = *(uint64_t*)(stackaddr + 0x8);
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x10);

    sysprint("Scope: close of %ld\n", fd);
    doCloseAndReportFailures(fd, (rc != -1), "go_close");
}

EXPORTON void *
go_close(char *stackptr)
{
    return go_switch(stackptr, c_close, go_hook_close);
}

static void
c_read(char *stackptr)
{
    char * stackaddr = stackptr + 0x50;
    uint64_t fd    = *(uint64_t*)(stackaddr + 0x8);
    uint64_t buf   = *(uint64_t*)(stackaddr + 0x10);
    uint64_t rc    = *(uint64_t*)(stackaddr + 0x28);
    uint64_t initialTime = getTime();

    if (rc == -1) return;

    sysprint("Scope: read of %ld\n", fd);
    doRead(fd, initialTime, (rc != -1), (void*)buf, rc, "go_read", BUF, 0);
}

EXPORTON void *
go_read(char *stackptr)
{
    return go_switch(stackptr, c_read, go_hook_read);
}

static void
c_socket(char *stackptr)
{
    char * stackaddr = stackptr + 0x48;
    uint64_t domain = *(uint64_t*)(stackaddr + 0x8);  // aka family
    uint64_t type   = *(uint64_t*)(stackaddr + 0x10);
    uint64_t sd     = *(uint64_t*)(stackaddr + 0x20);

    if (sd == -1) return;

    sysprint("Scope: socket of %ld\n", sd);
    addSock(sd, type, domain);
}

EXPORTON void *
go_socket(char *stackptr)
{
    return go_switch(stackptr, c_socket, go_hook_socket);
}

static void
c_accept4(char *stackptr)
{
    char * stackaddr = stackptr + 0x70;
    struct sockaddr *addr  = *(struct sockaddr **)(stackaddr + 0x10);
    socklen_t *addrlen = *(socklen_t**)(stackaddr + 0x18);
    uint64_t sd_out = *(uint64_t*)(stackaddr + 0x28);

    sysprint("Scope: accept4 of %ld\n", sd_out);
    if (sd_out != -1) {
        doAccept(sd_out, addr, addrlen, "go_accept4");
    }
}

EXPORTON void *
go_accept4(char *stackptr)
{
    return go_switch(stackptr, c_accept4, go_hook_accept4);
}

static void
c_tls_read(char *stackptr)
{
    char * stackaddr = stackptr + 0x58;
    uint64_t connReader = *(uint64_t*)(stackaddr + 0x8);
    uint64_t buf        = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);

//  type connReader struct {
//        conn *conn
    uint64_t conn =  *(uint64_t*)connReader;
//  type conn struct {
//          server *Server
//          cancelCtx context.CancelFunc
//          rwc net.Conn
//          remoteAddr string
//          tlsState *tls.ConnectionState
    uint64_t tlsState =   *(uint64_t*)(conn + 0x30);

    // if tlsState is non-zero, then this is a tls connection
    if (tlsState != 0ULL) {
        sysprint("Scope: go_tls_read of %ld\n", -1);
        doProtocol(tlsState, -1, (void*)buf, rc, TLSRX, BUF);
    }
}

EXPORTON void *
go_tls_read(char *stackptr)
{
    return go_switch(stackptr, c_tls_read, go_hook_tls_read);
}

static void
c_tls_write(char *stackptr)
{
    char * stackaddr = stackptr + 0x58;
    uint64_t conn = *(uint64_t*)(stackaddr + 0x8);
    uint64_t buf  = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);

    uint64_t tlsState = *(uint64_t*)(conn + 0x30);

    // if tlsState is non-zero, then this is a tls connection
    if (tlsState != 0ULL) {
        sysprint("Scope: go_tls_write of %ld\n", -1);
        doProtocol(tlsState, -1, (void*)buf, rc, TLSTX, BUF);
    }
}

EXPORTON void *
go_tls_write(char *stackptr)
{
    return go_switch(stackptr, c_tls_write, go_hook_tls_write);
}

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

    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {
        void* orig_func = getSymbol(buf, tap->func_name);
        if (!orig_func) {
            scopeLog(tap->func_name, -1, CFG_LOG_ERROR);
            continue;
        }
        orig_func += tap->byte_off;
        if (funchook_prepare(funchook, (void**)&orig_func, tap->assembly_fn)) {
            scopeLog(tap->func_name, -1, CFG_LOG_ERROR);
            continue;
        }
        tap->return_addr = orig_func;
    }

    /*
     * Note: calling runtime.cgocall results in the Go error
     *       "fatal error: cgocall unavailable"
     * Calling runtime.asmcgocall does work. Possibly becasue we
     * are entering the Go func past the runtime stack check?
     * Need to investigate later.
     */
    if ((go_runtime_cgocall = getSymbol(buf, "runtime.asmcgocall")) == 0) {
        printf("ERROR: can't get the address for runtime.cgocall\n");
        exit(-1);
    }

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

    g_ongostack = TRUE;
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
