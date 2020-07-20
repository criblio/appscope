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

typedef struct go_store {
    uint64_t tcb;
    uint64_t store;
} go_store_t;

#define MAX_STORES 256
#define HEAP_SIZE (size_t)(500 * 1024)
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
int (*go_syscall_socket)(int, int, int);
int (*go_syscall_accept4)(int, void*, int, int);
ssize_t (*go_syscall_read)(int, const void *, size_t);
int (*go_syscall_close)(int);
void (*go_runtime_cgocall)(void);
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
uint64_t *g_currsheap = NULL;
uint64_t *g_heapend = NULL;
bool g_ongostack = FALSE;
go_store_t g_gostore[MAX_STORES];

extern int go_hook_write();
extern int go_hook_open();
extern int go_hook_socket();
extern int go_hook_accept4();
extern int go_hook_read();
extern int go_hook_close();
extern int go_hook_entersyscall();
extern int go_hook_exitsyscall();
extern int go_hook_morestack_noctxt();
extern void threadNow(int);
extern int arch_prctl(int, unsigned long);

EXPORTON go_store_t *
go_new_store(uint64_t tcb)
{
    int i;
    go_store_t *storep;

    for (i = 0, storep = g_gostore; i < MAX_STORES; i++) {
        if (storep[i].tcb == (uint64_t)0) {
            storep[i].tcb = tcb;
            return &storep[i];
        }
    }

    scopeLog("ERROR:go_new_store: Not good; no stack store available", -1, CFG_LOG_ERROR);
    return NULL;
}

EXPORTON go_store_t *
go_reset_store(uint64_t tcb)
{
    int i;
    go_store_t *storep;

    for (i = 0, storep = g_gostore; i < MAX_STORES; i++) {
        if (storep[i].tcb == tcb) {
            storep[i].tcb = (uint64_t)0;
            return &storep[i];
        }
    }

    scopeLog("ERROR:go_reset_Store: didn't find the stack store", -1, CFG_LOG_ERROR);
    return NULL;
}

int
regexec_wrap(const regex_t *preg, const char *string, size_t nmatch,
             regmatch_t pmatch[], int eflags)
{
    int rc;
    uint64_t res;
    go_store_t *store;
    pthread_t self;

    if (g_ongostack == TRUE) {
        self = pthread_self();
        if ((store = go_new_store(self)) == NULL) return REG_NOMATCH;

        // RDI, RSI, RDX, RCX, R8, R9
        __asm__ volatile (
            "mov %1, %%r11 \n"
            "mov %%rsp, 0x8(%%r11)  \n"          // save the current stack
            "lea scope_stack(%%rip), %%r11 \n"
            "mov (%%r11), %%rsp \n"              // switch the stack
            "mov %2, %%rdi  \n"
            "mov %3, %%rsi  \n"
            "mov %4, %%rdx  \n"
            "mov %5, %%rcx  \n"
            "mov %6, %%r8d  \n"
            "callq pcre2_regexec  \n"
            : "=r"(rc)                        // output
            : "r"(store), "r"(preg), "r"(string),
              "r"(nmatch), "r"(pmatch), "r"(eflags)   // input
            : "%r11"                          // clobbered register
            );

        __asm__ volatile (
            "mov %1, %%r11 \n"
            "mov 0x8(%%r11), %%rsp \n"        // switch the stack
            : "=r"(res)                       // output
            : "r"(store)                      // inputs
            : "%r11"                          // clobbered register
            );

        go_reset_store(self);
    } else {
        rc = pcre2_regexec(preg, string, nmatch, pmatch, eflags);
    }

    return rc;
}

EXPORTON size_t
go_write(char *stackaddr)
{
    sigset_t mask;
    uint64_t fd  = *(uint64_t *)(stackaddr + 0x8);
    uint64_t buf = *(uint64_t *)(stackaddr + 0x10);
    //uint64_t len = *(uint64_t *)(stackaddr + 0x18);
    uint64_t rc  = *(uint64_t *)(stackaddr + 0x28); // <- should be 0x20, we thought?
    uint64_t initialTime = getTime();


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_write:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_write:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_write_test:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: write of %ld\n", fd);
    doWrite(fd, initialTime, (rc != -1), (char *)buf, rc, "go_write", BUF, 0);

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_write:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_write:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return rc;

}

EXPORTON size_t
go_open(uint64_t *stackaddr)
{
    char *path = NULL;
    size_t rc = 0;
    sigset_t mask;
    uint64_t fd = *((uint64_t *)((char *)(stackaddr) + 0x30));
    uint64_t buf = *((uint64_t *)((char *)(stackaddr) + 0x10));
    uint64_t len = *((uint64_t *)((char *)(stackaddr) + 0x18));

    if ((len <= 0) || (len >= PATH_MAX)) return -1;


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_open:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_open:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_open_test:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: open of %ld\n", rc);
    if (buf) {
        if ((path = calloc(1, len+1)) == NULL) return -1;
        memmove(path, (char *)buf, len);
        path[len] = '\0';
    } else {
        scopeLog("ERROR:go_open: null pathname", -1, CFG_LOG_ERROR);
        puts("Scope:ERROR:open:no path");
        if (path) free(path);
        return -1;
    }

    doOpen(fd, path, FD, "open");

    if (path) free(path);

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_open:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_open:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return rc;

}

EXPORTON int
go_socket(char *stackaddr)
{
    sigset_t mask;

    uint64_t domain = *(uint64_t*)(stackaddr + 0x8);  // aka family
    uint64_t type   = *(uint64_t*)(stackaddr + 0x10);
    //uint64_t proto  = *(uint64_t*)(stackaddr + 0x18);
    uint64_t sd     = *(uint64_t*)(stackaddr + 0x20);

    if (sd == -1) return -1;


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_socket:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_socket:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_socket_test:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: socket of %ld\n", sd);
    addSock(sd, type, domain);

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_socket:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_socket:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return sd;
}

EXPORTON int
go_accept4(char *stackaddr)
{
    sigset_t mask;
    //uint64_t sd_in   = *(uint64_t*)(stackaddr + 0x08);
    struct sockaddr *addr  = *(struct sockaddr **)(stackaddr + 0x10);
    socklen_t *addrlen = *(socklen_t**)(stackaddr + 0x18);
    //uint64_t flags   = *(uint64_t*)(stackaddr + 0x20);
    uint64_t sd_out  = *(uint64_t*)(stackaddr + 0x28);


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_accept4:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_accept4:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_accept4:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: accept4 of %ld\n", sd_out);
    if (sd_out != -1) {
        doAccept(sd_out, addr, addrlen, "go_accept4");
    }

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_accept4:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_accept4:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return sd_out;
}

EXPORTON ssize_t
go_read(char *stackaddr)
{
    sigset_t mask;
    uint64_t fd    = *(uint64_t*)(stackaddr + 0x8);
    uint64_t buf   = *(uint64_t*)(stackaddr + 0x10);
    //uint64_t len   = *(uint64_t*)(stackaddr + 0x18);
    uint64_t rc    = *(uint64_t*)(stackaddr + 0x28); // <- should be 0x20, we thought?
    uint64_t initialTime = getTime();

    if (rc == -1) return -1;


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_read:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_read:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_read_test:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: read of %ld\n", fd);
    doRead(fd, initialTime, (rc != -1), (void*)buf, rc, "go_read", BUF, 0);

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_read:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_read:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return rc;
}


EXPORTON int
go_close(char *stackaddr)
{
    sigset_t mask;
    uint64_t fd  = *(uint64_t*)(stackaddr + 0x8);
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x10);


    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_close:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_GET_FS, (unsigned long)&go_fs) == -1) {
        scopeLog("go_close:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_close:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    printf("Scope: close of %ld\n", fd);
    doCloseAndReportFailures(fd, (rc != -1), "go_close");

    if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
        scopeLog("go_close:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("go_close:blocking signals", -1, CFG_LOG_ERROR);
        return -1;
    }

    return rc;
}


EXPORTON int
go_hook_test(void)
{
    int rc = 0;
    int fd;
    size_t len;
    //go_funcs_t gofuncs;
    const void *buf;
    uint64_t initialTime;
    char path[PATH_MAX];

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("go_hook_test:arch_prctl", -1, CFG_LOG_ERROR);
        return -1;
    }

    puts("from Scope");
    //calloc(1, 1024);

    switch (go_what) {
    case SYSCALL_WRITE:
        fd = (int)go_params[0];
        buf = (void *)go_params[1];
        len = (size_t)go_params[2];
        initialTime = getTime();

        if ((len <= 0) || (len >= PATH_MAX)) break;

        if (buf) {
            memmove(path, buf, len);
            path[len] = '\0';
        } else {
            scopeLog("go_hook_test:write: null pathname", -1, CFG_LOG_ERROR);
            break;
        }

        doWrite(fd, initialTime, 1, path, len, "write", BUF, 0);
        break;

    case SYSCALL_OPEN:
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

    case SYSCALL_READ:
        fd = (int)go_params[0];
        buf = (void *)go_params[1];
        len = (size_t)go_params[2];
        initialTime = getTime();

        char msg[128];
        snprintf(msg, sizeof(msg), "%s:%d fd %d len %ld",
                 __FUNCTION__, __LINE__, fd, len);
        scopeLog(msg, -1, CFG_LOG_ERROR);

        doRead(fd, initialTime, 1, buf, len, "read", BUF, 0);
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

    if ((go_syscall_socket = getSymbol(buf, "syscall.socket")) != 0) {
        go_syscall_socket += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_socket, go_hook_socket);
    }

    if ((go_syscall_accept4 = getSymbol(buf, "syscall.accept4")) != 0) {
        go_syscall_accept4 += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_accept4, go_hook_accept4);
    }

    if ((go_syscall_read = getSymbol(buf, "syscall.read")) != 0) {
        go_syscall_read += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_read, go_hook_read);
    }

    if ((go_syscall_close = getSymbol(buf, "syscall.Close")) != 0) {
        go_syscall_close += 19;
        rc = funchook_prepare(funchook, (void**)&go_syscall_close, go_hook_close);
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

    memset(g_gostore, 0, sizeof(go_store_t) * MAX_STORES);

    threadNow(0);

    set_go((char *)buf, argc, argv, env, lastaddr);

    return 0;
}
