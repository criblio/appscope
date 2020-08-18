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
#include "../contrib/funchook/distorm/include/distorm.h"


typedef struct {            // Structure                     Field              Offset
    int g_to_m;             // "runtime.g"                   "m"                "48"
    int m_to_tls;           // "runtime.m"                   "tls"              "136"
    int connReader_to_conn; // "net/http.connReader"         "conn"             "0"
    int conn_to_tlsState;   // "net/http.conn"               "tlsState"         "48"
} go_offsets_t;

go_offsets_t g_go = {.g_to_m=48,              // 0x30
                     .m_to_tls=136,           // 0x88
                     .connReader_to_conn=0,   // 0x0
                     .conn_to_tlsState=48};   // 0x30

//#define ENABLE_SIGNAL_MASKING_IN_SYSEXEC 1
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
#define SCOPE_STACK_SIZE (size_t)(32 * 1024)

#define AUX_ENT(id, val)                        \
	do { \
		*elf_info = (Elf64_Addr)id; \
        elf_info++;     \
		*elf_info = (Elf64_Addr)val; \
        elf_info++;     \
	} while (0)
#define EXPORTON __attribute__((visibility("default")))

uint64_t g_glibc_guard = 0LL;
void (*go_runtime_cgocall)(void);
uint64_t scope_stack;
unsigned long scope_fs;
uint64_t *g_currsheap = NULL;
uint64_t *g_heapend = NULL;
unsigned char *g_text_addr = NULL;
uint64_t g_text_len = -1;

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


// Go strings are not null delimited like c strings.
// Instead, go strings have structure which includes a length field.
typedef struct {
    char* str;  // 0x0 offset
    int   len;  // 0x8 offset
} gostring_t;

// c_str() is provided to convert a go-style string to a c-style string.
// The resulting c_str will need to be passed to free() when it is no
// longer needed.
char*
c_str(gostring_t* go_str)
{
    if (!go_str || go_str->len <= 0) return NULL;

    char * path;
    if ((path = calloc(1, go_str->len+1)) == NULL) return NULL;
    memmove(path, go_str->str, go_str->len);
    path[go_str->len] = '\0';

    return path;
}


typedef struct {
    // These are constants at build time
    char *   func_name;    // name of go function
    void *   assembly_fn;  // scope handler function (in assembly)

    // These are set at runtime.
    void *   return_addr;  // addr of where in go to resume after patch
    uint32_t frame_size;   // size of go stack frame
} tap_t;

tap_t g_go_tap[] = {
    {"syscall.write",                         go_hook_write,       NULL, 0},
    {"syscall.openat",                        go_hook_open,        NULL, 0},
    {"syscall.socket",                        go_hook_socket,      NULL, 0},
    {"syscall.accept4",                       go_hook_accept4,     NULL, 0},
    {"syscall.read",                          go_hook_read,        NULL, 0},
    {"syscall.Close",                         go_hook_close,       NULL, 0},
    {"net/http.(*connReader).Read",           go_hook_tls_read,    NULL, 0},
    {"net/http.checkConnErrorWriter.Write",   go_hook_tls_write,   NULL, 0},
    {"TAP_TABLE_END", NULL, NULL, 0}
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

static uint32_t
frame_size(assembly_fn fn)
{
    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {
        if (tap->assembly_fn == fn) return tap->frame_size;
    }

    scopeLog("FATAL ERROR: no frame size", -1, CFG_LOG_ERROR);
    exit(-1);
}

////////////////
// compile-time control for debugging
////////////////
//#define funcprint sysprint
#define funcprint devnull
//#define patchprint sysprint
#define patchprint devnull

static void
devnull(const char* fmt, ...)
{
    return;
}

static void
sysprint(const char* fmt, ...)
{
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
        scopeLog(str, -1, CFG_LOG_DEBUG);
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
            sysprint("symbol found %s = 0x%08lx\n", strtab + symtab[i].st_name, symtab[i].st_value);
            break;
        }
    }

    return (void *)symaddr;
}

/*
 * Switch from the Go system stack to a single libc stack
 * Switch from the 'm' TCB to the libc TCB
 * Call the C handler
 * Switch 'g' and 'm' state back
 * Return to the calling 'g'
 *
 * This implementation of go_switch uses
 * a single libc stack for all 'g'.
 * As such it must create a critical section around the
 * stack switch in order serialize acess to the libc stack.
 *
 * The signal masking is not needed. It's left in place and
 * disabled so that it can be compiled in to test as needed.
 *
 * go_switch_one_stack
 */
inline static void *
go_switch_one_stack(char *stackptr, void *cfunc, void *gfunc)
{
    uint64_t rc;
    unsigned long go_tls, *go_ptr;
    char *gstack;
    char *go_g = NULL, *go_m = NULL;

// This disables the signal masking by default.
#ifdef ENABLE_SIGNAL_MASKING_IN_SYSEXEC
    sigset_t mask;
    if ((sigfillset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("blocking signals", -1, CFG_LOG_ERROR);
        goto out;
    }
#endif

    // Get the Go routine's struct g
    __asm__ volatile (
        "mov %%fs:0xfffffffffffffff8, %%r11 \n"
        "mov %%r11, %1  \n"
        : "=r"(rc)                        // output
        : "m"(go_g)                       // inputs
        :                                 // clobbered register
        );

    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("arch_prctl set scope", -1, CFG_LOG_ERROR);
        goto out;
    }

    uint32_t frame_offset = frame_size(gfunc);
    if (!frame_offset) goto out;
    stackptr += frame_offset;

    // Beginning of critical section.
    while (!atomicCasU64(&g_glibc_guard, 0ULL, 1ULL)) ;
    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        "mov %3, %%rdi  \n"
        "callq *%4  \n"
        : "=r"(rc)                         // output
        : "m"(scope_stack), "m"(gstack),   // input
          "r"(stackptr), "r"(cfunc)
        :                                  // clobbered register
        );

    // Switch stack back to Go
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(rc)                        // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    // End of critical section.
    atomicCasU64(&g_glibc_guard, 1ULL, 0ULL);

    // get struct m from g
    go_ptr = (unsigned long *)(go_g + g_go.g_to_m);
    go_tls = *go_ptr;
    go_m = (char *)go_tls;
    go_tls = (unsigned long)(go_m + g_go.m_to_tls);

    if (arch_prctl(ARCH_SET_FS, go_tls) == -1) {
        scopeLog("arch_prctl restore go ", -1, CFG_LOG_ERROR);
        goto out;
    }

// This disables the signal masking by default.
#ifdef ENABLE_SIGNAL_MASKING_IN_SYSEXEC
    if ((sigemptyset(&mask) == -1) || (sigprocmask(SIG_SETMASK, &mask, NULL) == -1)) {
        scopeLog("unblocking signals", -1, CFG_LOG_ERROR);
        goto out;
    }
#endif

out:
    return return_addr(gfunc);
}

/*
 * Switch from the Go system stack to a new libc stack
 * Switch from the 'm' TCB to the libc TCB
 * Call the C handler
 * Switch 'g' and 'm' state back
 * Return to the calling 'g'
 *
 * This implementation of go_switch creates
 * a new libc stack for every calling 'g'.
 * As such no signal mask or guard is required.
 *
 *go_switch_new_stack
 */
inline static void *
go_switch_new_stack(char *stackptr, void *cfunc, void *gfunc)
{
    uint64_t rc;
    unsigned long go_tls, *go_ptr;
    char *gstack;
    char *tstack = NULL, *sstack = NULL;
    char *go_g = NULL, *go_m = NULL;

    // Get the Go routine's struct g
    __asm__ volatile (
        "mov %%fs:0xfffffffffffffff8, %%r11 \n"
        "mov %%r11, %1  \n"
        : "=r"(rc)                        // output
        : "m"(go_g)                       // inputs
        :                                 // clobbered register
        );

    // Swtich to the libc TCB
    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("arch_prctl set scope", -1, CFG_LOG_ERROR);
        goto out;
    }

    // Get a libc stack
    if (tstack == NULL) {
        if ((sstack = mmap(NULL, SCOPE_STACK_SIZE,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, (off_t)NULL)) == MAP_FAILED) {
            scopeLog("go_write:mmap", -1, CFG_LOG_ERROR);
            exit(-1);
        }
        tstack = sstack + SCOPE_STACK_SIZE;
    }

    uint32_t frame_offset = frame_size(gfunc);
    if (!frame_offset) goto out;
    stackptr += frame_offset;

    // save the 'g' stack, switch to the tstack, call the C handler
    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        "mov %3, %%rdi  \n"
        "callq *%4  \n"
        : "=r"(rc)                    // output
        : "m"(tstack), "m"(gstack),   // input
          "r"(stackptr), "r"(cfunc)
        :                            // clobbered register
        );

    // Switch stack back to the 'g' stack
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(rc)                        // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    // get struct m from g and pull out the TLS from 'm'
    go_ptr = (unsigned long *)(go_g + g_go.g_to_m);
    go_tls = *go_ptr;
    go_m = (char *)go_tls;
    go_tls = (unsigned long)(go_m + g_go.m_to_tls);

    // Switch back to the 'm' TLS
    if (arch_prctl(ARCH_SET_FS, go_tls) == -1) {
        scopeLog("arch_prctl restore go ", -1, CFG_LOG_ERROR);
        goto out;
    }

out:
    if (munmap(sstack, SCOPE_STACK_SIZE) != 0) {
        scopeLog("munmap", -1, CFG_LOG_ERROR);
    }
    return return_addr(gfunc);
}

/*
 * Switch from the 'm' TCB to the libc TCB
 * Call the C handler
 * Switch 'm' state back
 * Return to the calling 'g'
 *
 * This implementation of go_switch does not
 * create a new libc stack. All C functions
 * are run on the Go system stack. There are
 * several C functions that require a larger
 * stack than that provided by the Go system
 * stack, namely specifici pcre2 functions.
 * We use wrapper functions to give those
 * functions a larger stack.
 *
 * go_switch_no_stack
 */
inline static void *
go_switch(char *stackptr, void *cfunc, void *gfunc)
{
    uint64_t rc;
    unsigned long go_tls, *go_ptr;
    char *go_g = NULL, *go_m = NULL;

    // Get the Go routine's struct g
    __asm__ volatile (
        "mov %%fs:0xfffffffffffffff8, %%r11 \n"
        "mov %%r11, %1  \n"
        : "=r"(rc)                        // output
        : "m"(go_g)                       // inputs
        :                                 // clobbered register
        );

    // Switch to the libc TCB
    if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
        scopeLog("arch_prctl set scope", -1, CFG_LOG_ERROR);
        goto out;
    }

    uint32_t frame_offset = frame_size(gfunc);
    if (!frame_offset) goto out;
    stackptr += frame_offset;

    // call the C handler
    __asm__ volatile (
        "mov %1, %%rdi  \n"
        "callq *%2  \n"
        : "=r"(rc)                    // output
        : "r"(stackptr), "r"(cfunc)   // inputs
        :                             // clobbered register
        );

    // get struct m from g and pull out the TLS from 'm'
    go_ptr = (unsigned long *)(go_g + g_go.g_to_m);
    go_tls = *go_ptr;
    go_m = (char *)go_tls;
    go_tls = (unsigned long)(go_m + g_go.m_to_tls);

    // Switch back to the 'm' TLS
    if (arch_prctl(ARCH_SET_FS, go_tls) == -1) {
        scopeLog("arch_prctl restore go ", -1, CFG_LOG_ERROR);
        goto out;
    }

out:
    return return_addr(gfunc);
}

static void
c_write(char *stackaddr)
{
    uint64_t fd  = *(uint64_t *)(stackaddr + 0x8);
    uint64_t buf = *(uint64_t *)(stackaddr + 0x10);
    uint64_t rc =  *(uint64_t *)(stackaddr + 0x28);
    uint64_t initialTime = getTime();

    funcprint("Scope: write fd %ld rc %ld buf 0x%lx\n", fd, rc, buf);
    doWrite(fd, initialTime, (rc != -1), (char *)buf, rc, "go_write", BUF, 0);
}

EXPORTON void *
go_write(char *stackptr)
{
    return go_switch(stackptr, c_write, go_hook_write);
}

static void
c_open(char *stackaddr)
{
    uint64_t fd  = *((uint64_t *)(stackaddr + 0x30));
    // The gostring_t* here has an implicit len field at stackaddr + 0x18
    char *path = c_str((gostring_t*)(stackaddr + 0x10));

    if (!path) {
        scopeLog("ERROR:go_open: null pathname", -1, CFG_LOG_ERROR);
        puts("Scope:ERROR:open:no path");
        return;
    }

    funcprint("Scope: open of %ld\n", fd);
    doOpen(fd, path, FD, "open");

    if (path) free(path);
}

EXPORTON void *
go_open(char *stackptr)
{
    return go_switch(stackptr, c_open, go_hook_open);
}

static void
c_close(char *stackaddr)
{
    uint64_t fd  = *(uint64_t*)(stackaddr + 0x8);
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x10);

    funcprint("Scope: close of %ld\n", fd);
    doCloseAndReportFailures(fd, (rc != -1), "go_close");
}

EXPORTON void *
go_close(char *stackptr)
{
    return go_switch(stackptr, c_close, go_hook_close);
}

static void
c_read(char *stackaddr)
{
    uint64_t fd    = *(uint64_t*)(stackaddr + 0x8);
    uint64_t buf   = *(uint64_t*)(stackaddr + 0x10);
    uint64_t rc    = *(uint64_t*)(stackaddr + 0x28);
    uint64_t initialTime = getTime();

    if (rc == -1) return;

    funcprint("Scope: read of %ld\n", fd);
    doRead(fd, initialTime, (rc != -1), (void*)buf, rc, "go_read", BUF, 0);
}

EXPORTON void *
go_read(char *stackptr)
{
    return go_switch(stackptr, c_read, go_hook_read);
}

static void
c_socket(char *stackaddr)
{
    uint64_t domain = *(uint64_t*)(stackaddr + 0x8);  // aka family
    uint64_t type   = *(uint64_t*)(stackaddr + 0x10);
    uint64_t sd     = *(uint64_t*)(stackaddr + 0x20);

    if (sd == -1) return;

    funcprint("Scope: socket of %ld\n", sd);
    addSock(sd, type, domain);
}

EXPORTON void *
go_socket(char *stackptr)
{
    return go_switch(stackptr, c_socket, go_hook_socket);
}

static void
c_accept4(char *stackaddr)
{
    struct sockaddr *addr  = *(struct sockaddr **)(stackaddr + 0x10);
    socklen_t *addrlen = *(socklen_t**)(stackaddr + 0x18);
    uint64_t sd_out = *(uint64_t*)(stackaddr + 0x28);

    funcprint("Scope: accept4 of %ld\n", sd_out);
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
c_tls_read(char *stackaddr)
{
    uint64_t connReader = *(uint64_t*)(stackaddr + 0x8);
    if (!connReader) return;   // protect from dereferencing null
    uint64_t buf        = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);

//  type connReader struct {
//        conn *conn
    uint64_t conn =  *(uint64_t*)(connReader + g_go.connReader_to_conn);
    if (!conn) return;         // protect from dereferencing null
//  type conn struct {
//          server *Server
//          cancelCtx context.CancelFunc
//          rwc net.Conn
//          remoteAddr string
//          tlsState *tls.ConnectionState
    uint64_t tlsState =   *(uint64_t*)(conn + g_go.conn_to_tlsState);

    // if tlsState is non-zero, then this is a tls connection
    if (tlsState != 0ULL) {
        funcprint("Scope: go_tls_read of %ld\n", -1);
        doProtocol(tlsState, -1, (void*)buf, rc, TLSRX, BUF);
    }
}

EXPORTON void *
go_tls_read(char *stackptr)
{
    return go_switch(stackptr, c_tls_read, go_hook_tls_read);
}

static void
c_tls_write(char *stackaddr)
{
    uint64_t conn = *(uint64_t*)(stackaddr + 0x8);
    if (!conn) return;         // protect from dereferencing null
    uint64_t buf  = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);

    uint64_t tlsState = *(uint64_t*)(conn + g_go.conn_to_tlsState);

    // if tlsState is non-zero, then this is a tls connection
    if (tlsState != 0ULL) {
        funcprint("Scope: go_tls_write of %ld\n", -1);
        doProtocol(tlsState, -1, (void*)buf, rc, TLSTX, BUF);
    }
}

EXPORTON void *
go_tls_write(char *stackptr)
{
    return go_switch(stackptr, c_tls_write, go_hook_tls_write);
}



static bool
looks_like_first_inst_of_go_func(_DecodedInst* asm_inst)
{
    return (!strcmp((const char*)asm_inst->mnemonic.p, "MOV") &&
            !strcmp((const char*)asm_inst->operands.p, "RCX, [FS:0xfffffffffffffff8]"));
}

static uint32_t
add_argument(_DecodedInst* asm_inst)
{
    if (!asm_inst) return 0;

    // In this example, add_argument is 0x58:
    // 000000000063a083 (04) 4883c458                 ADD RSP, 0x58
    // 000000000063a087 (01) c3                       RET
    if (asm_inst->size == 4) {
        unsigned char* inst_addr = (unsigned char*)asm_inst->offset;
        return ((unsigned char*)inst_addr)[3];
    }

    // In this example, add_argument is 0x80:
    // 00000000004a9cc9 (07) 4881c480000000           ADD RSP, 0x80
    // 00000000004a9cd0 (01) c3                       RET
    if (asm_inst->size == 7) {
        unsigned char* inst_addr = (unsigned char*)asm_inst->offset;
        // x86_64 is little-endian.
        return inst_addr[3] +
              (inst_addr[4] << 8 ) +
              (inst_addr[5] << 16) +
              (inst_addr[6] << 24);
    }

    return 0;
}

static void
patch_return_addrs(funchook_t *funchook,
                   _DecodedInst* asm_inst, unsigned int asm_count, tap_t* tap)
{
    if (!funchook || !asm_inst || !asm_count || !tap) return;

    int i;
    for (i=0; i<asm_count; i++) {

        // We've observed that the first instruction in every goroutine
        // is the same (it retrieves a pointer to the goroutine.)
        // We're checking for it here to make sure things are coherent.
        if (i == 0 && !looks_like_first_inst_of_go_func(&asm_inst[i])) break;

        // Stop when it looks like we've hit another goroutine
        if (i > 0 && (looks_like_first_inst_of_go_func(&asm_inst[i]) ||
                  (!strcmp((const char*)asm_inst[i].mnemonic.p, "INT 3") &&
                  asm_inst[i].size == 1 ))) {
            break;
        }

        patchprint("%0*lx (%02d) %-24s %s%s%s\n",
               16,
               asm_inst[i].offset,
               asm_inst[i].size,
               (char*)asm_inst[i].instructionHex.p,
               (char*)asm_inst[i].mnemonic.p,
               asm_inst[i].operands.length != 0 ? " " : "",
               (char*)asm_inst[i].operands.p);

        // If the current instruction is a RET
        // we want to patch the ADD immediately before it.
        uint32_t add_arg = 0;
        if ((!strcmp((const char*)asm_inst[i].mnemonic.p, "RET") &&
             asm_inst[i].size == 1) &&
             !strcmp((const char*)asm_inst[i-1].mnemonic.p, "ADD") &&
             (add_arg = add_argument(&asm_inst[i-1]))) {

            void *pre_patch_addr = (void*)asm_inst[i-1].offset;
            void *patch_addr = (void*)asm_inst[i-1].offset;

            // all add_arg values within a function should be the same
            if (tap->frame_size && (tap->frame_size != add_arg)) {
                patchprint("aborting patch of 0x%p due to mismatched frame size 0x%x\n", pre_patch_addr, add_arg);
                break;
            }

            if (funchook_prepare(funchook, (void**)&patch_addr, tap->assembly_fn)) {
                patchprint("failed to patch 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
                continue;
            }
            patchprint("patched 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
            tap->return_addr = patch_addr;
            tap->frame_size = add_arg;
        }
    }
    patchprint("\n\n");
}


static void
initGoHook(const char *buf)
{
    int rc;
    funchook_t *funchook;

    int (*ni_open)(const char *pathname, int flags, mode_t mode);
    ni_open = dlsym(RTLD_NEXT, "open");
    int (*ni_close)(int fd);
    ni_close = dlsym(RTLD_NEXT, "close");
    if (!ni_open || !ni_close) return;

    // A go app may need to expand stacks for some C functions
    g_need_stack_expand = TRUE;

    funchook = funchook_create();

    if (logLevel(g_log) <= CFG_LOG_DEBUG) {
        // TODO: add some mechanism to get the config'd log file path
        funchook_set_debug_file(DEFAULT_LOG_PATH);
    }

    gostring_t* go_ver; // There is an implicit len field at go_ver + 0x8
    char* go_runtime_version = NULL;
    if (!(go_ver= getSymbol(buf, "runtime.buildVersion")) ||
        !(go_runtime_version = c_str(go_ver))) {
        sysprint("ERROR: can't get the address for runtime.buildVersion\n");
        return;
    }
    sysprint("go_runtime_version = %s\n", go_runtime_version);

    // go 1.8 has a different m_to_tls offset than other supported versions.
    if (strstr(go_runtime_version, "go1.8.")) g_go.m_to_tls = 96; // 0x60


    // This creates a file specified by test/testContainers/go/test_go.sh
    // and used by test/testContainers/go/test_go_struct.sh.
    //
    // Why?  To test structure offsets in go that might change.  (like in 1.8
    // immediately above)
    char* debug_file;
    int fd;
    if ((debug_file = getenv("SCOPE_GO_STRUCT_PATH")) &&
        ((fd = ni_open(debug_file, O_CREAT|O_WRONLY|O_CLOEXEC, 0666)) != -1)) {
        dprintf(fd, "runtime.g|m=%d\n", g_go.g_to_m);
        dprintf(fd, "runtime.m|tls=%d\n", g_go.m_to_tls);
        dprintf(fd, "net/http.connReader|conn=%d\n", g_go.connReader_to_conn);
        dprintf(fd, "net/http.conn|tlsState=%d\n", g_go.conn_to_tlsState);
        ni_close(fd);
    }


    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {
        void* orig_func = getSymbol(buf, tap->func_name);
        if (!orig_func) {
            sysprint("ERROR: can't get the address for %s\n", tap->func_name);
            continue;
        }

        const int MAX_INST = 250;
        unsigned int asm_count = 0;
        _DecodedInst asm_inst[MAX_INST];
        uint64_t offset_into_txt = (uint64_t)orig_func - (uint64_t)g_text_addr;
        int rc = distorm_decode((uint64_t)orig_func, orig_func,
                                 g_text_len - offset_into_txt,
                                 Decode64Bits, asm_inst, MAX_INST, &asm_count);
        if (rc == DECRES_INPUTERR) {
            continue;
        }

        patchprint ("********************************\n");
        patchprint ("** %s  %s **\n", go_runtime_version, tap->func_name);
        patchprint ("********************************\n");

        patch_return_addrs(funchook, asm_inst, asm_count, tap);
    }

    /*
     * Note: calling runtime.cgocall results in the Go error
     *       "fatal error: cgocall unavailable"
     * Calling runtime.asmcgocall does work. Possibly becasue we
     * are entering the Go func past the runtime stack check?
     * Need to investigate later.
     */
    if ((go_runtime_cgocall = getSymbol(buf, "runtime.asmcgocall")) == 0) {
        sysprint("ERROR: can't get the address for runtime.cgocall\n");
        exit(-1);
    }

    // hook a few Go funcs
    rc = funchook_install(funchook, 0);
    if (rc != 0) {
        sysprint("ERROR: funchook_install failed.  (%s)\n",
                funchook_error_message(funchook));
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

            sysprint("%s:%d %s addr %p - %p\n",
                     __FUNCTION__, __LINE__, sec_name, laddr, laddr + len);

            if (!strcmp(sec_name, ".text")) {
                g_text_addr = (unsigned char*)laddr;
                g_text_len = len;
            }
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
    
    // do env
    // Note that the env array on the stack are pointers to strings.
    // We are pointing to the strings provided from the executable,
    // re-using them for the new app we are starting.
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
