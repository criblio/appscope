#define _GNU_SOURCE
#include <sys/mman.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <signal.h>

#include "dbg.h"
#include "os.h"
#include "com.h"
#include "state.h"
#include "gocontext.h"
#include "../contrib/funchook/distorm/include/distorm.h"

#define SCOPE_STACK_SIZE (size_t)(32 * 1024)
#define funcprint devnull
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


// compile-time control for debugging
//#define funcprint sysprint
#define funcprint devnull
//#define patchprint sysprint
#define patchprint devnull

go_offsets_t g_go = {.g_to_m=48,              // 0x30
                     .m_to_tls=136,           // 0x88
                     .connReader_to_conn=0,   // 0x0
                     .conn_to_tlsState=48};   // 0x30

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

uint64_t g_glibc_guard = 0LL;
void (*go_runtime_cgocall)(void);

static void
devnull(const char* fmt, ...)
{
    return;
}

// c_str() is provided to convert a go-style string to a c-style string.
// The resulting c_str will need to be passed to free() when it is no
// longer needed.
static char *
c_str(gostring_t* go_str)
{
    if (!go_str || go_str->len <= 0) return NULL;

    char * path;
    if ((path = calloc(1, go_str->len+1)) == NULL) return NULL;
    memmove(path, go_str->str, go_str->len);
    path[go_str->len] = '\0';

    return path;
}

// If possible, we want to set GODEBUG=http2server=0
// This tells go not to upgrade servers to http2, which allows
// our http1 protocol capture stuff to do it's thing.
// We consider this temporary, because when we support http2
// it will not be necessary.
#define GO_ENV_VAR "GODEBUG"
#define GO_ENV_VALUE "http2server"
static void
setGoHttpEnvVariable(void)
{
    char *cur_val = getenv(GO_ENV_VAR);

    // If GODEBUG isn't set, try to set it to http2server=0
    if (!cur_val) {
        if (setenv(GO_ENV_VAR, GO_ENV_VALUE "=0", 1)) {
            scopeLog("ERROR: Could not set GODEBUG to http2server=0\n", -1, CFG_LOG_ERROR);
        }
        return;
    }

    // GODEBUG is set.
    // If http2server wasn't specified, let's append ",http2server=0"
    if (!strstr(cur_val, GO_ENV_VALUE)) {
        char *new_val = NULL;
        if ((asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_VALUE) == -1)) {
            scopeLog("ERROR: Could not create GODEBUG value\n", -1, CFG_LOG_ERROR);
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            scopeLog("ERROR: Could not append http2server=0 to GODEBUG\n", -1, CFG_LOG_ERROR);
        }
        if (new_val) free(new_val);
    }
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

void
initGoHook(const char *buf)
{
    int rc;
    funchook_t *funchook;

    // A go app may need to expand stacks for some C functions
    g_need_stack_expand = TRUE;

    funchook = funchook_create();

    if (logLevel(g_log) <= CFG_LOG_DEBUG) {
        // TODO: add some mechanism to get the config'd log file path
        funchook_set_debug_file(DEFAULT_LOG_PATH);
    }

    // ask Go to use HTTP 1 instead of HTTP 2 by default
    setGoHttpEnvVariable();

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
        ((fd = open(debug_file, O_CREAT|O_WRONLY|O_CLOEXEC, 0666)) != -1)) {
        dprintf(fd, "runtime.g|m=%d\n", g_go.g_to_m);
        dprintf(fd, "runtime.m|tls=%d\n", g_go.m_to_tls);
        dprintf(fd, "net/http.connReader|conn=%d\n", g_go.connReader_to_conn);
        dprintf(fd, "net/http.conn|tlsState=%d\n", g_go.conn_to_tlsState);
        close(fd);
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

/*
 * There are 3 go_switch() functions that all accomplish the 
 * same thing. They take different approaches. We want to 
 * see what effect an approach has on performance and reliability.
 * We'll likley get to one instance as we gain experience. 
 */

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

/*
 * Putting a comment here that applies to all of the
 * 'C' handlers for interposed functions.
 *
 * The go_xxx function is called by the assy code
 * in gocontext.S. It is running on a Go system
 * stack and the fs register points to an 'm' TLS
 * base address.
 *
 * We need to switch from a 'g' context to a
 * libc context in order to execute 'C' code that
 * calls libc functions. The go_switch() function
 * handles all of that. The address of a specific
 * handler is passed to go_swtich which calls the
 * handler when the appropriate switch has been
 * accopmlished.
 * 
 * Specific handlers are defined as the c_xxx functions.
 * Example, there is a c_write() that handles extracting
 * details for write operations. The address of c_write
 * is passed to go_switch.
 *
 * All handlers take a single parameter, the stack address
 * from the interposed Go function. All params are passed
 * on the stack in the Go Runtime. No params passed in
 * registers. This means return values are also on the
 * stack. It also means that we need offsets into the stack
 * in order to know how to extract values from params and
 * return values.
 */
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
