#define _GNU_SOURCE
#include <sys/mman.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <signal.h>
#include <pthread.h>

#include "com.h"
#include "dbg.h"
#include "gocontext.h"
#include "linklist.h"
#include "os.h"
#include "state.h"
#include "utils.h"
#include "fn.h"
#include "capstone/capstone.h"

#define SCOPE_STACK_SIZE (size_t)(32 * 1024)
//#define ENABLE_SIGNAL_MASKING_IN_SYSEXEC 1
#define ENABLE_CAS_IN_SYSEXEC 1

#define SWITCH_ENV "SCOPE_SWITCH"
#define SWITCH_USE_NO_THREAD "no_thread"
#define SWITCH_USE_THREAD "thread"

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
#define NEEDEVNULL 1
#define funcprint sysprint
//#define funcprint devnull
//#define patchprint sysprint
#define patchprint devnull

#define UNDEF_OFFSET (-1)
go_offsets_t g_go = {.g_to_m=48,                   // 0x30
                     .m_to_tls=136,                // 0x88
                     .connReader_to_conn=0,        // 0x00
                     .persistConn_to_conn=80,      // 0x50
                     .persistConn_to_bufrd=104,    // 0x68
                     .iface_data=8,                // 0x08
                     .netfd_to_pd=0,               // 0x00
                     .pd_to_fd=16,                 // 0x10
                     .netfd_to_sysfd=UNDEF_OFFSET, // 0x10 (defined for go1.8)
                     .bufrd_to_buf=0,              // 0x00
                     .conn_to_rwc=16,              // 0x10
                     .conn_to_tlsState=48,         // 0x30
                     .persistConn_to_tlsState=96}; // 0x60

tap_t g_go_tap[] = {
    {"syscall.write",                        go_hook_write,        NULL, 0},
    {"syscall.openat",                       go_hook_open,         NULL, 0},
    {"syscall.unlinkat",                     go_hook_unlinkat,     NULL, 0},
    {"syscall.Getdents",                     go_hook_getdents,     NULL, 0},
    {"syscall.socket",                       go_hook_socket,       NULL, 0},
    {"syscall.accept4",                      go_hook_accept4,      NULL, 0},
    {"syscall.read",                         go_hook_read,         NULL, 0},
    {"syscall.Close",                        go_hook_close,        NULL, 0},
    //{"net/http.(*connReader).Read",          go_hook_tls_read,     NULL, 0},
    //{"net/http.checkConnErrorWriter.Write",  go_hook_tls_write,    NULL, 0},
    //{"net/http.(*persistConn).readResponse", go_hook_readResponse, NULL, 0},
    {"net/http.persistConnWriter.Write",     go_hook_pc_write,     NULL, 0},
    {"runtime.exit",                         go_hook_exit,         NULL, 0},
    {"runtime.dieFromSignal",                go_hook_die,          NULL, 0},
    {"TAP_TABLE_END", NULL, NULL, 0}
};

uint64_t g_glibc_guard = 0LL;
uint64_t g_go_static = 0LL;
static list_t *g_threadlist;
static void *g_stack;
static bool g_switch_thread;

void (*go_runtime_cgocall)(void);
void (*go_runtime_cgocall_regs)(void);

#if NEEDEVNULL > 0
static void
devnull(const char* fmt, ...)
{
    return;
}
#endif

// c_str() is provided to convert a go-style string to a c-style string.
// The resulting c_str will need to be passed to free() when it is no
// longer needed.
static char *
c_str(gostring_t *go_str)
{
    if (!go_str || go_str->len <= 0) return NULL;

    char *path;
    if ((path = calloc(1, go_str->len+1)) == NULL) return NULL;
    memmove(path, go_str->str, go_str->len);
    path[go_str->len] = '\0';

    return path;
}

static bool
looks_like_first_inst_of_go_func(cs_insn* asm_inst)
{
        patchprint("%0*lx (%02d) %-24s %s %s\n",
                   16,
                   asm_inst->address,
                   asm_inst->size,
                   (char*)asm_inst->bytes,
                   (char*)asm_inst->mnemonic,
                   (char*)asm_inst->op_str);

    return (!strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !strcmp((const char*)asm_inst->op_str, "rcx, qword ptr fs:[0xfffffffffffffff8]"))
        || // -buildmode=pie compiles to this:
        (!strcmp((const char*)asm_inst->mnemonic, "mov") &&
         !strcmp((const char*)asm_inst->op_str, "rcx, -8"))
        || // Go 17
        (!strcmp((const char*)asm_inst->mnemonic, "cmp") &&
         !strcmp((const char*)asm_inst->op_str, "rsp, qword ptr [r14 + 0x10]"));
}

static uint32_t
add_argument(cs_insn* asm_inst)
{
    if (!asm_inst) return 0;

    // In this example, add_argument is 0x58:
    // 000000000063a083 (04) 4883c458                 ADD RSP, 0x58
    // 000000000063a087 (01) c3                       RET
    // In this example, add_argument is 0xffffffffffffff80:
    // 000000000046f833 (04) 4883ec80                 SUB RSP, $0xffffffffffffff80
    // 000000000046f837 (01) c3                       RET
    if (asm_inst->size == 4) {
        unsigned char* inst_addr = (unsigned char*)asm_inst->address;
        return ((unsigned char*)inst_addr)[3];
    }

    // In this example, add_argument is 0x80:
    // 00000000004a9cc9 (07) 4881c480000000           ADD RSP, 0x80
    // 00000000004a9cd0 (01) c3                       RET
    // In this example, add_argument is 0xffffffffffffff80:
    // 000000000046f833 (07) 4883ec80000000           SUB RSP, $0xffffffffffffff80
    // 000000000046f837 (01) c3                       RET
    if (asm_inst->size == 7) {
        unsigned char* inst_addr = (unsigned char*)asm_inst->address;
        // x86_64 is little-endian.
        return inst_addr[3] +
              (inst_addr[4] << 8 ) +
              (inst_addr[5] << 16) +
              (inst_addr[6] << 24);
    }

    return 0;
}

// Caution!  patch_first_instruction can only be used for things
// that won't trigger a stackscan in go.  If your go function contains
// a call to runtime.morestack_noctxt or calls any other function that
// does, it can trigger a stackscan.  And you'll end up here trying
// to understand what this comment is telling you.
static void
patch_first_instruction(funchook_t *funchook,
               cs_insn* asm_inst, unsigned int asm_count, tap_t* tap)
{
    int i;
    for (i=0; i<asm_count; i++) {
        // Stop when it looks like we've hit another goroutine
        if (i > 0 && (looks_like_first_inst_of_go_func(&asm_inst[i]) ||
                  (!strcmp((const char*)asm_inst[i].mnemonic, "int3") &&
                  asm_inst[i].size == 1 ))) {
            break;
        }

        patchprint("%0*lx (%02d) %-24s %s %s\n",
                   16,
                   asm_inst[i].address,
                   asm_inst[i].size,
                   (char*)asm_inst[i].bytes,
                   (char*)asm_inst[i].mnemonic,
                   (char*)asm_inst[i].op_str);

        uint32_t add_arg=8; // not used, but can't be zero because of go_switch
        if (i == 0) {

            void *pre_patch_addr = (void*)asm_inst[i].address;
            void *patch_addr = (void*)asm_inst[i].address;

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
patch_return_addrs(funchook_t *funchook,
                   cs_insn* asm_inst, unsigned int asm_count, tap_t* tap)
{
    if (!funchook || !asm_inst || !asm_count || !tap) return;

    // special handling for runtime.exit, runtime.dieFromSignal
    // since the go folks wrote them in assembly, they don't follow
    // conventions that other go functions do.
    if ((tap->assembly_fn == go_hook_exit) ||
        (tap->assembly_fn == go_hook_die)) {
        patch_first_instruction(funchook, asm_inst, asm_count, tap);
        return;
    }

    int i;
    for (i=0; i<asm_count; i++) {
        // We've observed that the first instruction in every goroutine
        // is the same (it retrieves a pointer to the goroutine.)
        // We're checking for it here to make sure things are coherent.
        if (i == 0 && !looks_like_first_inst_of_go_func(&asm_inst[i])) break;

        // Stop when it looks like we've hit another goroutine
        if (i > 0 && (looks_like_first_inst_of_go_func(&asm_inst[i]) ||
                  (!strcmp((const char*)asm_inst[i].mnemonic, "int3") &&
                  asm_inst[i].size == 1 ))) {
            break;
        }

        patchprint("%0*lx (%02d) %-24s %s %s\n",
               16,
               asm_inst[i].address,
               asm_inst[i].size,
               (char*)asm_inst[i].bytes,
               (char*)asm_inst[i].mnemonic,
               (char*)asm_inst[i].op_str);

        // If the current instruction is a RET
        // and previous inst is add or sub, then get the stack frame size.
        // Or, if the current inst is xorps then proceed without a stack frame size.
        // If the current inst is not a ret or xorps, don't funchook.
        uint32_t add_arg = 0;
        if (((!strcmp((const char*)asm_inst[i].mnemonic, "ret") &&
              (asm_inst[i].size == 1) &&
              (!strcmp((const char*)asm_inst[i-1].mnemonic, "add") ||
               !strcmp((const char*)asm_inst[i-1].mnemonic, "sub")) &&
              (add_arg = add_argument(&asm_inst[i-1]))) ||

             (!strcmp((const char*)asm_inst[i].mnemonic, "xorps") &&
              (asm_inst[i].size == 4)))) {


                void *pre_patch_addr = (void*)asm_inst[i-1].address;
                void *patch_addr = (void*)asm_inst[i].address;

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
                break;
        }
    }
    patchprint("\n\n");
}

static void
patchClone()
{
    void *clone = dlsym(RTLD_DEFAULT, "__clone");
    if (clone) {
        size_t pageSize = getpagesize();
        void *addr = (void *)((ptrdiff_t) clone & ~(pageSize - 1));

        // set write perms on the page
        if (mprotect(addr, pageSize, PROT_WRITE | PROT_READ | PROT_EXEC)) {
            scopeLogError("ERROR: patchCLone: mprotect failed\n");
            return;
        }

        uint8_t ass[6] = {
            0xb8, 0x00, 0x00, 0x00, 0x00,      // mov $0x0,%eax
            0xc3                               // retq
        };
        memcpy(clone, ass, sizeof(ass));

        scopeLog(CFG_LOG_DEBUG, "patchClone: CLONE PATCHED\n");

        // restore perms to the page
        if (mprotect(addr, pageSize, PROT_READ | PROT_EXEC)) {
            scopeLogError("ERROR: patchCLone: mprotect restore failed\n");
            return;
        }
    }
}

#define UNKNOWN_GO_VER (-1)
#define MIN_SUPPORTED_GO_VER (8)
#define PARAM_ON_REG_GO_VER (19)

static int
go_major_version(const char *go_runtime_version)
{
    if (!go_runtime_version) return UNKNOWN_GO_VER;

    char buf[256] = {0};
    strncpy(buf, go_runtime_version, sizeof(buf)-1);

    char *token = strtok(buf, ".");
    token = strtok(NULL, ".");
    if (!token) {
        return UNKNOWN_GO_VER;
    }

    errno = 0;
    long val = strtol(token, NULL, 10);
    if (errno || val <= 0 || val > INT_MAX) {
        return UNKNOWN_GO_VER;
    }

    return val;
}

static void
adjustGoStructOffsetsForVersion(int go_ver)
{
    if (!go_ver) {
        sysprint("ERROR: can't determine major go version\n");
        return;
    }

    // go 1.8 has a different m_to_tls offset than other supported versions.
    if (go_ver == 8) {
        g_go.m_to_tls = 96; // 0x60
    }

    // go 1.8 is the only version that directly goes from netfd to sysfd.
    if (go_ver == 8) {
        g_go.netfd_to_sysfd = 16;
    }

    // before go 1.12, persistConn_to_conn and persistConn_to_bufrd
    // have different values than 12 and after
    if (go_ver < 12) {
        g_go.persistConn_to_conn = 72;  // 0x48
        g_go.persistConn_to_bufrd = 96; // 0x60
        g_go.persistConn_to_tlsState=88; // 0x58
    }


    // This creates a file specified by test/integration/go/test_go.sh
    // and used by test/integration/go/test_go_struct.sh.
    //
    // Why?  To test structure offsets in go that can vary. (above)
    //
    // The format is:
    //   StructureName|FieldName=DecimalOffsetValue|OptionalTag
    //
    // If an OptionalTag is provided, test_go_struct.sh will not process
    // the line unless it matches a TAG_FILTER which is provided as an
    // argument to the test_go_struct.sh.
    if (!g_fn.open || !g_fn.close) return;

    char* debug_file;
    int fd;
    if ((debug_file = getenv("SCOPE_GO_STRUCT_PATH")) &&
        ((fd = g_fn.open(debug_file, O_CREAT|O_WRONLY|O_CLOEXEC, 0666)) != -1)) {
        dprintf(fd, "runtime.g|m=%d|\n", g_go.g_to_m);
        dprintf(fd, "runtime.m|tls=%d|\n", g_go.m_to_tls);
        dprintf(fd, "net/http.connReader|conn=%d|Server\n", g_go.connReader_to_conn);
        dprintf(fd, "net/http.persistConn|conn=%d|Client\n", g_go.persistConn_to_conn);
        dprintf(fd, "net/http.persistConn|br=%d|Client\n", g_go.persistConn_to_bufrd);
        dprintf(fd, "runtime.iface|data=%d|\n", g_go.iface_data);
        // go 1.8 has a direct netfd_to_sysfd field, others are less direct
        if (g_go.netfd_to_sysfd == UNDEF_OFFSET) {
            dprintf(fd, "net.netFD|pfd=%d|\n", g_go.netfd_to_pd);
            dprintf(fd, "internal/poll.FD|Sysfd=%d|\n", g_go.pd_to_fd);
        } else {
            dprintf(fd, "net.netFD|sysfd=%d|\n", g_go.netfd_to_sysfd);
        }
        dprintf(fd, "bufio.Reader|buf=%d|\n", g_go.bufrd_to_buf);
        dprintf(fd, "net/http.conn|rwc=%d|Server\n", g_go.conn_to_rwc);
        dprintf(fd, "net/http.conn|tlsState=%d|Server\n", g_go.conn_to_tlsState);
        dprintf(fd, "net/http.persistConn|tlsState=%d|Client\n", g_go.persistConn_to_tlsState);
        g_fn.close(fd);
    }

}

int
getBaseAddress(uint64_t *addr) {
    uint64_t base_addr = 0;
    char perms[5];
    char offset[20];
    char buf[1024];
    char pname[1024];
    FILE *fp;

    if (!g_fn.fopen || !g_fn.fgets || !g_fn.fclose) return -1;

    if (osGetProcname(pname, sizeof(pname)) == -1) return -1;

    if ((fp = g_fn.fopen("/proc/self/maps", "r")) == NULL) {
        return -1;
    }

    while (g_fn.fgets(buf, sizeof(buf), fp) != NULL) {
        uint64_t addr_start;
        sscanf(buf, "%lx-%*x %s %*s %s %*d", &addr_start, perms, offset);
        if (strstr(buf, pname) != NULL) {
            base_addr = addr_start;
            break;
        }
    }

    g_fn.fclose(fp);
    if (base_addr) {
        *addr = base_addr;
        return 0;
    }
    return -1;
}

void
initGoHook(elf_buf_t *ebuf)
{
    int rc;
    funchook_t *funchook;
    gostring_t *go_ver; // There is an implicit len field at go_ver + 0x8
    char *go_runtime_version = NULL;

    g_stack = malloc(32 * 1024);
    g_threadlist = lstCreate(NULL);

    // A go app may need to expand stacks for some C functions
    g_need_stack_expand = TRUE;

    funchook = funchook_create();

    if (logLevel(g_log) <= CFG_LOG_DEBUG) {
        // TODO: add some mechanism to get the config'd log file path
        funchook_set_debug_file(DEFAULT_LOG_PATH);
    }

    // default to a dynamic app?
    if (checkEnv("SCOPE_EXEC_TYPE", "static")) {
        g_go_static = 1LL;
        sysprint("This is a static app\n");
    } else {
        g_go_static = 0LL;
        sysprint("This is a dynamic app\n");
    }

    // which go_switch function are we using?
    // Note that __ctype_init needs to be called through
    // a function pointer since musl lib support was added.
    // If that pointer is not initialized, then don't allow
    // go_switch_no_thread() to be used.
    if (g_go_static == 1) {
        if (g_fn.__ctype_init && checkEnv(SWITCH_ENV, SWITCH_USE_NO_THREAD)) {
            g_switch_thread = FALSE;
            patchClone();
        } else if (checkEnv(SWITCH_ENV, SWITCH_USE_THREAD)) {
            g_switch_thread = TRUE;
        } else {
            g_switch_thread = TRUE;
        }
    } else {
        g_switch_thread = TRUE;
    }

    int go_major_ver = UNKNOWN_GO_VER;
    go_ver = getSymbol(ebuf->buf, "runtime.buildVersion");
    if (!go_ver) {
        //runtime.buildVersion symbol not found, probably dealing with a stripped binary
        //try to retrieve the version symbol address from the .go.buildinfo section
        go_ver = getGoVersionAddr(ebuf->buf);
    }
    //check ELF type
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)ebuf->buf;
    // if it's a position independent executable, get the base address from /proc/self/maps
    uint64_t base = 0LL;
    if (ehdr->e_type == ET_DYN && !g_go_static) {
        if (getBaseAddress(&base) != 0) {
            sysprint("ERROR: can't get the base address\n");
            return; // don't install our hooks
        }
        Elf64_Shdr* textSec = getElfSection(ebuf->buf, ".text");
        sysprint("base %lx %lx %lx\n", base, (uint64_t)ebuf->text_addr, textSec->sh_offset);
        base = base - (uint64_t)ebuf->text_addr + textSec->sh_offset;
    }
    
    go_ver = (void *) ((uint64_t)go_ver + base);
    
    if (go_ver && (go_runtime_version = c_str(go_ver))) {

        sysprint("go_runtime_version = %s\n", go_runtime_version);

        go_major_ver = go_major_version(go_runtime_version);
    }
    if (go_major_ver < MIN_SUPPORTED_GO_VER) {
        if (!is_go(ebuf->buf)) {
            // Don't expect to get here, but try to be clear if we do.
            scopeLogWarn("%s is not a go application.  Continuing without AppScope.", ebuf->cmd);
        } else if (go_runtime_version) {
            scopeLogWarn("%s was compiled with go version `%s`.  AppScope can only instrument go1.8 or newer.  Continuing without AppScope.", ebuf->cmd, go_runtime_version);
        } else {
            scopeLogWarn("%s was either compiled with a version of go older than go1.4, or symbols have been stripped.  AppScope can only instrument go1.8 or newer, and requires symbols if compiled with a version of go older than go1.13.  Continuing without AppScope.", ebuf->cmd);
        }
        return; // don't install our hooks
    } else if (go_major_ver >= PARAM_ON_REG_GO_VER) {
        scopeLogWarn("%s was compiled with go version `%s`. Versions newer than Go 1.18 are not yet supported. Continuing without AppScope.", ebuf->cmd, go_runtime_version);
        return; // don't install our hooks
    }
    /*
     * Note: calling runtime.cgocall results in the Go error
     *       "fatal error: cgocall unavailable"
     * Calling runtime.asmcgocall does work. Possibly because we
     * are entering the Go func past the runtime stack check?
     * Need to investigate later.
     */
    //"runtime.asmcgocall.abi0" ??
    if (((go_runtime_cgocall = getGoSymbol(ebuf->buf, "runtime.asmcgocall")) == 0) &&
        ((go_runtime_cgocall = getSymbol(ebuf->buf, "runtime.asmcgocall")) == 0)) {
        sysprint("ERROR: can't get the address for runtime.cgocall\n");
        return; // don't install our hooks
    }
    go_runtime_cgocall = (void *) ((uint64_t)go_runtime_cgocall + base);

    if ((go_runtime_cgocall_regs = getSymbol(ebuf->buf, "runtime.asmcgocall")) == 0) {
        sysprint("ERROR: can't get the address for runtime.cgocall-regs\n");
        return; // don't install our hooks
    }
    go_runtime_cgocall_regs = (void *) ((uint64_t)go_runtime_cgocall_regs + base);

    funcprint("asmcgocall %p asmcgocall_regs %p\n", go_runtime_cgocall, go_runtime_cgocall_regs);

    csh disass_handle = 0;
    cs_arch arch;
    cs_mode mode;
#if defined(__aarch64__)
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_LITTLE_ENDIAN;
#elif defined(__x86_64__)
    arch = CS_ARCH_X86;
    mode = CS_MODE_64;
#else
    return;
#endif
    if (cs_open(arch, mode, &disass_handle) != CS_ERR_OK) return;

    cs_insn *asm_inst = NULL;
    unsigned int asm_count = 0;

    adjustGoStructOffsetsForVersion(go_major_ver);
    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {

        if (asm_inst) {
            cs_free(asm_inst, asm_count);
            asm_inst = NULL;
            asm_count = 0;
        }

        void* orig_func;
        if (((orig_func = getGoSymbol(ebuf->buf, tap->func_name)) == NULL) &&
            ((orig_func = getSymbol(ebuf->buf, tap->func_name)) == NULL)) {
            sysprint("ERROR: can't get the address for %s\n", tap->func_name);
            continue;
        }

        uint64_t offset_into_txt = (uint64_t)orig_func - (uint64_t)ebuf->text_addr;
        uint64_t text_len_left = ebuf->text_len - offset_into_txt;
        uint64_t max_bytes = 4096;  // somewhat arbitrary limit.  Allows for
                                  // >250 instructions @ 15 bytes/inst (x86_64)
        uint64_t size = MIN(text_len_left, max_bytes); // limit size

        orig_func = (void *) ((uint64_t)orig_func + base);
        asm_count = cs_disasm(disass_handle, orig_func, size,
                                 (uint64_t)orig_func, 0, &asm_inst);
        if (asm_count <= 0) {
            sysprint("ERROR: disassembler fails: %s\n\tlen %" PRIu64 " code %p result %lu\n\ttext addr %p text len %zu oinfotext 0x%" PRIx64 "\n",
                     tap->func_name, size,
                     orig_func, sizeof(asm_inst), ebuf->text_addr, ebuf->text_len, offset_into_txt);
            continue;
        }

        patchprint ("********************************\n");
        patchprint ("** %s  %s 0x%p **\n", go_runtime_version, tap->func_name, orig_func);
        patchprint ("********************************\n");

        patch_return_addrs(funchook, asm_inst, asm_count, tap);
    }

    if (asm_inst) {
        cs_free(asm_inst, asm_count);
    }
    cs_close(&disass_handle);

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

    scopeLogError("FATAL ERROR: no return addr");
    exit(-1);
}

static uint32_t
frame_size(assembly_fn fn)
{
    tap_t* tap = NULL;
    for (tap = g_go_tap; tap->assembly_fn; tap++) {
        if (tap->assembly_fn == fn) return tap->frame_size;
    }

    scopeLogError("FATAL ERROR: no frame size");
    exit(-1);
}

static void *
dumb_thread(void *arg)
{
    pthread_barrier_t *pbarrier = (pthread_barrier_t *)arg;
    sigset_t mask;

    sigfillset(&mask);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);

    void *dummy = calloc(1, 32);
    if (dummy) free(dummy);
    pthread_barrier_wait(pbarrier);
    while (1) {
        sleep(0xffffffff);
    }
    return NULL;
}

/*
 * There are 2 go_switch() functions that accomplish the
 * same thing. They take different approaches. We want to 
 * see what effect an approach has on performance and reliability.
 * We'll likley get to one instance as we gain experience. 
 *
 * These go_switch() functions are meant to support
 * static and dynamic apps at the same time. It depends
 * on an env var being set in the scope
 * executable. In initGoHook() we get the
 * env var and set the variable g_go_static.
 * If not 0 the executable is static.
 *
 * In the static case we switch TLS/TCB.
 * In the dynamic case we do not switch TLS/TCB.
 */
/*
* go_switch_thread() creates a thread in glibc
* for each 'm'. It then syncs with that thread
* after the newly created thread inits memory
* allocation.
*/
inline static void *
go_switch_thread(char *stackptr, void *cfunc, void *gfunc)
{
    int fs_valid = 0;
    uint64_t rc;
    unsigned long go_tls = 0;
    unsigned long go_fs = 0;
    unsigned long go_m = 0;
    unsigned long *go_v;
    char *go_g = NULL;

    if (g_go_static) {
        // Get the Go routine's struct g
        __asm__ volatile (
            "mov %%fs:-8, %0"
            : "=r"(go_g)                      // output
            :                                 // inputs
            :                                 // clobbered register
            );

        if (go_g) {
            // get struct m from g and pull out the TLS from 'm'
            go_m = *((unsigned long *)(go_g + g_go.g_to_m));
            go_v = (unsigned long *)go_m;

            // first entry in a valid m is a pointer to g
            if (go_v && (*go_v == (unsigned long)go_g)) { //ok to continue;
                go_tls = (unsigned long)(go_m + g_go.m_to_tls);
                go_v = (unsigned long *)go_tls;

                // is tls set?
                if (go_v && (*go_v != 0)) { // continue
                    go_fs = go_tls + 8; //go compiler uses -8(FS)
                    go_v = (unsigned long *)go_fs;

                    if (go_v && (*go_v != 0)) { // continue
                        fs_valid = 1;
                    }
                }
            }
        }

        if (fs_valid == 0) {
            // We've seen a case where on process exit static cgo
            // apps do not have a go_g while they're exiting.
            // In this case we need to pull the TLS from the kernel
            // Also seen a case on libmusl where TLS is not set.
            // Therefore, get fs from the kernel.
            if (arch_prctl(ARCH_GET_FS, (unsigned long) &go_fs) == -1) {
                scopeLogError("arch_prctl get go");
                goto out;
            }
        }

        void *thread_fs = NULL;
        if ((thread_fs = lstFind(g_threadlist, go_fs)) == NULL) {
            // Switch to the main thread TCB
            if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
                scopeLogError("arch_prctl set scope");
                goto out;
            }
            pthread_t thread;
            pthread_barrier_t barrier;
            if (pthread_barrier_init(&barrier, NULL, 2) != 0) {
                scopeLogError("pthread_barrier_init failed");
                goto out;
            }

            if (!g_fn.pthread_create ||
                (g_fn.pthread_create(&thread, NULL, dumb_thread, &barrier) != 0)) {
                scopeLogError("pthread_create failed");
                goto out;
            }

            //wait until the thread starts
            pthread_barrier_wait(&barrier);

            thread_fs = (void *)thread;

            if (arch_prctl(ARCH_SET_FS, (unsigned long) thread_fs) == -1) {
                scopeLogError("arch_prctl set scope");
                goto out;
            }

            if (pthread_barrier_destroy(&barrier) != 0) {
                scopeLogError("pthread_barrier_destroy failed");
                goto out;
            }

            if (lstInsert(g_threadlist, go_fs, thread_fs) == FALSE) {
                scopeLogError("lstInsert failed");
                goto out;
            }

            sysprint("New thread created for GO TLS = 0x%08lx\n", go_fs);
        } else {
            if (arch_prctl(ARCH_SET_FS, (unsigned long) thread_fs) == -1) {
                scopeLogError("arch_prctl set scope");
                goto out;
            }
        }
    }

    uint32_t frame_offset = frame_size(gfunc);
    //if (!frame_offset) goto out;
    stackptr += frame_offset;

    // call the C handler
    __asm__ volatile (
        "mov %1, %%rdi  \n"
        "callq *%2  \n"
        : "=r"(rc)                    // output
        : "r"(stackptr), "r"(cfunc)   // inputs
        :                             // clobbered register
        );
out:
    if (g_go_static && go_fs) {
        // Switch back to the 'm' TLS
        if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
            scopeLogError("arch_prctl restore go ");
        }
    }
    return return_addr(gfunc);
}

/*
* go_switch_no_thread() creates a glibc TCB for
* each 'm' and initializes TLS and memory allocation
* in the newly created TCB. It uses pthread_create()
* to create a TCB. However, __clone() has previously
* been disabled in initGoHook(). Therefore, no thread
* is actually created.
*
* Note: we found that this version of go_switch()
* results in a malloc error when running with the
* influxdb stress test. Making the thread version
* the default resolves the issue. Have not found
* the root cause. Reproduce by running a static
* influxd, then run the old stress client
* influx_stress_stat and write event files from
* the server and client to the file system.
*/
inline static void *
go_switch_no_thread(char *stackptr, void *cfunc, void *gfunc)
{
    int fs_valid = 0;
    uint64_t rc;
    unsigned long go_tls = 0;
    unsigned long go_fs = 0;
    unsigned long go_m = 0;
    char *go_g = NULL;
    unsigned long *go_v;
    int newThread = FALSE;
    unsigned long rsp;

    if (g_go_static) {
        // Get the Go routine's struct g
        __asm__ volatile (
            "mov %%fs:-8, %0"
            : "=r"(go_g)                      // output
            :                                 // inputs
            :                                 // clobbered register
            );

        if (go_g) {
            // get struct m from g and pull out the TLS from 'm'
            go_m = *((unsigned long *)(go_g + g_go.g_to_m));
            go_v = (unsigned long *)go_m;

            // first entry in a valid m is a pointer to g
            if (go_v && (*go_v == (unsigned long)go_g)) { //ok to continue;
                go_tls = (unsigned long)(go_m + g_go.m_to_tls);
                go_v = (unsigned long *)go_tls;

                // is tls set?
                if (go_v && (*go_v != 0)) { // continue
                    go_fs = go_tls + 8; //go compiler uses -8(FS)
                    go_v = (unsigned long *)go_fs;

                    if (go_v && (*go_v != 0)) { // continue
                        fs_valid = 1;
                    }
                }
            }
        }

        if (fs_valid == 0) {
            // We've seen a case where on process exit static cgo
            // apps do not have a go_g while they're exiting.
            // In this case we need to pull the TLS from the kernel
            // Also seen a case on libmusl where TLS is not set.
            // Therefore, get fs from the kernel.
            if (arch_prctl(ARCH_GET_FS, (unsigned long) &go_fs) == -1) {
                scopeLogError("arch_prctl get go");
                goto out;
            }
        }

        void *thread_fs = NULL;
        if ((thread_fs = lstFind(g_threadlist, go_fs)) == NULL) {
            // Switch to the main thread TCB
            if (arch_prctl(ARCH_SET_FS, scope_fs) == -1) {
                scopeLogError("arch_prctl set scope");
                goto out;
            }
            pthread_t thread;
            if (!g_fn.pthread_create ||
                (g_fn.pthread_create(&thread, NULL, dumb_thread, NULL) != 0)) {
                scopeLogError("pthread_create failed");
                goto out;
            }
            thread_fs = (void *)thread;
            newThread = TRUE;            
        } 

        if (arch_prctl(ARCH_SET_FS, (unsigned long) thread_fs) == -1) {
            scopeLogError("arch_prctl set scope");
            goto out;
        }

        if (newThread) {
            while (!atomicCasU64(&g_glibc_guard, 0ULL, 1ULL)) {};

            //Initialize pointers to locale data.
            if (g_fn.__ctype_init) g_fn.__ctype_init();

            //switch to a bigger stack
            __asm__ volatile (
                "mov %%rsp, %0 \n"
                "mov %1, %%rsp \n"
                : "=r"(rsp) 
                : "m"(g_stack)
                : 
            );

            //initialize tcache
            void *buf = calloc(1, 0xff);
            free(buf);

            //restore stack
            __asm__ volatile (
                "mov %0, %%rsp \n"
                : 
                : "r"(rsp)
                : 
            );
             
            atomicCasU64(&g_glibc_guard, 1ULL, 0ULL);

            if (lstInsert(g_threadlist, go_fs, thread_fs) == FALSE) {
                scopeLogError("lstInsert failed");
                goto out;
            }
        }
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
out:
    if (g_go_static && go_fs) {
        // Switch back to the 'm' TLS
        if (arch_prctl(ARCH_SET_FS, go_fs) == -1) {
            scopeLogError("arch_prctl restore go ");
        }
    }
    return return_addr(gfunc);
}

/*
* We want to be able to determine which go_switch
* function to use at run time.
*/
inline static void *
go_switch(char *stackptr, void *cfunc, void *gfunc)
{
    if (g_switch_thread == FALSE) {
        return go_switch_no_thread(stackptr, cfunc, gfunc);
    } else {
        return go_switch_thread(stackptr, cfunc, gfunc);
    }
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
c_getdents(char *stackaddr)
{
    uint64_t dirfd  = *(uint64_t *)(stackaddr + 0x8);
    uint64_t rc     = *(uint64_t *)(stackaddr + 0x28);
    uint64_t initialTime = getTime();

    funcprint("Scope: getdents dirfd %ld rc %ld\n", dirfd, rc);
    doRead(dirfd, initialTime, (rc != -1), NULL, rc, "go_getdents", BUF, 0);
}

EXPORTON void *
go_getdents(char *stackptr)
{
    return go_switch(stackptr, c_getdents, go_hook_getdents);
}

static void
c_unlinkat(char *stackaddr)
{
    uint64_t dirfd  = *(uint64_t *)(stackaddr + 0x8);
    char *pathname = c_str((gostring_t*)(stackaddr + 0x10));
    uint64_t flags  = *(uint64_t *)(stackaddr + 0x18);

    if (!pathname) {
        scopeLogError("ERROR:go_open: null pathname");
        puts("Scope:ERROR:open:no pathname");
        return;
    }

    funcprint("Scope: unlinkat dirfd %ld pathname %s flags %ld\n", dirfd, pathname, flags);
    doDelete(pathname, "go_unlinkat");

    if (pathname) free(pathname);
}

EXPORTON void *
go_unlinkat(char *stackptr)
{
    return go_switch(stackptr, c_unlinkat, go_hook_unlinkat);
}

static void
c_open(char *stackaddr)
{
    uint64_t fd  = *((uint64_t *)(stackaddr + 0x30));
    // The gostring_t* here has an implicit len field at stackaddr + 0x18
    char *path = c_str((gostring_t*)(stackaddr + 0x10));

    if (!path) {
        scopeLogError("ERROR:go_open: null pathname");
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

    funcprint("Scope: read of %ld rc %ld\n", fd, rc);
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
    uint64_t domain = *(uint64_t*)(stackaddr + 0x08);  // aka family
    uint64_t type   = *(uint64_t*)(stackaddr + 0x10);
    uint64_t sd     = *(uint64_t*)(stackaddr + 0x20);

    if (sd == -1) return;

    funcprint("Scope: socket domain: %ld type: 0x%lx sd: %ld\n", domain, type, sd);
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
    uint64_t fd = *(uint64_t*)(stackaddr + 0x8); 
    struct sockaddr *addr  = *(struct sockaddr **)(stackaddr + 0x10);
    socklen_t *addrlen = *(socklen_t**)(stackaddr + 0x18);
    uint64_t sd_out = *(uint64_t*)(stackaddr + 0x28);

    if (sd_out != -1) {
        funcprint("Scope: accept4 of %ld\n", sd_out);
        doAccept(fd, sd_out, addr, addrlen, "go_accept4");
    }
}

EXPORTON void *
go_accept4(char *stackptr)
{
    return go_switch(stackptr, c_accept4, go_hook_accept4);
}

/*
  net/http.(*connReader).Read
  /usr/local/go/src/net/http/server.go:758

  cr = stackaddr + 0x08
  cr.conn = *cr
  cr.conn.rwc_if = cr.conn + 0x10
  cr.conn.rwc = cr.conn.rwc_if + 0x08
  netFD = cr.conn.rwc + 0x08
  pfd = *netFD  (/usr/local/go/src/net/fd_unix.go:20)
  fd = netFD + 0x10

  reference: net/http/server.go
  type connReader struct {
        conn *conn

  type conn struct {
          server *Server
          cancelCtx context.CancelFunc
          rwc net.Conn
          remoteAddr string
          tlsState *tls.ConnectionState
 */
static void
c_http_server_read(char *stackaddr)
{
    int fd = -1;
    uint64_t connReader = *(uint64_t*)(stackaddr + 0x8);
    if (!connReader) return;   // protect from dereferencing null
    uint64_t buf        = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);
    uint64_t cr_conn_rwc_if, cr_conn_rwc, netFD, pfd;

    uint64_t conn =  *(uint64_t*)(connReader + g_go.connReader_to_conn);
    if (!conn) return;         // protect from dereferencing null

    cr_conn_rwc_if = conn + g_go.conn_to_rwc;
    uint64_t tls =  *(uint64_t*)(conn + g_go.conn_to_tlsState);

    /*
     * The rwc net.Conn value can be wrapped as either a *net.TCPConn or
     * *tls.Conn. We are using tlsState *tls.ConnectionState as the indicator
     * of type. If the tlsState field is not 0, the rwc field is of type
     * *tls.Conn. Example; net/http/server.go type conn struct.
     *
     * For reference, we were looking at the I/F type from go.itab.*crypto/tls.Conn,net.Conn
     * and checking the type to determine TLS. This doesn't work on stripped
     * executables and should no longer be needed.
     */
    if (cr_conn_rwc_if && tls) {
        cr_conn_rwc = *(uint64_t *)(cr_conn_rwc_if + g_go.iface_data);
        netFD = *(uint64_t *)(cr_conn_rwc + g_go.iface_data);
        if (netFD) {
            pfd = *(uint64_t *)(netFD + g_go.netfd_to_pd);
            if (pfd) {
                //funcprint("Scope: %s:%d cr %p cr.conn %p cr.conn.rwc_if %p cr.conn.rwc %p netFD %p pfd %p fd %p\n",
                //          __FUNCTION__, __LINE__, connReader, conn, cr_conn_rwc_if, cr_conn_rwc,
                //          netFD, pfd, pfd + g_go.pd_to_fd);
                fd = *(int *)(pfd + g_go.pd_to_fd);
            }

            funcprint("Scope: go_http_server_read of %d\n", fd);
            doProtocol((uint64_t)0, fd, (void *)buf, rc, TLSRX, BUF);
        }
    }
}

EXPORTON void *
go_tls_read(char *stackptr)
{
    return go_switch(stackptr, c_http_server_read, go_hook_tls_read);
}

/*
  net/http.checkConnErrorWriter.Write
  /usr/local/go/src/net/http/server.go:3433

  conn = stackaddr + 0x08
  conn.rwc_if = conn + 0x10
  conn.rwc = conn.rwc_if + 0x08
  netFD = conn.rwc + 0x08
  pfd = *netFD
  fd = pfd + 0x10
 */
static void
c_http_server_write(char *stackaddr)
{
    int fd = -1;
    uint64_t conn = *(uint64_t*)(stackaddr + 0x8);
    if (!conn) return;         // protect from dereferencing null
    uint64_t buf  = *(uint64_t*)(stackaddr + 0x10);
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc  = *(uint64_t*)(stackaddr + 0x28);
    uint64_t w_conn_rwc_if, w_conn_rwc, netFD, pfd;

    w_conn_rwc_if = (conn + g_go.conn_to_rwc);
    uint64_t tls =  *(uint64_t*)(conn + g_go.conn_to_tlsState);

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (w_conn_rwc_if && tls) {
        w_conn_rwc = *(uint64_t *)(w_conn_rwc_if + g_go.iface_data);
        netFD = *(uint64_t *)(w_conn_rwc + g_go.iface_data);
        if (netFD) {
            pfd = *(uint64_t *)(netFD + g_go.netfd_to_pd);
            if (pfd) {
                fd = *(int *)(pfd + g_go.pd_to_fd);
            }

            funcprint("Scope: c_http_server_write of %d\n", fd);
            doProtocol((uint64_t)0, fd, (void *)buf, rc, TLSTX, BUF);
        }
    }
}

EXPORTON void *
go_tls_write(char *stackptr)
{
    return go_switch(stackptr, c_http_server_write, go_hook_tls_write);
}

/*
  net/http.persistConnWriter.Write
  /usr/local/go/src/net/http/transport.go:1662

  p = stackaddr + 0x10  (request buffer)
  *p = request string

  w = stackaddr + 0x08        (net/http.persistConnWriter *)
  w.pc = stackaddr + 0x08     (net/http.persistConn *)
  w.pc.conn_if = w.pc + 0x50  (interface)
  w.pc.conn = w.pc.conn_if + 0x08 (net.conn->TCPConn)
  netFD = w.pc.conn + 0x08    (netFD)
  pfd = netFD + 0x0           (poll.FD)
  fd = pfd + 0x10             (pfd.sysfd)
 */
static void
c_http_client_write(char *stackaddr)
{
    int fd = -1;
    uint64_t buf = *(uint64_t *)(stackaddr + 0x10);
    uint64_t w_pc  = *(uint64_t *)(stackaddr + 0x08);
    uint64_t rc =  *(uint64_t *)(stackaddr + 0x20); // 0x20 -> 0x78??
    uint64_t pc_conn_if, w_pc_conn, netFD, pfd;

    if (rc < 1) return;

    //uint64_t w_pc = *(uint64_t *)(w + 0x38);
    pc_conn_if = (w_pc + g_go.persistConn_to_conn); // 0x50
    uint64_t tls =  *(uint64_t*)(w_pc + g_go.persistConn_to_tlsState); //0x60

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (pc_conn_if && tls) {
        w_pc_conn = *(uint64_t *)(pc_conn_if + g_go.iface_data); //0x08
        netFD = *(uint64_t *)(w_pc_conn + g_go.iface_data);
        if (!netFD) return;
        if (g_go.netfd_to_sysfd == UNDEF_OFFSET) { 
            pfd = *(uint64_t *)(netFD + g_go.netfd_to_pd); //0x00
            if (!pfd) return;
            fd = *(int *)(pfd + g_go.pd_to_fd); //0x10
        } else {
            fd = *(int *)(netFD + g_go.netfd_to_sysfd); //0x10
        }

        doProtocol((uint64_t)0, fd, (void *)buf, rc, TLSRX, BUF);
        funcprint("Scope: c_http_client_write of %d\n", fd);
    }
}

EXPORTON void *
go_pc_write(char *stackptr)
{
    return go_switch(stackptr, c_http_client_write, go_hook_pc_write);
}

/*
  net/http.(*persistConn).readResponse
  /usr/local/go/src/net/http/transport.go:2161

  pc = stackaddr + 0x08    (net/http.persistConn *)

  pc.conn_if = pc + 0x50   (interface)
  conn.data = pc.conn_if + 0x08 (net.conn->TCPConn)
  netFD = conn.data + 0x08 (netFD)
  pfd = netFD + 0x0        (poll.FD)
  fd = pfd + 0x10          (pfd.sysfd)

  pc.br = pc.conn + 0x68   (bufio.Reader)
  len = pc.br + 0x08       (bufio.Reader)
  resp = buf + 0x0         (bufio.Reader.buf)
  resp = http response     (char *)
 */
static void
c_http_client_read(char *stackaddr)
{
    int fd = -1;
    uint64_t pc  = *(uint64_t *)(stackaddr + 0x08);
    uint64_t pc_conn_if, pc_conn, netFD, pfd, pc_br, buf = 0, len = 0;

    pc_conn_if = (pc + g_go.persistConn_to_conn);
    uint64_t tls =  *(uint64_t*)(pc + g_go.persistConn_to_tlsState);

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (pc_conn_if && tls) {
        pc_conn = *(uint64_t *)(pc_conn_if + g_go.iface_data);
        netFD = *(uint64_t *)(pc_conn + g_go.iface_data);
        if (!netFD) return;
        if (g_go.netfd_to_sysfd == UNDEF_OFFSET) {
            pfd = *(uint64_t *)(netFD + g_go.netfd_to_pd);
            if (!pfd) return;
            fd = *(int *)(pfd + g_go.pd_to_fd);
        } else {
            fd = *(int *)(netFD + g_go.netfd_to_sysfd);
        }

        if ((pc_br = *(uint64_t *)(pc + g_go.persistConn_to_bufrd)) != 0) {
            buf = *(uint64_t *)(pc_br + g_go.bufrd_to_buf);
            // len is part of the []byte struct; the func doesn't return a len
            len = *(uint64_t *)(pc_br + 0x08);
        }

        if (buf && (len > 0)) {
            doProtocol((uint64_t)0, fd, (void *)buf, len, TLSRX, BUF);
            funcprint("Scope: c_http_client_read of %d\n", fd);
        }
    }
}

EXPORTON void *
go_readResponse(char *stackptr)
{
    return go_switch(stackptr, c_http_client_read, go_hook_readResponse);
}

extern void handleExit(void);
static void
c_exit(char *stackaddr)
{
    // don't use stackaddr; patch_first_instruction() does not provide
    // frame_size, so stackaddr isn't usable
    funcprint("c_exit");

    int i;
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 10000}; // 10 us

    // ensure the circular buffer is empty
    for (i = 0; i < 100; i++) {
        if (cmdCbufEmpty(g_ctl)) break;
        sigSafeNanosleep(&ts);
    }

    handleExit();
    // flush the data
    sigSafeNanosleep(&ts);
}

EXPORTON void *
go_exit(char *stackptr)
{
    return go_switch(stackptr, c_exit, go_hook_exit);
}

EXPORTON void *
go_die(char *stackptr)
{
    return go_switch(stackptr, c_exit, go_hook_die);
}
