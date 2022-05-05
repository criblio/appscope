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
#include "scopestdlib.h"

#define SCOPE_STACK_SIZE (size_t)(32 * 1024)
//#define ENABLE_SIGNAL_MASKING_IN_SYSEXEC 1
#define ENABLE_CAS_IN_SYSEXEC 1

#define SWITCH_ENV "SCOPE_SWITCH"
#define SWITCH_USE_NO_THREAD "no_thread"
#define SWITCH_USE_THREAD "thread"
#define EXIT_STACK_SIZE (32 * 1024)

#define UNKNOWN_GO_VER (-1)
#define MIN_SUPPORTED_GO_VER (8)
#define PARAM_ON_REG_GO_VER (19)

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
//#define funcprint sysprint
#define funcprint devnull
//#define patchprint sysprint
#define patchprint devnull

#define UNDEF_OFFSET (-1)

int g_go_major_ver = UNKNOWN_GO_VER;

go_schema_t go_16_schema = {
    .arg_offsets = {
        .c_write_fd=0x8,
        .c_write_buf=0x10,
        .c_write_rc=0x28,
        .c_getdents_dirfd=0x8,
        .c_getdents_rc=0x28,
        .c_unlinkat_dirfd=0x8,
        .c_unlinkat_pathname=0x10,
        .c_unlinkat_flags=0x18,
        .c_open_fd=0x30,
        .c_open_path=0x10,
        .c_close_fd=0x8,
        .c_close_rc=0x10,
        .c_read_fd=0x8,
        .c_read_buf=0x10,
        .c_read_rc=0x28,
        .c_socket_domain=0x8,
        .c_socket_type=0x10,
        .c_socket_sd=0x20,
        .c_accept4_fd=0x8,
        .c_accept4_addr=0x10,
        .c_accept4_addrlen=0x18,
        .c_accept4_sd_out=0x28,

        // For all system calls, we read the stack from the Caller. For
        // non-system calls, we (for some functions) read the stack from
        // the Callee. We use an offset of either 0 (stay in Caller) or
        // >0 (go into Callee) to put ourselves into the desired context
        // where all values are available.
        // Consideration: In future iterations, it might be a more logical
        // approach to always grab arguments from the Caller, and always
        // grab return values from the Callee.

        .c_http_server_read_callee=0x0,
        .c_http_server_read_connReader=0x8,
        .c_http_server_read_buf=0x10,
        .c_http_server_read_rc=0x28,
        .c_http_server_write_callee=0x0,
        .c_http_server_write_conn=0x8,
        .c_http_server_write_buf=0x10,
        .c_http_server_write_rc=0x28,
        .c_http_client_write_callee=0x0,
        .c_http_client_write_w_pc=0x8,
        .c_http_client_write_buf=0x10,
        .c_http_client_write_rc=0x28,
        .c_http_client_read_callee=0x0,
        .c_http_client_read_pc=0x8,
    },
    .struct_offsets = {
        .g_to_m=0x30,
        .m_to_tls=0x88,
        .connReader_to_conn=0x0,
        .persistConn_to_conn=0x50,
        .persistConn_to_bufrd=0x68,
        .iface_data=0x8,
        .netfd_to_pd=0x0,
        .pd_to_fd=0x10,
        .netfd_to_sysfd=UNDEF_OFFSET, // defined for go1.8
        .bufrd_to_buf=0x0,
        .conn_to_rwc=0x10,
        .conn_to_tlsState=0x30,
        .persistConn_to_tlsState=0x60,
    },
    .tap = {
        {"syscall.write",                        go_hook_write,        NULL, 0},
        {"syscall.openat",                       go_hook_open,         NULL, 0},
        {"syscall.unlinkat",                     go_hook_unlinkat,     NULL, 0},
        {"syscall.Getdents",                     go_hook_getdents,     NULL, 0},
        {"syscall.socket",                       go_hook_socket,       NULL, 0},
        {"syscall.accept4",                      go_hook_accept4,      NULL, 0},
        {"syscall.read",                         go_hook_read,         NULL, 0},
        {"syscall.Close",                        go_hook_close,        NULL, 0},
        {"net/http.(*connReader).Read",          go_hook_tls_read,     NULL, 0},
        {"net/http.checkConnErrorWriter.Write",  go_hook_tls_write,    NULL, 0},
        {"net/http.(*persistConn).readResponse", go_hook_readResponse, NULL, 0}, 
        {"net/http.persistConnWriter.Write",     go_hook_pc_write,     NULL, 0},
        {"runtime.exit",                         go_hook_exit,         NULL, 0},
        {"runtime.dieFromSignal",                go_hook_die,          NULL, 0},
        {"TAP_TABLE_END", NULL, NULL, 0}
    },
};

go_schema_t go_17_schema = {
    .arg_offsets = {
        .c_write_fd=0x8,
        .c_write_buf=0x10,
        .c_write_rc=0x28,
        .c_getdents_dirfd=0x8,
        .c_getdents_rc=0x20,
        .c_unlinkat_dirfd=0x8,
        .c_unlinkat_pathname=0x10,
        .c_unlinkat_flags=0x18,
        .c_open_fd=0x38,
        .c_open_path=0x10,
        .c_close_fd=0x8,
        .c_close_rc=0x10,
        .c_read_fd=0x8,
        .c_read_buf=0x10,
        .c_read_rc=0x28,
        .c_socket_domain=0x8,
        .c_socket_type=0x10,
        .c_socket_sd=0x20,
        .c_accept4_fd=0x8,
        .c_accept4_addr=0x10,
        .c_accept4_addrlen=0x18,
        .c_accept4_sd_out=0x38,
        .c_http_server_read_callee=0x48,
        .c_http_server_read_connReader=0x50,
        .c_http_server_read_buf=0x8,
        .c_http_server_read_rc=0x28,
        .c_http_server_write_callee=0x48,
        .c_http_server_write_conn=0x30,
        .c_http_server_write_buf=0x8,
        .c_http_server_write_rc=0x10,
        .c_http_client_write_callee=0x30,
        .c_http_client_write_w_pc=0x20,
        .c_http_client_write_buf=0x8,
        .c_http_client_write_rc=0x10,
        .c_http_client_read_callee=0x0,
        .c_http_client_read_pc=0x8,
    },
    .struct_offsets = {
        .g_to_m=0x30,
        .m_to_tls=0x88,
        .connReader_to_conn=0x0,
        .persistConn_to_conn=0x50,
        .persistConn_to_bufrd=0x68,
        .iface_data=0x8,
        .netfd_to_pd=0x0,
        .pd_to_fd=0x10,
        .netfd_to_sysfd=UNDEF_OFFSET, // defined for go1.8
        .bufrd_to_buf=0x0,
        .conn_to_rwc=0x10,
        .conn_to_tlsState=0x30,
        .persistConn_to_tlsState=0x60,
    },
    // use the _reg_ assembly functions here, to support changes to Go 1.17
    // where we preserve the return values stored in registers
    // and we preserve the g in r14 for future stack checks
    // Note: we do not need to use the reg functions for go_hook_exit and go_hook_die
    .tap = {
        {"syscall.write",                        go_hook_reg_write,            NULL, 0}, // write
        {"syscall.openat",                       go_hook_reg_open,             NULL, 0}, // file open
        {"syscall.unlinkat",                     go_hook_reg_unlinkat,         NULL, 0}, // delete file
        {"syscall.Getdents",                     go_hook_reg_getdents,         NULL, 0}, // read dir
        {"syscall.socket",                       go_hook_reg_socket,           NULL, 0}, // net open
        {"syscall.accept4",                      go_hook_reg_accept4,          NULL, 0}, // plain server accept
        {"syscall.read",                         go_hook_reg_read,             NULL, 0}, // read
        {"syscall.Close",                        go_hook_reg_close,            NULL, 0}, // close
        {"net/http.(*connReader).Read",          go_hook_reg_tls_read,         NULL, 0}, // tls server read
        {"net/http.checkConnErrorWriter.Write",  go_hook_reg_tls_write,        NULL, 0}, // tls server write
        {"net/http.(*persistConn).readResponse", go_hook_reg_readResponse,     NULL, 0}, // tls client read
        {"net/http.persistConnWriter.Write",     go_hook_reg_pc_write,         NULL, 0}, // tls client write
        {"runtime.exit.abi0",                    go_hook_exit,                 NULL, 0},
        {"runtime.dieFromSignal",                go_hook_die,                  NULL, 0},
        {"TAP_TABLE_END", NULL, NULL, 0}
    },
};

go_schema_t *g_go_schema = &go_16_schema; // overridden if later version
uint64_t g_glibc_guard = 0LL;

void (*go_runtime_cgocall)(void);

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
    if ((path = scope_calloc(1, go_str->len+1)) == NULL) return NULL;
    scope_memmove(path, go_str->str, go_str->len);
    path[go_str->len] = '\0';

    return path;
}

static char *
go_str(void *go_str)
{
    // Go 17 and higher use null terminated strings instead of a string and a length
    if (g_go_major_ver > 16) {
       // We need to deference the address first before casting to a char *
       if (!go_str) return NULL;
       return (char *)*(uint64_t *)go_str;
    }
    return c_str((gostring_t *)go_str);
}

static void
free_go_str(char *str) {
    // Go 17 and higher use null terminated strings instead of a string and a length
    if (g_go_major_ver > 16) {
        return;
    }
    if(str) scope_free(str);
}

// Detect the beginning of a Go Function
// by identifying instructions in the preamble.
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
        
    return (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rcx, qword ptr fs:[0xfffffffffffffff8]")) ||
            // -buildmode=pie compiles to this:
            (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rcx, -8")) || 
            // In Go 17 we extended the definition of function preamble with:
            (!scope_strcmp((const char*)asm_inst->mnemonic, "cmp") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rsp, qword ptr [r14 + 0x10]")) ||
            (!scope_strcmp((const char*)asm_inst->mnemonic, "lea") &&
            !scope_strcmp((const char*)asm_inst->op_str, "r12, [rsp - 0x10]"));
}

// Calculate the value to be added/subtracted at an add/sub instruction
// Returns an absolute value
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
                  (!scope_strcmp((const char*)asm_inst[i].mnemonic, "int3") &&
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

        uint32_t add_arg=8; // not used, but can't be zero because of do_cfunc
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

// Patch all return addresses
// If we intend to patch a return, we should actually patch the instruction prior.
// In some cases, we actually patch the "xorps" instruction instead of the return instruction.
// In that event, we intend to patch the actual "xorps" instruction, not the instruction prior.
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
                  (!scope_strcmp((const char*)asm_inst[i].mnemonic, "int3") &&
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


        // It is necessary for us to interpose system calls (only) for 1.17 and 1.18
        // at the "xorps" instruction. This "xorps" instruction is a good place for
        // us to patch our code in, since it appears right after the system call. We
        // are able to patch non-system calls at the return instruction per prior versions. 

        // If the current instruction is a RET
        // and previous inst is add or sub, then get the stack frame size.
        // Or, if the current inst is xorps then proceed without a stack frame size.
        // If the current inst is not a ret or xorps, don't funchook.
        uint32_t add_arg = 0;
        if (((!scope_strcmp((const char*)asm_inst[i].mnemonic, "ret") &&
              (asm_inst[i].size == 1) &&
              (!scope_strcmp((const char*)asm_inst[i-1].mnemonic, "add") ||
              !scope_strcmp((const char*)asm_inst[i-1].mnemonic, "sub")) &&
              (add_arg = add_argument(&asm_inst[i-1]))) ||
              ((g_go_major_ver > 16) && !scope_strcmp((const char*)asm_inst[i].mnemonic, "xorps") &&
              (asm_inst[i].size == 4)))) {

            // Patch the address before the ret instruction to maintain the callee stack context
            void *pre_patch_addr = (void*)asm_inst[i-1].address;
            void *patch_addr = (void*)asm_inst[i-1].address;
            // we aren't dealing with a ret, we must patch the xorps instruction exactly
            if (!scope_strcmp((const char*)asm_inst[i].mnemonic, "xorps")) {
                pre_patch_addr = (void*)asm_inst[i].address;
                patch_addr = (void*)asm_inst[i].address;
            }

            // All add_arg values within a function should be the same
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

            // We need to force a break in the "xorps" case since the code won't be returning here
            if ((g_go_major_ver > 16) && !scope_strcmp((const char*)asm_inst[i].mnemonic, "xorps")) break;
        }
    }
    patchprint("\n\n");
}

static void
patchClone()
{
    void *clone = dlsym(RTLD_DEFAULT, "__clone");
    if (clone) {
        size_t pageSize = scope_getpagesize();
        void *addr = (void *)((ptrdiff_t) clone & ~(pageSize - 1));

        // set write perms on the page
        if (scope_mprotect(addr, pageSize, PROT_WRITE | PROT_READ | PROT_EXEC)) {
            scopeLogError("ERROR: patchCLone: mprotect failed\n");
            return;
        }

        uint8_t ass[6] = {
            0xb8, 0x00, 0x00, 0x00, 0x00,      // mov $0x0,%eax
            0xc3                               // retq
        };
        scope_memcpy(clone, ass, sizeof(ass));

        scopeLog(CFG_LOG_DEBUG, "patchClone: CLONE PATCHED\n");

        // restore perms to the page
        if (scope_mprotect(addr, pageSize, PROT_READ | PROT_EXEC)) {
            scopeLogError("ERROR: patchCLone: mprotect restore failed\n");
            return;
        }
    }
}

// Detect the Go Version of an executable
static int
go_major_version(const char *go_runtime_version)
{
    if (!go_runtime_version) return UNKNOWN_GO_VER;

    char buf[256] = {0};
    scope_strncpy(buf, go_runtime_version, sizeof(buf)-1);

    char *token = scope_strtok(buf, ".");
    token = scope_strtok(NULL, ".");
    if (!token) {
        return UNKNOWN_GO_VER;
    }

    scope_errno = 0;
    long val = scope_strtol(token, NULL, 10);
    if (scope_errno || val <= 0 || val > INT_MAX) {
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
        g_go_schema->struct_offsets.m_to_tls = 96; // 0x60
    }

    // go 1.8 is the only version that directly goes from netfd to sysfd.
    if (go_ver == 8) {
        g_go_schema->struct_offsets.netfd_to_sysfd = 16;
    }

    // before go 1.12, persistConn_to_conn and persistConn_to_bufrd
    // have different values than 12 and after
    if (go_ver < 12) {
        g_go_schema->struct_offsets.persistConn_to_conn = 72;  // 0x48
        g_go_schema->struct_offsets.persistConn_to_bufrd = 96; // 0x60
        g_go_schema->struct_offsets.persistConn_to_tlsState=88; // 0x58
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

    char* debug_file;
    int fd;
    if ((debug_file = getenv("SCOPE_GO_STRUCT_PATH")) &&
        ((fd = scope_open(debug_file, O_CREAT|O_WRONLY|O_CLOEXEC, 0666)) != -1)) {
        scope_dprintf(fd, "runtime.g|m=%d|\n", g_go_schema->struct_offsets.g_to_m);
        scope_dprintf(fd, "runtime.m|tls=%d|\n", g_go_schema->struct_offsets.m_to_tls);
        scope_dprintf(fd, "net/http.connReader|conn=%d|Server\n", g_go_schema->struct_offsets.connReader_to_conn);
        scope_dprintf(fd, "net/http.persistConn|conn=%d|Client\n", g_go_schema->struct_offsets.persistConn_to_conn);
        scope_dprintf(fd, "net/http.persistConn|br=%d|Client\n", g_go_schema->struct_offsets.persistConn_to_bufrd);
        scope_dprintf(fd, "runtime.iface|data=%d|\n", g_go_schema->struct_offsets.iface_data);
        // go 1.8 has a direct netfd_to_sysfd field, others are less direct
        if (g_go_schema->struct_offsets.netfd_to_sysfd == UNDEF_OFFSET) {
            scope_dprintf(fd, "net.netFD|pfd=%d|\n", g_go_schema->struct_offsets.netfd_to_pd);
            scope_dprintf(fd, "internal/poll.FD|Sysfd=%d|\n", g_go_schema->struct_offsets.pd_to_fd);
        } else {
            scope_dprintf(fd, "net.netFD|sysfd=%d|\n", g_go_schema->struct_offsets.netfd_to_sysfd);
        }
        scope_dprintf(fd, "bufio.Reader|buf=%d|\n", g_go_schema->struct_offsets.bufrd_to_buf);
        scope_dprintf(fd, "net/http.conn|rwc=%d|Server\n", g_go_schema->struct_offsets.conn_to_rwc);
        scope_dprintf(fd, "net/http.conn|tlsState=%d|Server\n", g_go_schema->struct_offsets.conn_to_tlsState);
        scope_dprintf(fd, "net/http.persistConn|tlsState=%d|Client\n", g_go_schema->struct_offsets.persistConn_to_tlsState);
        scope_close(fd);
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

    if (osGetProcname(pname, sizeof(pname)) == -1) return -1;

    if ((fp = scope_fopen("/proc/self/maps", "r")) == NULL) {
        return -1;
    }

    while (scope_fgets(buf, sizeof(buf), fp) != NULL) {
        uint64_t addr_start;
        scope_sscanf(buf, "%lx-%*x %s %*s %s %*d", &addr_start, perms, offset);
        if (scope_strstr(buf, pname) != NULL) {
            base_addr = addr_start;
            break;
        }
    }

    scope_fclose(fp);
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

    // A go app may need to expand stacks for some C functions
    g_need_stack_expand = TRUE;

    funchook = funchook_create();

    if (logLevel(g_log) <= CFG_LOG_DEBUG) {
        // TODO: add some mechanism to get the config'd log file path
        funchook_set_debug_file(DEFAULT_LOG_PATH);
    }

    // default to a dynamic app?
    if (checkEnv("SCOPE_EXEC_TYPE", "static")) {
        setGoAppStateStatic(TRUE);
        patchClone();
        sysprint("This is a static app\n");
    } else {
        setGoAppStateStatic(FALSE);
        sysprint("This is a dynamic app\n");
    }

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
    if (ehdr->e_type == ET_DYN && (getgoAppStateStatic() == FALSE)) {
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

        g_go_major_ver = go_major_version(go_runtime_version);
    }
    if (g_go_major_ver < MIN_SUPPORTED_GO_VER) {
        if (!is_go(ebuf->buf)) {
            // Don't expect to get here, but try to be clear if we do.
            scopeLogWarn("%s is not a go application.  Continuing without AppScope.", ebuf->cmd);
        } else if (go_runtime_version) {
            scopeLogWarn("%s was compiled with go version `%s`.  AppScope can only instrument go1.8 or newer.  Continuing without AppScope.", ebuf->cmd, go_runtime_version);
        } else {
            scopeLogWarn("%s was either compiled with a version of go older than go1.4, or symbols have been stripped.  AppScope can only instrument go1.8 or newer, and requires symbols if compiled with a version of go older than go1.13.  Continuing without AppScope.", ebuf->cmd);
        }
        return; // don't install our hooks
    } else if (g_go_major_ver >= PARAM_ON_REG_GO_VER) {
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

    /* Go 1.17 introduced a secondary calling convention for the ABI
     * that allows developers to choose from the native ABI (with
     * latest changes) vs ABI0 (previous ABI). The new native ABI
     * was a no-go for us as in some cases (-buildmode=pie) register
     * %rax was overwritten via a mov to %eax, effectively truncating
     * the return value to 32 bits where on occasion a 64 bit return
     * value might be desired.
     */
    if (g_go_major_ver > 16) {
        // Use the abi0 I/F for syscall based functions in Go >= 1.17
        if (((go_runtime_cgocall = getSymbol(ebuf->buf, "runtime.asmcgocall.abi0")) == 0) &&
            ((go_runtime_cgocall = getGoSymbol(ebuf->buf, "runtime.asmcgocall.abi0")) == 0)) {
            sysprint("ERROR: can't get the address for runtime.cgocall\n");
            return; // don't install our hooks
        }
    } else {
        // Use the native abi - the only abi present in <= Go 1.16
        if (((go_runtime_cgocall = getSymbol(ebuf->buf, "runtime.asmcgocall")) == 0) &&
            ((go_runtime_cgocall = getGoSymbol(ebuf->buf, "runtime.asmcgocall")) == 0)) {
            sysprint("ERROR: can't get the address for runtime.cgocall\n");
            return; // don't install our hooks
        }
    }

    go_runtime_cgocall = (void *) ((uint64_t)go_runtime_cgocall + base);
    funcprint("asmcgocall %p\n", go_runtime_cgocall);

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

    // Update the schema to suit the current version
    adjustGoStructOffsetsForVersion(g_go_major_ver);
    if (g_go_major_ver > 16) {
        // The Go 17 schema works for 18 also, and possibly future versions
        g_go_schema = &go_17_schema;
    }

    for (tap_t *tap = g_go_schema->tap; tap->assembly_fn; tap++) {
        if (asm_inst) {
            cs_free(asm_inst, asm_count);
            asm_inst = NULL;
            asm_count = 0;
        }

        void* orig_func;
        // Look for the symbol in the ELF symbol table
        if (((orig_func = getSymbol(ebuf->buf, tap->func_name)) == NULL) &&
        // Otherwise look in the .gopclntab section
         ((orig_func = getGoSymbol(ebuf->buf, tap->func_name)) == NULL)) {
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
    for (tap_t *tap = g_go_schema->tap; tap->assembly_fn; tap++) {
        if (tap->assembly_fn == fn) return tap->return_addr;
    }

    scopeLogError("FATAL ERROR: no return addr");
    exit(-1);
}

static uint32_t
frame_size(assembly_fn fn)
{
    for (tap_t *tap = g_go_schema->tap; tap->assembly_fn; tap++) {
        if (tap->assembly_fn == fn) return tap->frame_size;
    }

    scopeLogError("FATAL ERROR: no frame size");
    exit(-1);
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
 * Specific handlers are defined as the c_xxx functions.
 * Example, there is a c_write() that handles extracting
 * details for write operations. The address of c_write
 * is passed to do_cfunc.
 *
 * All handlers take a single parameter, the stack address
 * from the interposed Go function. All params are passed
 * on the stack in the Go Runtime. No params passed in
 * registers. This means return values are also on the
 * stack. It also means that we need offsets into the stack
 * in order to know how to extract values from params and
 * return values.
 */
inline static void *
do_cfunc(char *stackptr, void *cfunc, void *gfunc)
{
    uint64_t rc;

    // We add the frame size to the stackptr to put our stackaddr in the context of the Caller.
    // We won't always know the frame size however. For example, when we patch an "xorps" instruction.
    // Therefore in some functions we will end up with a stack address in the context of
    // the Callee instead of the Caller.
    uint32_t frame_offset = frame_size(gfunc);
    stackptr += frame_offset;

    // Call the C handler
    __asm__ volatile (
        "mov %1, %%rdi  \n"
        "callq *%2  \n"
        : "=r"(rc)                    // output
        : "r"(stackptr), "r"(cfunc)   // inputs
        :                             // clobbered register
        );
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

// Extract data from syscall.write (write)
static void
c_write(char *stackaddr)
{
    uint64_t fd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_write_fd);
    char *buf   = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_write_buf));
    uint64_t rc = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_write_rc);
    uint64_t initialTime = getTime();

    funcprint("Scope: write fd %ld rc %ld buf %s\n", fd, rc, buf);
    doWrite(fd, initialTime, (rc != -1), buf, rc, "go_write", BUF, 0);

    free_go_str(buf);
}

EXPORTON void *
go_write(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_write, go_hook_reg_write);
    } else {
        return do_cfunc(stackptr, c_write, go_hook_write);
    }
}

// Extract data from syscall.Getdents (read dir)
static void
c_getdents(char *stackaddr)
{
    uint64_t dirfd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_getdents_dirfd);
    uint64_t rc    = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_getdents_rc);
    uint64_t initialTime = getTime();

    funcprint("Scope: getdents dirfd %ld rc %ld\n", dirfd, rc);
    doRead(dirfd, initialTime, (rc != -1), NULL, rc, "go_getdents", BUF, 0);
}

EXPORTON void *
go_getdents(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_getdents, go_hook_reg_getdents);
    } else {
        return do_cfunc(stackptr, c_getdents, go_hook_getdents);
    }
}

// Extract data from syscall.unlinkat (delete file)
static void
c_unlinkat(char *stackaddr)
{
    uint64_t dirfd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_unlinkat_dirfd);
    char *pathname = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_unlinkat_pathname));
    uint64_t flags = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_unlinkat_flags);

    if (!pathname) {
        scopeLogError("ERROR:go_unlinkat: null pathname");
        scope_puts("Scope:ERROR:unlinkat:no pathname");
        scope_fflush(scope_stdout);
        return;
    }

    funcprint("Scope: unlinkat dirfd %ld pathname %s flags %ld\n", dirfd, pathname, flags);
    doDelete(pathname, "go_unlinkat");

    free_go_str(pathname);
}

EXPORTON void *
go_unlinkat(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_unlinkat, go_hook_reg_unlinkat);
    } else {
        return do_cfunc(stackptr, c_unlinkat, go_hook_unlinkat);
    }
}

// Extract data from syscall.openat (file open)
// Deals with files only. net opens are handles by c_socket
static void
c_open(char *stackaddr)
{
    uint64_t fd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_open_fd);
    char *path  = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_open_path));

    if (!path) {
        scopeLogError("ERROR:go_open: null pathname");
        scope_puts("Scope:ERROR:open:no path");
        scope_fflush(scope_stdout);
        return;
    }

    funcprint("Scope: open of %ld\n", fd);
    doOpen(fd, path, FD, "open");

    free_go_str(path);
}

EXPORTON void *
go_open(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_open, go_hook_reg_open);
    } else {
        return do_cfunc(stackptr, c_open, go_hook_open);
    }
}

// Extract data from syscall.Close (close)
static void
c_close(char *stackaddr)
{
    uint64_t fd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_close_fd);
    uint64_t rc = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_close_rc);

    funcprint("Scope: close of %ld\n", fd);

    // If net, deletes a net object
    doCloseAndReportFailures(fd, (rc != -1), "go_close");
}

EXPORTON void *
go_close(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_close, go_hook_reg_close);
    } else {
        return do_cfunc(stackptr, c_close, go_hook_close);
    }
}

// Extract data from syscall.read (read)
static void
c_read(char *stackaddr)
{
    uint64_t fd = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_read_fd);
    char *buf   = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_read_buf));
    uint64_t rc = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_read_rc);
    uint64_t initialTime = getTime();

    if (rc == -1) {
        free_go_str(buf);
        return;
    }

    funcprint("Scope: read of %ld rc %ld\n", fd, rc);
    doRead(fd, initialTime, (rc != -1), buf, rc, "go_read", BUF, 0);

    free_go_str(buf);
}

EXPORTON void *
go_read(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_read, go_hook_reg_read);
    } else {
        return do_cfunc(stackptr, c_read, go_hook_read);
    }
}

// Extract data from syscall.socket (net open)
static void
c_socket(char *stackaddr)
{
    uint64_t domain = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_socket_domain);  // aka family
    uint64_t type   = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_socket_type);
    uint64_t sd     = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_socket_sd);

    if (sd == -1) return;

    funcprint("Scope: socket domain: %ld type: 0x%lx sd: %ld\n", domain, type, sd);

    // Creates a net object
    addSock(sd, type, domain);
}

EXPORTON void *
go_socket(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_socket, go_hook_reg_socket);
    } else {
        return do_cfunc(stackptr, c_socket, go_hook_socket);
    }
}

// Extract data from syscall.accept4 (plain server accept)
static void
c_accept4(char *stackaddr)
{
    uint64_t fd           = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_accept4_fd); 
    struct sockaddr *addr = *(struct sockaddr **)(stackaddr + g_go_schema->arg_offsets.c_accept4_addr);
    socklen_t *addrlen    = *(socklen_t **)(stackaddr + g_go_schema->arg_offsets.c_accept4_addrlen);
    uint64_t sd_out       = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_accept4_sd_out);

    if (sd_out != -1) {
        funcprint("Scope: accept4 of %ld\n", sd_out);
        doAccept(fd, sd_out, addr, addrlen, "go_accept4");
    }
}

EXPORTON void *
go_accept4(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_accept4, go_hook_reg_accept4);
    } else {
        return do_cfunc(stackptr, c_accept4, go_hook_accept4);
    }
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
  cr = stackaddr + 0x08
  cr.conn = *cr
  cr.conn.rwc_if = cr.conn + 0x10
  cr.conn.rwc = cr.conn.rwc_if + 0x08
  netFD = cr.conn.rwc + 0x08
  pfd = *netFD  (/usr/local/go/src/net/fd_unix.go:20)
  fd = netFD + 0x10
  type connReader struct {
        conn *conn
  type conn struct {
          server *Server
          cancelCtx context.CancelFunc
          rwc net.Conn
          remoteAddr string
          tlsState *tls.ConnectionState
 */
// Extract data from net/http.(*connReader).Read (tls server read)
static void
c_http_server_read(char *stackaddr)
{
    // Take us to the stack frame we're interested in
    // If this is defined as 0x0, we have decided to stay in the caller stack frame
    stackaddr -= g_go_schema->arg_offsets.c_http_server_read_callee;

    int fd = -1;

    uint64_t connReader = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_server_read_connReader); 
    if (!connReader) return;   // protect from dereferencing null
    char *buf           = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_http_server_read_buf));
    // buf len 0x18
    // buf cap 0x20
    uint64_t rc         = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_server_read_rc);
    uint64_t cr_conn_rwc_if, cr_conn_rwc, netFD, pfd;

    uint64_t conn =  *(uint64_t *)(connReader + g_go_schema->struct_offsets.connReader_to_conn);

    // protect from dereferencing null
    if (!conn) {
        free_go_str(buf);
        return;
    }

    cr_conn_rwc_if = conn + g_go_schema->struct_offsets.conn_to_rwc;
    uint64_t tls        = *(uint64_t *)(conn + g_go_schema->struct_offsets.conn_to_tlsState);

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
        cr_conn_rwc = *(uint64_t *)(cr_conn_rwc_if + g_go_schema->struct_offsets.iface_data);
        netFD = *(uint64_t *)(cr_conn_rwc + g_go_schema->struct_offsets.iface_data);
        if (netFD) {
            pfd = *(uint64_t *)(netFD + g_go_schema->struct_offsets.netfd_to_pd);
            if (pfd) {
                //funcprint("Scope: %s:%d cr %p cr.conn %p cr.conn.rwc_if %p cr.conn.rwc %p netFD %p pfd %p fd %p\n",
                //          __FUNCTION__, __LINE__, connReader, conn, cr_conn_rwc_if, cr_conn_rwc,
                //          netFD, pfd, pfd + g_go_schema->struct_offsets.pd_to_fd);
                fd = *(int *)(pfd + g_go_schema->struct_offsets.pd_to_fd);
            }

            funcprint("Scope: go_http_server_read of %d\n", fd);
            doProtocol((uint64_t)0, fd, buf, rc, TLSRX, BUF);
        }
    }

    free_go_str(buf);
}

EXPORTON void *
go_tls_read(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_http_server_read, go_hook_reg_tls_read);
    } else {
        return do_cfunc(stackptr, c_http_server_read, go_hook_tls_read);
    }
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
  conn (w.c) = stackaddr + 0x08
  conn.rwc_if = conn + 0x10
  conn.rwc = conn.rwc_if + 0x08
  netFD = conn.rwc + 0x08
  pfd = *netFD
  fd = pfd + 0x10
 */
// Extract data from net/http.checkConnErrorWriter.Write (tls server write)
static void
c_http_server_write(char *stackaddr)
{
    // Take us to the stack frame we're interested in
    // If this is defined as 0x0, we have decided to stay in the caller stack frame
    stackaddr -= g_go_schema->arg_offsets.c_http_server_write_callee;

    int fd = -1;
    uint64_t conn = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_server_write_conn);
    if (!conn) return;         // protect from dereferencing null
    char *buf     = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_http_server_write_buf));
    uint64_t rc   = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_server_write_rc);
    uint64_t w_conn_rwc_if, w_conn_rwc, netFD, pfd;

    w_conn_rwc_if = (conn + g_go_schema->struct_offsets.conn_to_rwc);
    uint64_t tls =  *(uint64_t *)(conn + g_go_schema->struct_offsets.conn_to_tlsState);

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (w_conn_rwc_if && tls) {
        w_conn_rwc = *(uint64_t *)(w_conn_rwc_if + g_go_schema->struct_offsets.iface_data);
        netFD = *(uint64_t *)(w_conn_rwc + g_go_schema->struct_offsets.iface_data);
        if (netFD) {
            pfd = *(uint64_t *)(netFD + g_go_schema->struct_offsets.netfd_to_pd);
            if (pfd) {
                fd = *(int *)(pfd + g_go_schema->struct_offsets.pd_to_fd);
            }

            funcprint("Scope: c_http_server_write of %d\n", fd);
            doProtocol((uint64_t)0, fd, buf, rc, TLSTX, BUF);
        }
    }

    free_go_str(buf);
}

EXPORTON void *
go_tls_write(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_http_server_write, go_hook_reg_tls_write);
    } else {
        return do_cfunc(stackptr, c_http_server_write, go_hook_tls_write);
    }
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
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
// Extract data from net/http.persistConnWriter.Write (tls client write)
static void
c_http_client_write(char *stackaddr)
{
    // Take us to the stack frame we're interested in
    // If this is defined as 0x0, we have decided to stay in the caller stack frame
    stackaddr -= g_go_schema->arg_offsets.c_http_client_write_callee;

    int fd = -1;
    uint64_t w_pc = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_client_write_w_pc);
    char *buf     = go_str((void *)(stackaddr + g_go_schema->arg_offsets.c_http_client_write_buf));
    uint64_t rc   = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_client_write_rc);
    uint64_t pc_conn_if, w_pc_conn, netFD, pfd;

    if (rc < 1) {
        free_go_str(buf);
        return;
    }

    pc_conn_if = (w_pc + g_go_schema->struct_offsets.persistConn_to_conn); 
    uint64_t tls =  *(uint64_t*)(w_pc + g_go_schema->struct_offsets.persistConn_to_tlsState); 

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (pc_conn_if && tls) {
        w_pc_conn = *(uint64_t *)(pc_conn_if + g_go_schema->struct_offsets.iface_data); 
        netFD = *(uint64_t *)(w_pc_conn + g_go_schema->struct_offsets.iface_data);
        if (!netFD) return;
        if (g_go_schema->struct_offsets.netfd_to_sysfd == UNDEF_OFFSET) { 
            pfd = *(uint64_t *)(netFD + g_go_schema->struct_offsets.netfd_to_pd); 
            if (!pfd) return;
            fd = *(int *)(pfd + g_go_schema->struct_offsets.pd_to_fd); 
        } else {
            fd = *(int *)(netFD + g_go_schema->struct_offsets.netfd_to_sysfd); 
        }

        doProtocol((uint64_t)0, fd, buf, rc, TLSRX, BUF);
        funcprint("Scope: c_http_client_write of %d\n", fd);
    }

    free_go_str(buf);
}

EXPORTON void *
go_pc_write(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_http_client_write, go_hook_reg_pc_write);
    } else {
        return do_cfunc(stackptr, c_http_client_write, go_hook_pc_write);
    }
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
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
// Extract data from net/http.(*persistConn).readResponse (tls server read)
static void
c_http_client_read(char *stackaddr)
{
    // Take us to the stack frame we're interested in
    // If this is defined as 0x0, we have decided to stay in the caller stack frame
    stackaddr -= g_go_schema->arg_offsets.c_http_client_read_callee;

    int fd = -1;
    stackaddr += g_go_schema->arg_offsets.c_http_client_read_callee;
    uint64_t pc  = *(uint64_t *)(stackaddr + g_go_schema->arg_offsets.c_http_client_read_pc); 
    uint64_t pc_conn_if, pc_conn, netFD, pfd, pc_br, len = 0;
    char *buf = NULL;

    pc_conn_if = (pc + g_go_schema->struct_offsets.persistConn_to_conn);
    uint64_t tls = *(uint64_t*)(pc + g_go_schema->struct_offsets.persistConn_to_tlsState);

    // conn I/F checking. Ref the comment on c_http_server_read.
    if (pc_conn_if && tls) {
        pc_conn = *(uint64_t *)(pc_conn_if + g_go_schema->struct_offsets.iface_data);
        netFD = *(uint64_t *)(pc_conn + g_go_schema->struct_offsets.iface_data);
        if (!netFD) return;
        if (g_go_schema->struct_offsets.netfd_to_sysfd == UNDEF_OFFSET) {
            pfd = *(uint64_t *)(netFD + g_go_schema->struct_offsets.netfd_to_pd);
            if (!pfd) return;
            fd = *(int *)(pfd + g_go_schema->struct_offsets.pd_to_fd);
        } else {
            fd = *(int *)(netFD + g_go_schema->struct_offsets.netfd_to_sysfd);
        }

        if ((pc_br = *(uint64_t *)(pc + g_go_schema->struct_offsets.persistConn_to_bufrd)) != 0) {
            buf = go_str((void *)(pc_br + g_go_schema->struct_offsets.bufrd_to_buf));
            // len is part of the []byte struct; the func doesn't return a len
            len = *(uint64_t *)(pc_br + 0x08);
        }

        if (buf && (len > 0)) {
            doProtocol((uint64_t)0, fd, buf, len, TLSRX, BUF);
            funcprint("Scope: c_http_client_read of %d\n", fd);
        }

        free_go_str(buf);
    }
}

EXPORTON void *
go_readResponse(char *stackptr)
{
    if (g_go_major_ver > 16) {
        return do_cfunc(stackptr, c_http_client_read, go_hook_reg_readResponse);
    } else {
        return do_cfunc(stackptr, c_http_client_read, go_hook_readResponse);
    }
}

extern void handleExit(void);
static void
c_exit(char *stackaddr)
{
    /*
     * Need to extend the system stack size when calling handleExit().
     * We see that the stack is exceeded now that we are using an internal libc.
     */
    int arc;
    char *exit_stack, *tstack, *gstack;
    if ((exit_stack = scope_malloc(EXIT_STACK_SIZE)) == NULL) {
        return;
    }

    tstack = exit_stack + EXIT_STACK_SIZE;

    // save the original stack, switch to the tstack
    __asm__ volatile (
        "mov %%rsp, %2 \n"
        "mov %1, %%rsp \n"
        : "=r"(arc)                  // output
        : "m"(tstack), "m"(gstack)   // input
        :                            // clobbered register
        );

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

    // Switch stack back to the original stack
    __asm__ volatile (
        "mov %1, %%rsp \n"
        : "=r"(arc)                       // output
        : "r"(gstack)                     // inputs
        :                                 // clobbered register
        );

    scope_free(exit_stack);
}

EXPORTON void *
go_exit(char *stackptr)
{
    return do_cfunc(stackptr, c_exit, go_hook_exit);
}

EXPORTON void *
go_die(char *stackptr)
{
    return do_cfunc(stackptr, c_exit, go_hook_die);
}
