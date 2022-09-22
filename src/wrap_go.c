#define _GNU_SOURCE
#include <sys/mman.h>
#include <asm/prctl.h>
#include <sys/prctl.h>
#include <signal.h>

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

#define GOPCLNTAB_MAGIC_112 0xfffffffb
#define GOPCLNTAB_MAGIC_116 0xfffffffa
#define GOPCLNTAB_MAGIC_118 0xfffffff0
#define SCOPE_STACK_SIZE (size_t)(32 * 1024)
#define EXIT_STACK_SIZE (32 * 1024)
#define UNKNOWN_GO_VER (-1)
#define MIN_SUPPORTED_GO_VER (9)
#define MAX_SUPPORTED_GO_VER (19)
#define HTTP2_FRAME_HEADER_LEN (9)
#define PRI_STR "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
#define PRI_STR_LEN sizeof(PRI_STR)
#define UNDEF_OFFSET (-1)

// compile-time control for debugging
#define NEEDEVNULL 1
//#define funcprint sysprint
#define funcprint devnull
//#define patchprint sysprint
#define patchprint devnull

int g_go_minor_ver = UNKNOWN_GO_VER;
int g_go_maint_ver = UNKNOWN_GO_VER;
static char g_go_build_ver[7];
static char g_ReadFrame_addr[sizeof(void *)];
go_schema_t *g_go_schema = &go_9_schema; // overridden if later version
uint64_t g_glibc_guard = 0LL;
uint64_t go_systemstack_switch;

#if NEEDEVNULL > 0
static void
devnull(const char* fmt, ...)
{
    return;
}
#endif

enum index_hook_t {
    INDEX_HOOK_SYSCALL,
    INDEX_HOOK_RAWSYSCALL,
    INDEX_HOOK_SYSCALL6,
    INDEX_HOOK_TLS_CLIENT_READ,
    INDEX_HOOK_TLS_CLIENT_WRITE,
    INDEX_HOOK_TLS_SERVER_READ,
    INDEX_HOOK_TLS_SERVER_WRITE,
    INDEX_HOOK_HTTP2_CLIENT_READ,
    INDEX_HOOK_HTTP2_CLIENT_WRITE,
    INDEX_HOOK_HTTP2_SERVER_READ,
    INDEX_HOOK_HTTP2_SERVER_WRITE,
    INDEX_HOOK_HTTP2_SERVER_PREFACE,
    INDEX_HOOK_EXIT,
    INDEX_HOOK_DIE,
    INDEX_HOOK_MAX,
};

go_schema_t go_9_schema = {
    .arg_offsets = {
        .c_syscall_rc=0x0,
        .c_syscall_num=0x60,
        .c_syscall_p1=0x20,
        .c_syscall_p2=0x28,
        .c_syscall_p3=0x18,
        .c_syscall_p4=0x10,
        .c_syscall_p5=0x30,
        .c_syscall_p6=0x38,
        .c_tls_server_read_connReader=0x8,
        .c_tls_server_read_buf=0x10,
        .c_tls_server_read_rc=0x28,
        .c_tls_server_write_conn=0x8,
        .c_tls_server_write_buf=0x10,
        .c_tls_server_write_rc=0x28,
        .c_tls_client_read_pc=0x8,
        .c_tls_client_write_w_pc=0x8,
        .c_tls_client_write_buf=0x10,
        .c_tls_client_write_rc=0x28,
        .c_http2_server_read_sc=0x188,
        .c_http2_server_write_sc=0x8,
        .c_http2_server_preface_callee=0x108,
        .c_http2_server_preface_sc=0x110,
        .c_http2_server_preface_rc=0x120,
        .c_http2_client_read_cc=0x78,
        .c_http2_client_write_tcpConn=0x10,
        .c_http2_client_write_buf=0x68,
        .c_http2_client_write_rc=0x28,
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
        .netfd_to_sysfd=UNDEF_OFFSET,
        .bufrd_to_buf=0x0,
        .conn_to_rwc=0x10,
        .conn_to_tlsState=0x30,
        .persistConn_to_tlsState=0x60,
        .fr_to_readBuf=0x50,
        .fr_to_writeBuf=0x80,
        .fr_to_headerBuf=0x38,
        .cc_to_fr=0xc8,
        .cc_to_tconn=0x8,
        .sc_to_fr=0x48,
        .sc_to_conn=0x10,
    },
    .tap = {
        [INDEX_HOOK_SYSCALL]              = {"syscall.Syscall",                         go_hook_reg_syscall,          NULL, 0},
        [INDEX_HOOK_RAWSYSCALL]           = {"syscall.RawSyscall",                      go_hook_reg_rawsyscall,       NULL, 0},
        [INDEX_HOOK_SYSCALL6]             = {"syscall.Syscall6",                        go_hook_reg_syscall6,         NULL, 0},
        [INDEX_HOOK_TLS_CLIENT_READ]      = {"net/http.(*persistConn).readResponse",    go_hook_reg_tls_client_read,      NULL, 0}, 
        [INDEX_HOOK_TLS_CLIENT_WRITE]     = {"net/http.persistConnWriter.Write",        go_hook_reg_tls_client_write,     NULL, 0},
        [INDEX_HOOK_TLS_SERVER_READ]      = {"net/http.(*connReader).Read",             go_hook_reg_tls_server_read,      NULL, 0},
        [INDEX_HOOK_TLS_SERVER_WRITE]     = {"net/http.checkConnErrorWriter.Write",     go_hook_reg_tls_server_write,     NULL, 0},
        [INDEX_HOOK_HTTP2_CLIENT_READ]    = {"net/http.(*http2clientConnReadLoop).run", go_hook_reg_http2_client_read,    NULL, 0},
        [INDEX_HOOK_HTTP2_CLIENT_WRITE]   = {"net/http.http2stickyErrWriter.Write",     go_hook_reg_http2_client_write,   NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_READ]    = {"net/http.(*http2serverConn).readFrames",  go_hook_reg_http2_server_read,    NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_WRITE]   = {"net/http.(*http2serverConn).Flush",       go_hook_reg_http2_server_write,   NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_PREFACE] = {"net/http.(*http2serverConn).readPreface", go_hook_reg_http2_server_preface, NULL, 0},
        [INDEX_HOOK_EXIT]                 = {"runtime.exit",                            go_hook_exit,                 NULL, 0},
        [INDEX_HOOK_DIE]                  = {"runtime.dieFromSignal",                   go_hook_die,                  NULL, 0},
        [INDEX_HOOK_MAX]                  = {"TAP_TABLE_END",                           NULL,                         NULL, 0}
    },
};

go_schema_t go_17_schema = {
    .arg_offsets = {
        .c_syscall_rc=0x0,
        .c_syscall_num=0x60,
        .c_syscall_p1=0x20,
        .c_syscall_p2=0x28,
        .c_syscall_p3=0x18,
        .c_syscall_p4=0x10,
        .c_syscall_p5=0x30,
        .c_syscall_p6=0x38,
        .c_tls_server_read_connReader=0x50,
        .c_tls_server_read_buf=0x8,
        .c_tls_server_read_rc=0x28,
        .c_tls_server_write_conn=0x30,
        .c_tls_server_write_buf=0x8,
        .c_tls_server_write_rc=0x10,
        .c_tls_client_read_pc=0x28,
        .c_tls_client_write_w_pc=0x20,
        .c_tls_client_write_buf=0x8,
        .c_tls_client_write_rc=0x10,
        .c_http2_server_read_sc=0xd0,
        .c_http2_server_write_sc=0x40,
        .c_http2_server_preface_sc=0xd0,
        .c_http2_server_preface_rc=0x58,
        .c_http2_client_read_cc=0x68,
        .c_http2_client_write_tcpConn=0x40,
        .c_http2_client_write_buf=0x8,
        .c_http2_client_write_rc=0x10,
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
        .netfd_to_sysfd=UNDEF_OFFSET,
        .bufrd_to_buf=0x0,
        .conn_to_rwc=0x10,
        .conn_to_tlsState=0x30,
        .persistConn_to_tlsState=0x60,
        .fr_to_readBuf=0x58,
        .fr_to_writeBuf=0x88,
        .fr_to_headerBuf=0x40,
        .cc_to_fr=0x130,
        .cc_to_tconn=0x8,
        .sc_to_fr=0x48,
        .sc_to_conn=0x10,
    },
    .tap = {
        [INDEX_HOOK_SYSCALL]              = {"syscall.Syscall",       /* .abi0 */       go_hook_reg_syscall,              NULL, 0},
        [INDEX_HOOK_RAWSYSCALL]           = {"syscall.RawSyscall",    /* .abi0 */       go_hook_reg_rawsyscall,           NULL, 0},
        [INDEX_HOOK_SYSCALL6]             = {"syscall.Syscall6",      /* .abi0 */       go_hook_reg_syscall6,             NULL, 0},
        [INDEX_HOOK_TLS_CLIENT_READ]      = {"net/http.(*persistConn).readResponse",    go_hook_reg_tls_client_read,      NULL, 0},
        [INDEX_HOOK_TLS_CLIENT_WRITE]     = {"net/http.persistConnWriter.Write",        go_hook_reg_tls_client_write,     NULL, 0},
        [INDEX_HOOK_TLS_SERVER_READ]      = {"net/http.(*connReader).Read",             go_hook_reg_tls_server_read,      NULL, 0},
        [INDEX_HOOK_TLS_SERVER_WRITE]     = {"net/http.checkConnErrorWriter.Write",     go_hook_reg_tls_server_write,     NULL, 0},
        [INDEX_HOOK_HTTP2_CLIENT_READ]    = {"net/http.(*http2clientConnReadLoop).run", go_hook_reg_http2_client_read,    NULL, 0},
        [INDEX_HOOK_HTTP2_CLIENT_WRITE]   = {"net/http.http2stickyErrWriter.Write",     go_hook_reg_http2_client_write,   NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_READ]    = {"net/http.(*http2serverConn).readFrames",  go_hook_reg_http2_server_read,    NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_WRITE]   = {"net/http.(*http2serverConn).Flush",       go_hook_reg_http2_server_write,   NULL, 0},
        [INDEX_HOOK_HTTP2_SERVER_PREFACE] = {"net/http.(*http2serverConn).readPreface", go_hook_reg_http2_server_preface, NULL, 0},
        [INDEX_HOOK_EXIT]                 = {"runtime.exit",          /* .abi0 */       go_hook_exit,                     NULL, 0},
        [INDEX_HOOK_DIE]                  = {"runtime.dieFromSignal", /* .abi0 */       go_hook_die,                      NULL, 0},
        [INDEX_HOOK_MAX]                  = {"TAP_TABLE_END",                           NULL,                             NULL, 0}
    },
};

static void
adjustGoStructOffsetsForVersion()
{
    if (!g_go_minor_ver) {
        sysprint("ERROR: can't determine minor go version\n");
        return;
    }

    if (g_go_minor_ver == 10) {
        g_go_schema->arg_offsets.c_http2_client_read_cc=0x78;
        g_go_schema->arg_offsets.c_http2_server_read_sc=0x188;
        g_go_schema->arg_offsets.c_http2_server_preface_callee=0x108;
        g_go_schema->arg_offsets.c_http2_server_preface_sc=0x110;
        g_go_schema->arg_offsets.c_http2_server_preface_rc=0x120;
        g_go_schema->struct_offsets.cc_to_fr=0xd0;
    }

    if (g_go_minor_ver < 12) {
        g_go_schema->struct_offsets.persistConn_to_conn = 72;  // 0x48
        g_go_schema->struct_offsets.persistConn_to_bufrd = 96; // 0x60
        g_go_schema->struct_offsets.persistConn_to_tlsState=88; // 0x58
    }

    if ((g_go_minor_ver == 11) || (g_go_minor_ver == 12) || (g_go_minor_ver == 13) ||
        (g_go_minor_ver == 14) || (g_go_minor_ver == 15)) {
        g_go_schema->arg_offsets.c_http2_client_read_cc=0x78;
        g_go_schema->arg_offsets.c_http2_server_read_sc=0x128;
        g_go_schema->arg_offsets.c_http2_server_preface_callee=0x108;
        g_go_schema->arg_offsets.c_http2_server_preface_sc=0x110;
        g_go_schema->arg_offsets.c_http2_server_preface_rc=0x120;
        g_go_schema->struct_offsets.cc_to_fr=0xd0;
    }

    if (g_go_minor_ver == 16) {
        g_go_schema->arg_offsets.c_http2_client_read_cc=0xe0;
        g_go_schema->arg_offsets.c_http2_server_read_sc=0xe8;
        g_go_schema->arg_offsets.c_http2_server_preface_callee=0x108;
        g_go_schema->arg_offsets.c_http2_server_preface_sc=0x110;
        g_go_schema->arg_offsets.c_http2_server_preface_rc=0x120;
        g_go_schema->struct_offsets.cc_to_fr=0xd0;

        if (g_go_maint_ver > 9) {
            g_go_schema->struct_offsets.cc_to_fr=0x130;
        }
    }

    if (g_go_minor_ver == 17) {
        g_go_schema->struct_offsets.fr_to_readBuf=0x50;
        g_go_schema->struct_offsets.fr_to_writeBuf=0x80;
        g_go_schema->struct_offsets.fr_to_headerBuf=0x38;
        
        if (g_go_maint_ver < 3) {
            g_go_schema->struct_offsets.cc_to_fr=0xd0;
        }
    }

    if (g_go_minor_ver == 18) {
        g_go_schema->arg_offsets.c_tls_client_read_pc=0x80;
        g_go_schema->arg_offsets.c_http2_client_write_tcpConn=0x48;
    }

    if (g_go_minor_ver == 19) {
        g_go_schema->arg_offsets.c_tls_client_read_pc=0x80;
        g_go_schema->arg_offsets.c_http2_client_write_tcpConn=0x48;

        g_go_schema->tap[INDEX_HOOK_SYSCALL].func_name = "runtime/internal/syscall.Syscall6";
        g_go_schema->tap[INDEX_HOOK_RAWSYSCALL].func_name = "";
        g_go_schema->tap[INDEX_HOOK_SYSCALL6].func_name = "";
    }
}

// This creates a file specified by test/integration/go/test_go.sh
// and used by test/integration/go/test_go_struct.sh.
// Why?  To test structure offsets in go that can vary. (above)
// The format is:
//   StructureName|FieldName=DecimalOffsetValue|OptionalTag
// If an OptionalTag is provided, test_go_struct.sh will not process
// the line unless it matches a TAG_FILTER which is provided as an
// argument to the test_go_struct.sh.
void
createGoStructFile(void) {
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
        scope_dprintf(fd, "net.netFD|pfd=%d|\n", g_go_schema->struct_offsets.netfd_to_pd);
        scope_dprintf(fd, "internal/poll.FD|Sysfd=%d|\n", g_go_schema->struct_offsets.pd_to_fd);
        scope_dprintf(fd, "bufio.Reader|buf=%d|\n", g_go_schema->struct_offsets.bufrd_to_buf);
        scope_dprintf(fd, "net/http.conn|rwc=%d|Server\n", g_go_schema->struct_offsets.conn_to_rwc);
        scope_dprintf(fd, "net/http.conn|tlsState=%d|Server\n", g_go_schema->struct_offsets.conn_to_tlsState);
        scope_dprintf(fd, "net/http.persistConn|tlsState=%d|Client\n", g_go_schema->struct_offsets.persistConn_to_tlsState);
        scope_dprintf(fd, "net/http.http2Framer|readBuf=%d|Server\n", g_go_schema->struct_offsets.fr_to_readBuf);
        scope_dprintf(fd, "net/http.http2Framer|wbuf=%d|Server\n", g_go_schema->struct_offsets.fr_to_writeBuf);
        scope_dprintf(fd, "net/http.http2Framer|headerBuf=%d|Server\n", g_go_schema->struct_offsets.fr_to_headerBuf);
        scope_dprintf(fd, "net/http.http2ClientConn|fr=%d|Client\n", g_go_schema->struct_offsets.cc_to_fr);
        scope_dprintf(fd, "net/http.http2ClientConn|tconn=%d|Client\n", g_go_schema->struct_offsets.cc_to_tconn);
        scope_dprintf(fd, "net/http.http2serverConn|framer=%d|Server\n", g_go_schema->struct_offsets.sc_to_fr);
        scope_dprintf(fd, "net/http.http2serverConn|conn=%d|Server\n", g_go_schema->struct_offsets.sc_to_conn);
        scope_close(fd);
    }
}

// Use go_str() whenever a "go string" type needs to be interpreted.
// The resulting go_str will need to be passed to free_go_str() when it is no
// longer needed.
// Don't use go_str() for byte arrays.
static char *
go_str(void *go_str)
{
    // Go 17 and higher use "c style" null terminated strings instead of a string and a length
    if (g_go_minor_ver >= 17) {
       // We need to deference the address first before casting to a char *
       if (!go_str) return NULL;
       return (char *)*(uint64_t *)go_str;
    }

    gostring_t* go_str_tmp = (gostring_t *)go_str;
    if (!go_str_tmp || go_str_tmp->len <= 0) return NULL;

    char *c_str;
    if ((c_str = scope_calloc(1, go_str_tmp->len+1)) == NULL) return NULL;
    scope_memmove(c_str, go_str_tmp->str, go_str_tmp->len);
    c_str[go_str_tmp->len] = '\0';

    return c_str;
}

/* 
static void
free_go_str(char *str) {
    // Go 17 and higher use "c style" null terminated strings instead of a string and a length
    if (g_go_minor_ver >= 17) {
        return;
    }
    if(str) scope_free(str);
}
*/

static bool
match_assy_instruction(void *addr, char *mnemonic)
{
    csh dhandle = 0;
    cs_arch arch;
    cs_mode mode;
    cs_insn *asm_inst = NULL;
    unsigned int asm_count = 0;
    uint64_t size = 32;
    bool rc = FALSE;

#if defined(__aarch64__)
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_LITTLE_ENDIAN;
#elif defined(__x86_64__)
    arch = CS_ARCH_X86;
    mode = CS_MODE_64;
#else
    return FALSE;
#endif

    if (cs_open(arch, mode, &dhandle) != CS_ERR_OK) return FALSE;

    asm_count = cs_disasm(dhandle, addr, size, (uint64_t)addr, 0, &asm_inst);
    if (asm_count <= 0) return FALSE;

    if (!scope_strcmp((const char*)asm_inst->mnemonic, mnemonic)) rc = TRUE;

    if (asm_inst) cs_free(asm_inst, asm_count);
    cs_close(&dhandle);

    return rc;
}

static void *
getGoVersionAddr(const char* buf)
{
    int i;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name;
    const char *sec_data;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;
    const char magic[0xe] = "\xff Go buildinf:";
    void *go_build_ver_addr = NULL;
 
    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        sec_data = (const char *)buf + sections[i].sh_offset;
        // Since go1.13, the .go.buildinfo section has been added to
        // identify where runtime.buildVersion exists, for the case where
        // go apps have been stripped of their symbols.

        // offset into sec_data     field contents
        // -----------------------------------------------------------
        // 0x0                      build info magic = "\xff Go buildinf:"
        // 0xe                      binary ptrSize
        // 0xf                      endianness
        // 0x10                     pointer to string runtime.buildVersion
        // 0x10 + ptrSize           pointer to runtime.modinfo
        // 0x10 + 2 * ptr size      pointer to build flags

        if (!scope_strcmp(sec_name, ".go.buildinfo") &&
            (sections[i].sh_size >= 0x18) &&
            (!scope_memcmp(&sec_data[0], magic, sizeof(magic))) &&
            (sec_data[0xe] == 0x08)) {  // 64 bit executables only

            // debug/buildinfo/buildinfo.go
            // If the endianness has the 2 bit set, then the pointers are zero
            // and the 32-byte header is followed by varint-prefixed string data
            // for the two string values we care about.
            if (sec_data[0xf] == 0x00) {  // little-endian
                uint64_t *addressPtr = (uint64_t*)&sec_data[0x10];
                go_build_ver_addr = (void*)*addressPtr;
            } else if (sec_data[0xf] == 0x02) {
                scope_memmove(g_go_build_ver, (char*)&sec_data[0x21], 6);
                g_go_build_ver[6] = '\0';
                go_build_ver_addr = &g_go_build_ver;
            }
        }
    }
    return go_build_ver_addr;
}

static void *
getGoSymbol(const char *buf, char *sname, char *altname, char *mnemonic)
{
    int i;
    Elf64_Addr symaddr = 0;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    if (!buf || !sname) return NULL;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)((char *)buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        if (scope_strstr(sec_name, ".gopclntab")) {
            const void *pclntab_addr = buf + sections[i].sh_offset;
            /*
            Go symbol table is stored in the .gopclntab section
            More info: https://docs.google.com/document/d/1lyPIbmsYbXnpNj57a261hgOYVpNRcgydurVQIyZOz_o/pub
            */
            uint32_t magic = *((const uint32_t *)(pclntab_addr));
            if (magic == GOPCLNTAB_MAGIC_112) {
                uint64_t sym_count      = *((const uint64_t *)(pclntab_addr + 8));
                const void *symtab_addr = pclntab_addr + 16;

                for(i=0; i<sym_count; i++) {
                    uint64_t func_offset  = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset  = *((const uint32_t *)(pclntab_addr + func_offset + 8));
                    uint64_t sym_addr     = *((const uint64_t *)(symtab_addr));
                    const char *func_name = (const char *)(pclntab_addr + name_offset);

                    if (scope_strcmp(sname, func_name) == 0) {
                        symaddr = sym_addr;
                        scopeLog(CFG_LOG_TRACE, "symbol found %s = 0x%08lx\n", func_name, sym_addr);
                        break;
                    }

                    symtab_addr += 16;
                }
            } else if (magic == GOPCLNTAB_MAGIC_116) {
                uint64_t sym_count      = *((const uint64_t *)(pclntab_addr + 8));
                uint64_t funcnametab_offset = *((const uint64_t *)(pclntab_addr + (3 * 8)));
                uint64_t pclntab_offset = *((const uint64_t *)(pclntab_addr + (7 * 8)));
                const void *symtab_addr = pclntab_addr + pclntab_offset;
                for (i = 0; i < sym_count; i++) {
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 8));
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    if (scope_strcmp(sname, func_name) == 0) {
                        symaddr = sym_addr;
                        scopeLog(CFG_LOG_TRACE, "symbol found %s = 0x%08lx\n", func_name, sym_addr);
                        break;
                    }

                    // In go 1.17+ we need to ensure we find the correct symbol in the case of ambiguity
                    if (altname && mnemonic &&
                        (scope_strcmp(altname, func_name) == 0) &&
                        (match_assy_instruction((void *)sym_addr, mnemonic) == TRUE)) {
                        symaddr = sym_addr;
                        break;
                    }

                    symtab_addr += 16;
                }
            } else if (magic == GOPCLNTAB_MAGIC_118) {
                uint64_t sym_count = *((const uint64_t *)(pclntab_addr + 8));
                // In go 1.18 the funcname table and the pcln table are stored in the text section
                uint64_t text_start = *((const uint64_t *)(pclntab_addr + (3 * 8)));
                uint64_t funcnametab_offset = *((const uint64_t *)(pclntab_addr + (4 * 8)));
                //uint64_t funcnametab_addr = (uint64_t)(funcnametab_offset + pclntab_addr);
                uint64_t pclntab_offset = *((const uint64_t *)(pclntab_addr + (8 * 8)));
                // A "symbtab" is an entry in the pclntab, probably better known as a pcln
                const void *symtab_addr = (const void *)(pclntab_addr + pclntab_offset);
                for (i = 0; i < sym_count; i++) {
                    uint32_t func_offset = *((uint32_t *)(symtab_addr + 4));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 4));
                    func_offset = *((uint32_t *)(symtab_addr));
                    uint64_t sym_addr = (uint64_t)(func_offset + text_start);
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    if (scope_strcmp(sname, func_name) == 0) {
                        symaddr = sym_addr;
                        scopeLog(CFG_LOG_ERROR, "symbol found %s = 0x%08lx\n", func_name, sym_addr);
                        break;
                    }

                    // In go 1.17+ we need to ensure we find the correct symbol in the case of ambiguity
                    if (altname && mnemonic &&
                        (scope_strcmp(altname, func_name) == 0) &&
                        (match_assy_instruction((void *)sym_addr, mnemonic) == TRUE)) {
                        symaddr = sym_addr;
                        break;
                    }

                    symtab_addr += 8;
                }
            } else {
                scopeLog(CFG_LOG_DEBUG, "Invalid header in .gopclntab");
                break;
            }
            break;
        }
    }

    return (void *)symaddr;
}

// Detect the beginning of a Go Function
// by identifying instructions in the preamble.
static bool
looks_like_first_inst_of_go_func(cs_insn* asm_inst)
{
    return (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rcx, qword ptr fs:[0xfffffffffffffff8]")) ||
            // -buildmode=pie compiles to this:
            (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rcx, -8")) || 
            (!scope_strcmp((const char*)asm_inst->mnemonic, "cmp") &&
            !scope_strcmp((const char*)asm_inst->op_str, "rsp, qword ptr [r14 + 0x10]")) ||
            (!scope_strcmp((const char*)asm_inst->mnemonic, "lea") &&
            scope_strstr((const char*)asm_inst->op_str, "r12, [rsp - ")) ||
            (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "r10, rsi")) ||
            (!scope_strcmp((const char*)asm_inst->mnemonic, "mov") &&
            !scope_strcmp((const char*)asm_inst->op_str, "edi, dword ptr [rsp + 8]"))
            ;
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

// Patch all intended addresses
static void
patch_addrs(funchook_t *funchook,
                   cs_insn* asm_inst, unsigned int asm_count, tap_t* tap)
{
    if (!funchook || !asm_inst || !asm_count || !tap) return;

    uint32_t add_arg = 0;
    for (int i=0; i<asm_count; i++) {
        add_arg = 0;

        patchprint("%0*lx (%02d) %-24s %s %s\n",
               16, asm_inst[i].address, asm_inst[i].size,
               (char*)asm_inst[i].bytes, (char*)asm_inst[i].mnemonic,
               (char*)asm_inst[i].op_str);

        // Stop when it looks like we've hit another goroutine
        if (i > 0 && (looks_like_first_inst_of_go_func(&asm_inst[i]) ||
            (!scope_strcmp((const char*)asm_inst[i].mnemonic, "int3") &&
            asm_inst[i].size == 1 ))) {
            break;
        }

        // PATCH FIRST INSTRUCTION
        // Special handling for runtime.exit, runtime.dieFromSignal
        // Since the go folks wrote them in assembly, they don't follow
        // conventions that other go functions do.
        // We also patch syscalls at the first (and last) instruction.
        if (i == 0 && ((tap->assembly_fn == go_hook_exit) ||
            (tap->assembly_fn == go_hook_die))) {

            // In this case we want to patch the instruction directly
            void *pre_patch_addr = (void*)asm_inst[i].address;
            void *patch_addr = (void*)asm_inst[i].address;

            if (funchook_prepare(funchook, (void**)&patch_addr, tap->assembly_fn)) {
                patchprint("failed to patch 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
                continue;
            }

            patchprint("patched 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
            tap->return_addr = patch_addr;
            tap->frame_size = add_arg;

            break; // Done patching
        }

        // PATCH SYSCALLS
        if (!scope_strcmp((const char*)asm_inst[i].mnemonic, "syscall")) {

            // In the "syscall" case, we want to patch the instruction directly 
            void *pre_patch_addr = (void*)asm_inst[i].address;
            void *patch_addr = (void*)asm_inst[i].address;

            if (funchook_prepare(funchook, (void**)&patch_addr, tap->assembly_fn)) {
                patchprint("failed to patch 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
                continue;
            }

            patchprint("patched 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
            tap->return_addr = patch_addr;
            tap->frame_size = add_arg;

            break; // Done patching
        }

        // PATCH SPECIAL CALL INSTRUCTION
        // In the case of some functions, we want to patch just after a "call" instruction.
        // Note: We don't need a frame size here.
        if ((!scope_strcmp(tap->func_name, "net/http.(*http2serverConn).readFrames")) || 
            (!scope_strcmp(tap->func_name, "net/http.(*http2clientConnReadLoop).run"))) {
            if (
            (!scope_strcmp((const char*)asm_inst[i].mnemonic, "call")) &&
            (scope_strstr(g_ReadFrame_addr, (const char*)asm_inst[i].op_str)) &&
            (asm_inst[i].size == 5)) {

                // In the "call" case, we want to patch the instruction after the call
                void *pre_patch_addr = (void*)asm_inst[i+1].address;
                void *patch_addr = (void*)asm_inst[i+1].address;

                if (funchook_prepare(funchook, (void**)&patch_addr, tap->assembly_fn)) {
                    patchprint("failed to patch 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
                    continue;
                }

                patchprint("patched 0x%p with frame size 0x%x\n", pre_patch_addr, add_arg);
                tap->return_addr = patch_addr;
                tap->frame_size = add_arg;

                break; // Done patching
            }
        }

        // PATCH JUST BEFORE RET INSTRUCTION
        // If the current instruction is a RET
        // and previous inst is add or sub, then get the stack frame size.
        // Or, if the current inst is xorps then proceed without a stack frame size.
        else if ((!scope_strcmp((const char*)asm_inst[i].mnemonic, "ret")) &&
            (asm_inst[i].size == 1) &&
            ((!scope_strcmp((const char*)asm_inst[i-1].mnemonic, "add")) ||
            (!scope_strcmp((const char*)asm_inst[i-1].mnemonic, "sub"))) &&
            (add_arg = add_argument(&asm_inst[i-1]))) {

            // In the "ret" case, we want to patch previous instruction (to maintain the callee stack context)
            void *pre_patch_addr = (void*)asm_inst[i-1].address;
            void *patch_addr = (void*)asm_inst[i-1].address;

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

// Get the Go Version numbers from a complete version string
// Stores the minor and maintenance version numbers in global variables
static void
go_version_numbers(const char *go_runtime_version)
{
    if (!go_runtime_version) return;
    g_go_minor_ver = UNKNOWN_GO_VER;
    g_go_maint_ver = 0; // Default to 0 in case not present

    char buf[256] = {0};
    scope_strncpy(buf, go_runtime_version, sizeof(buf)-1);

    scope_strtok(buf, ".");

    // Get the minor version number
    char *minor = scope_strtok(NULL, ".");
    if (!minor) return;
    scope_errno = 0;
    long minor_val = scope_strtol(minor, NULL, 10);
    if (scope_errno || minor_val <= 0 || minor_val > INT_MAX) return;
    g_go_minor_ver = minor_val;

    // Get the maintenance version number
    char *maint = scope_strtok(NULL, ".");
    if (!maint) return;
    scope_errno = 0;
    long maint_val = scope_strtol(maint, NULL, 10);
    if (scope_errno || maint_val <= 0 || maint_val > INT_MAX) return;
    g_go_maint_ver = maint_val;
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
    char *go_ver;
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
        scopeSetGoAppStateStatic(TRUE);
        patchClone();
        sysprint("This is a static app\n");
    } else {
        scopeSetGoAppStateStatic(FALSE);
        sysprint("This is a dynamic app\n");
    }

    //check ELF type
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)ebuf->buf;
    // if it's a position independent executable, get the base address from /proc/self/maps
    uint64_t base = 0LL;
    if (ehdr->e_type == ET_DYN && (scopeGetGoAppStateStatic() == FALSE)) {
        if (getBaseAddress(&base) != 0) {
            sysprint("ERROR: can't get the base address\n");
            return; // don't install our hooks
        }
        Elf64_Shdr* textSec = getElfSection(ebuf->buf, ".text");
        sysprint("base %lx %lx %lx\n", base, (uint64_t)ebuf->text_addr, textSec->sh_offset);
        base = base - (uint64_t)ebuf->text_addr + textSec->sh_offset;
    }

    void *go_ver_sym = getSymbol(ebuf->buf, "runtime.buildVersion");
    if (!go_ver_sym) {
        // runtime.buildVersion symbol not found, probably dealing with a stripped binary
        // try to retrieve the version symbol address from the .go.buildinfo section
        // if g_go_build_ver is set we know we're dealing with a char *
        // if it is not set, we know we're dealing with a "go string"
        void *ver_addr = getGoVersionAddr(ebuf->buf);
        if (g_go_build_ver[0] != '\0') {
            go_ver = (char *)((uint64_t)ver_addr);
        } else {
            go_ver = go_str((void *)((uint64_t)ver_addr + base));
        }
    } else {
        go_ver = go_str((void *)((uint64_t)go_ver_sym + base));
    }
    
    if (go_ver && (go_runtime_version = go_ver)) {
        sysprint("go_runtime_version = %s\n", go_runtime_version);
        go_version_numbers(go_runtime_version);
        //scope_printf("minor ver: %d\n", g_go_minor_ver);
        //scope_printf("maint ver: %d\n", g_go_maint_ver);
    }
    if (g_go_minor_ver < MIN_SUPPORTED_GO_VER) {
        if (!is_go(ebuf->buf)) {
            // Don't expect to get here, but try to be clear if we do.
            scopeLogWarn("%s is not a go application.  Continuing without AppScope.", ebuf->cmd);
        } else if (go_runtime_version) {
            scopeLogWarn("%s was compiled with go version `%s`.  AppScope can only instrument go1.%d or newer.  Continuing without AppScope.", ebuf->cmd, go_runtime_version, MIN_SUPPORTED_GO_VER);
        } else {
            scopeLogWarn("%s was either compiled with a version of go older than go1.4, or symbols have been stripped.  AppScope can only instrument go1.%d or newer, and requires symbols if compiled with a version of go older than go1.13.  Continuing without AppScope.", ebuf->cmd, MIN_SUPPORTED_GO_VER);
        }
        return; // don't install our hooks
    } else if (g_go_minor_ver > MAX_SUPPORTED_GO_VER) {
        scopeLogWarn("%s was compiled with go version `%s`. Versions newer than Go 1.%d are not yet supported. Continuing without AppScope.", ebuf->cmd, go_runtime_version, MAX_SUPPORTED_GO_VER);
        return; // don't install our hooks
    } 

    uint64_t *ReadFrame_addr;
    if (((ReadFrame_addr = getSymbol(ebuf->buf, "net/http.(*http2Framer).ReadFrame")) == 0) &&
        ((ReadFrame_addr = getGoSymbol(ebuf->buf, "net/http.(*http2Framer).ReadFrame", NULL, NULL)) == 0)) {
        sysprint("WARN: can't get the address for net/http.(*http2Framer).ReadFrame\n");
    }
    ReadFrame_addr = (uint64_t *)((uint64_t)ReadFrame_addr + base);
    scope_sprintf(g_ReadFrame_addr, "%p\n", ReadFrame_addr);

    char gosave[30] = "gosave";
    if (g_go_minor_ver >= 17) scope_strcpy(gosave, "gosave_systemstack_switch");
    if (((go_systemstack_switch = (uint64_t)getSymbol(ebuf->buf, gosave)) == 0) &&
        ((go_systemstack_switch = (uint64_t)getGoSymbol(ebuf->buf, gosave, NULL, NULL)) == 0)) {
        sysprint("WARN: can't get the address for %s\n", gosave);
    }
    go_systemstack_switch = (uint64_t)((char *)go_systemstack_switch + base);
    sysprint("address for gosave_systemstack_switch: 0x%lx\n", go_systemstack_switch);

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

    if (g_go_minor_ver >= 17) {
        // The Go 17 schema works for 1.17-1.19 and possibly future versions
        g_go_schema = &go_17_schema;
    }
    // Update the schema to suit the current version
    adjustGoStructOffsetsForVersion();
    // For validation tests:
    createGoStructFile();

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
            ((orig_func = getGoSymbol(ebuf->buf, tap->func_name, NULL, NULL)) == NULL)) {
            sysprint("WARN: can't get the address for %s\n", tap->func_name);
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

        patch_addrs(funchook, asm_inst, asm_count, tap);
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

    scopeLogWarn("WARN: no frame size");
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
 * NOTE: Avoid using values from a Caller stack where possible since the Caller
 * function could differ.
 * In future iterations, it might be a more logical approach to always grab
 * input params at the beginning of the function, and return values at the return.
 */
inline static void *
do_cfunc(char *stackptr, void *cfunc, void *gfunc)
{
    uint64_t rc;
    char *sys_stack = stackptr;  
    char *g_stack = *(uint64_t *)(sys_stack + 0x50); 

    /*
     * In <= Go 1.16 we must rely on the caller stack for tls_ and http2_ functions
     * because the values we need only exist there, at the point where we patch.
     * We can get to the caller stack by adding the frame size which will be >0
     * for tls_and http2_ function hooks where we hook on the return -1 instruction.
     * This code will therefore put us in the caller context for <= 1.16.
     *
     * In Go 1.17 and higher, the values we need are present in the callee stack
     * (because register values are copied onto the callee stack).
     */
    if (g_go_minor_ver <= 16) {
        uint32_t frame_offset = frame_size(gfunc);
        g_stack += frame_offset;
    }

    // Call the C handler
    __asm__ volatile (
        "mov %1, %%rdi  \n"
        "mov %2, %%rsi  \n"
        "callq *%3  \n"
        : "=r"(rc)                                    // output
        : "r"(sys_stack), "r"(g_stack), "r"(cfunc)    // inputs
        :                                             // clobbered register
        );

    return return_addr(gfunc);
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
  conn                         (net.conn)
  conn.conn_if = conn + 0x8    (net.conn.interface)
  tcpConn = conn.conn_if + 0x8 (net.conn.data)
  netFD = tcpConn + 0x08       (netFD)
  pfd = netFD + 0x0            (poll.FD)
  fd = pfd + 0x10              (pfd.sysfd)
  */
// Retrieve a file descriptor from a Go conn interface
static int
getFDFromConn(uint64_t tcpConn) {
    if (!tcpConn) return -1;

    uint64_t netFD = *(uint64_t *)(tcpConn + g_go_schema->struct_offsets.iface_data);
    if (!netFD) return -1;

    if (g_go_schema->struct_offsets.netfd_to_sysfd == UNDEF_OFFSET) { 
        uint64_t pfd = *(uint64_t *)(netFD + g_go_schema->struct_offsets.netfd_to_pd); 
        if (!pfd) return -1;

        return *(int *)(pfd + g_go_schema->struct_offsets.pd_to_fd); 
    } else {
        return *(int *)(netFD + g_go_schema->struct_offsets.netfd_to_sysfd); 
    }
    return -1;
}

// Extract data from syscalls. Values are available in registers saved on sys_stack.
static void
c_syscall(char *sys_stack, char *g_stack)
{
    uint64_t syscall_num = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_num);
    int64_t rc           = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_rc);
    if(rc < 0) rc = -1; // kernel syscalls can return values < -1

    switch(syscall_num) {
    case 1: // write
        {
            uint64_t fd = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            char *buf   = (char *)*(uint64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);
            uint64_t initialTime = getTime();

            funcprint("Scope: write fd %ld rc %ld buf %s\n", fd, rc, buf);
            doWrite(fd, initialTime, (rc != -1), buf, rc, "go_write", BUF, 0);
        }
        break;
    case 257: // openat
        {
            char *path = (char *)*(uint64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);
            if (!path) {
                scopeLogError("ERROR:go_open: null pathname");
                scope_puts("Scope:ERROR:open:no path");
                scope_fflush(scope_stdout);
                return;
            }

            funcprint("Scope: open of %ld\n", rc);
            doOpen(rc, path, FD, "open");
        }
        break;
    case 263: // unlinkat
        {
            if (rc) return;

            uint64_t dirfd = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            char *pathname = (char *)*(uint64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);
            uint64_t flags = *(int64_t *)(sys_stack + 0x20);

            funcprint("Scope: unlinkat dirfd %ld pathname %s flags %ld\n", dirfd, pathname, flags);
            doDelete(pathname, "go_unlinkat");
        }
        break;
    case 217: // getdents64
        {
            uint64_t dirfd = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            uint64_t initialTime = getTime();

            funcprint("Scope: getdents dirfd %ld rc %ld\n", dirfd, rc);
            doRead(dirfd, initialTime, (rc != -1), NULL, rc, "go_getdents", BUF, 0);
        }
        break;
    case 41: // socket
        {
            if (rc == -1) return;

            uint64_t domain = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            uint64_t type   = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);

            funcprint("Scope: socket domain: %ld type: 0x%lx sd: %ld\n", domain, type, rc);
            addSock(rc, type, domain); // Creates a net object
        }
        break;
    case 288: // accept4
        {
            if (rc == -1) return;

            uint64_t fd           = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            struct sockaddr *addr = *(struct sockaddr **)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);
            socklen_t *addrlen    = *(socklen_t **)(sys_stack + g_go_schema->arg_offsets.c_syscall_p3);

            funcprint("Scope: accept4 of %ld\n", rc);
            doAccept(fd, rc, addr, addrlen, "go_accept4");
        }
        break;
    case 0: // read
        {
            uint64_t fd = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);
            char *buf   = (char *)*(uint64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p2);
            uint64_t initialTime = getTime();

            funcprint("Scope: read of %ld rc %ld\n", fd, rc);
            doRead(fd, initialTime, (rc >= 0), buf, rc, "go_read", BUF, 0);
        } 
        break;
    case 3: // close
        {
            uint64_t fd = *(int64_t *)(sys_stack + g_go_schema->arg_offsets.c_syscall_p1);

            funcprint("Scope: close of %ld\n", fd);
            doCloseAndReportFailures(fd, (rc != -1), "go_close"); // If net, deletes a net object
        }
        break;
    default:
        break;
    }
}

EXPORTON void *
go_syscall(char *stackptr)
{
    return do_cfunc(stackptr, c_syscall, g_go_schema->tap[INDEX_HOOK_SYSCALL].assembly_fn);
}

EXPORTON void *
go_rawsyscall(char *stackptr)
{
    return do_cfunc(stackptr, c_syscall, g_go_schema->tap[INDEX_HOOK_RAWSYSCALL].assembly_fn);
}

EXPORTON void *
go_syscall6(char *stackptr)
{
    return do_cfunc(stackptr, c_syscall, g_go_schema->tap[INDEX_HOOK_SYSCALL6].assembly_fn);
}

// Extract data from net/http.(*connReader).Read (tls server read)
static void
c_tls_server_read(char *sys_stack, char *g_stack)
{
    uint64_t connReader = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_read_connReader); 
    if (!connReader) return;   // protect from dereferencing null
    char *buf           = (char *)*(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_read_buf);
    uint64_t rc         = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_read_rc);
    uint64_t conn       =  *(uint64_t *)(connReader + g_go_schema->struct_offsets.connReader_to_conn);
    if (!conn) return;         // protect from dereferencing null

    /*
     * The rwc net.Conn value can be wrapped as either a *net.TCPConn or
     * *tls.Conn. We are using tlsState *tls.ConnectionState as the indicator
     * of type. If the tlsState field is not 0, the rwc field is of type
     * *tls.Conn. Example; net/http/server.go type conn struct.
     */
    uint64_t cr_conn_rwc_if = conn + g_go_schema->struct_offsets.conn_to_rwc;
    if (!cr_conn_rwc_if) return;
    uint64_t tls   = *(uint64_t *)(conn + g_go_schema->struct_offsets.conn_to_tlsState);
    if (!tls) return;
   
    uint64_t tcpConn = *(uint64_t *)(cr_conn_rwc_if + g_go_schema->struct_offsets.iface_data);
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    funcprint("Scope: go_http_server_read of %d\n", fd);
    doProtocol((uint64_t)0, fd, buf, rc, TLSRX, BUF);
}

EXPORTON void *
go_tls_server_read(char *stackptr)
{
    return do_cfunc(stackptr, c_tls_server_read, g_go_schema->tap[INDEX_HOOK_TLS_SERVER_READ].assembly_fn);
}

// Extract data from net/http.checkConnErrorWriter.Write (tls server write)
static void
c_tls_server_write(char *sys_stack, char *g_stack)
{
    uint64_t conn = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_write_conn);
    if (!conn) return;         // protect from dereferencing null
    char *buf     = (char *)*(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_write_buf);
    uint64_t rc   = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_server_write_rc);

    uint64_t w_conn_rwc_if = (conn + g_go_schema->struct_offsets.conn_to_rwc);
    if (!w_conn_rwc_if) return;
    uint64_t tls =  *(uint64_t *)(conn + g_go_schema->struct_offsets.conn_to_tlsState);
    if (!tls) return;

    uint64_t tcpConn = *(uint64_t *)(w_conn_rwc_if + g_go_schema->struct_offsets.iface_data);
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    funcprint("Scope: c_tls_server_write of %d\n", fd);
    doProtocol((uint64_t)0, fd, buf, rc, TLSTX, BUF);
}

EXPORTON void *
go_tls_server_write(char *stackptr)
{
    return do_cfunc(stackptr, c_tls_server_write, g_go_schema->tap[INDEX_HOOK_TLS_SERVER_WRITE].assembly_fn);
}

// Extract data from net/http.(*persistConn).readResponse (tls client read)
static void
c_tls_client_read(char *sys_stack, char *g_stack)
{
    uint64_t pc = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_client_read_pc); 
    if (!pc) return;

    uint64_t pc_conn_if = (pc + g_go_schema->struct_offsets.persistConn_to_conn);
    if (!pc_conn_if) return;
    uint64_t tls = *(uint64_t*)(pc + g_go_schema->struct_offsets.persistConn_to_tlsState);
    if (!tls) return;

    uint64_t tcpConn = *(uint64_t *)(pc_conn_if + g_go_schema->struct_offsets.iface_data);
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    uint64_t pc_br, len = 0;
    char *buf = NULL;
    if ((pc_br = *(uint64_t *)(pc + g_go_schema->struct_offsets.persistConn_to_bufrd)) != 0) {
        buf = (char *)*(uint64_t *)(pc_br + g_go_schema->struct_offsets.bufrd_to_buf);
        // len is part of the []byte struct; the func doesn't return a len
        len = *(uint64_t *)(pc_br + 0x08);
        if (buf && (len > 0)) {
            doProtocol((uint64_t)0, fd, buf, len, TLSRX, BUF);
            funcprint("Scope: c_tls_client_read of %d\n", fd);
        }
    }
}

EXPORTON void *
go_tls_client_read(char *stackptr)
{
    return do_cfunc(stackptr, c_tls_client_read, g_go_schema->tap[INDEX_HOOK_TLS_CLIENT_READ].assembly_fn);
}

// Extract data from net/http.persistConnWriter.Write (tls client write)
static void
c_tls_client_write(char *sys_stack, char *g_stack)
{
    uint64_t w_pc = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_client_write_w_pc);
    char *buf     = (char *)*(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_client_write_buf);
    uint64_t rc   = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_tls_client_write_rc);
    if (rc < 1) return;

    uint64_t pc_conn_if = (w_pc + g_go_schema->struct_offsets.persistConn_to_conn); 
    if (!pc_conn_if) return;
    uint64_t tls =  *(uint64_t*)(w_pc + g_go_schema->struct_offsets.persistConn_to_tlsState); 
    if (!tls) return;

    uint64_t tcpConn = *(uint64_t *)(pc_conn_if + g_go_schema->struct_offsets.iface_data); 
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);
    
    doProtocol((uint64_t)0, fd, buf, rc, TLSTX, BUF);
}

EXPORTON void *
go_tls_client_write(char *stackptr)
{
    return do_cfunc(stackptr, c_tls_client_write, g_go_schema->tap[INDEX_HOOK_TLS_CLIENT_WRITE].assembly_fn);
}

// Extract data from net/http.(*http2serverConn).readFrames (tls http2 server read)
static void
c_http2_server_read(char *sys_stack, char *g_stack)
{
    uint64_t sc = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_server_read_sc);
    if (!sc) return;
    uint64_t fr   = *(uint64_t *)(sc + g_go_schema->struct_offsets.sc_to_fr);
    if (!fr) return;
    char *buf     = (char *)(fr + g_go_schema->struct_offsets.fr_to_headerBuf);
    if (!buf) return;
    char *readBuf = (char *)*(uint64_t *)(fr + g_go_schema->struct_offsets.fr_to_readBuf);
    if (!readBuf) return;
    uint64_t conn_if = (sc + g_go_schema->struct_offsets.sc_to_conn); 
    if (!conn_if) return;
    uint64_t tcpConn = *(uint64_t *)(conn_if + g_go_schema->struct_offsets.iface_data); 
    if (!tcpConn) return;

    uint8_t *newbuf = (uint8_t *)buf; 
    uint32_t rc = 0;
    rc += newbuf[0] << 16;
    rc += newbuf[1] << 8;
    rc += newbuf[2];
    if (rc < 1) return;
    rc += HTTP2_FRAME_HEADER_LEN;

    int fd = getFDFromConn(tcpConn);
    funcprint("Scope: c_http2_server_read: buf 0x%x readBuf 0x%x\n", *buf, *readBuf);

    char *frame = (char *)scope_malloc(rc);
    if (!frame) return;
    scope_memmove(frame, buf, HTTP2_FRAME_HEADER_LEN);
    scope_memmove(frame + HTTP2_FRAME_HEADER_LEN, readBuf, rc - HTTP2_FRAME_HEADER_LEN);

    doProtocol((uint64_t)0, fd, frame, rc, TLSRX, BUF);
    funcprint("Scope: c_http2_server_read of %d\n", fd);

    scope_free(frame);
}

EXPORTON void *
go_http2_server_read(char *stackptr)
{
    return do_cfunc(stackptr, c_http2_server_read, g_go_schema->tap[INDEX_HOOK_HTTP2_SERVER_READ].assembly_fn);
}

// Extract data from net/http.(*http2serverConn).Flush (tls http2 server write)
static void
c_http2_server_write(char *sys_stack, char *g_stack)
{
    uint64_t sc      = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_server_write_sc);
    if (!sc) return;
    uint64_t fr      = *(uint64_t *)(sc + g_go_schema->struct_offsets.sc_to_fr);
    if (!fr) return;  
    char *writeBuf   = (char *)*(uint64_t *)(fr + g_go_schema->struct_offsets.fr_to_writeBuf);
    if (!writeBuf) return;
    uint64_t conn_if = (sc + g_go_schema->struct_offsets.sc_to_conn); 
    if (!conn_if) return;
    uint64_t tcpConn = *(uint64_t *)(conn_if + g_go_schema->struct_offsets.iface_data); 
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    /*
     * If Proto is not yet detected, wait for the preface string to be received.
     * We expect that the only write that occurs in this case is a settings
     * frame. Refer to src/net/http/h2_bundle.go:(sc *httpServerConn) serve().
     * This is related to the nature of the parallel/async behavior of sc.
     */
    if (isProtocolSet(fd) == FALSE) return;

    uint8_t *newbuf = (uint8_t *)writeBuf; 
    uint32_t rc = 0;
    rc += newbuf[0] << 16;
    rc += newbuf[1] << 8;
    rc += newbuf[2];
    rc += HTTP2_FRAME_HEADER_LEN;

    // At times, the buffer contains 0. We don't want to call doProtocol when
    // the header is empty (length and type bytes are all 0).
    // Especially since it will set PROTO to false if it's the first call to doProtocol
    // on this socket.
    if ((rc < 1) && (!newbuf[3])) return;

    funcprint("Scope: c_http2_server_write of %d %i\n", fd, rc);
    doProtocol((uint64_t)0, fd, writeBuf, rc, TLSTX, BUF);
}

EXPORTON void *
go_http2_server_write(char *stackptr)
{
    return do_cfunc(stackptr, c_http2_server_write, g_go_schema->tap[INDEX_HOOK_HTTP2_SERVER_WRITE].assembly_fn);
}

// Extract data from net/http.(*http2serverConn).readPreface (tls http2 server write)
static void
c_http2_server_preface(char *sys_stack, char *g_stack)
{
    // In this function in <= go 16 we have to go to the callee to get input params and return values
    g_stack -= g_go_schema->arg_offsets.c_http2_server_preface_callee;

    uint64_t *rc     = (uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_server_preface_rc);
    if ((rc == NULL) || (rc == (uint64_t *)0xffffffff)) return;
    if (*rc != 0) return;
    uint64_t sc      = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_server_preface_sc);
    if (!sc) return;
    uint64_t fr      = *(uint64_t *)(sc + g_go_schema->struct_offsets.sc_to_fr);
    if (!fr) return;
    uint64_t conn_if = (sc + g_go_schema->struct_offsets.sc_to_conn); 
    if (!conn_if) return;
    uint64_t tcpConn = *(uint64_t *)(conn_if + g_go_schema->struct_offsets.iface_data); 
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    funcprint("Scope: c_http2_server_preface of %d %ld\n", fd, *rc);
    doProtocol((uint64_t)0, fd, PRI_STR, PRI_STR_LEN - 1, TLSRX, BUF);
}

EXPORTON void *
go_http2_server_preface(char *stackptr)
{
    return do_cfunc(stackptr, c_http2_server_preface, g_go_schema->tap[INDEX_HOOK_HTTP2_SERVER_PREFACE].assembly_fn);
}

/*
  Offsets here may be outdated/incorrect for certain versions. Leaving for reference:
  Our variable    Memory loc        Go variable
  cc              stackaddr + 68    cc
  fr              sc + 0x130        cc.fr
  buf             fr + 0x40         cc.fr.headerBuf 
  readBuf         fr + 0x58         cc.fr.readBuf
  tcpConn         sc + 0x10         cc.conn
 */
// Extract data from net/http.(*http2clientConnReadLoop).run (tls http2 client read)
static void
c_http2_client_read(char *sys_stack, char *g_stack)
{
    uint64_t cc         = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_client_read_cc);
    if (!cc) return;
    uint64_t fr         = *(uint64_t *)(cc + g_go_schema->struct_offsets.cc_to_fr);
    if (!fr) return;
    char *buf           = (char *)(fr + g_go_schema->struct_offsets.fr_to_headerBuf);
    if (!buf) return;
    char *readBuf       = (char *)*(uint64_t *)(fr + g_go_schema->struct_offsets.fr_to_readBuf);
    if (!readBuf) return;
    uint64_t conn_if = (cc + g_go_schema->struct_offsets.cc_to_tconn); 
    if (!conn_if) return;
    uint64_t tcpConn    = *(uint64_t *)(conn_if + g_go_schema->struct_offsets.iface_data); 
    if (!tcpConn) return;

    int fd = getFDFromConn(tcpConn);

    uint8_t *newbuf = (uint8_t *)buf; 
    uint32_t rc = 0;
    rc += newbuf[0] << 16;
    rc += newbuf[1] << 8;
    rc += newbuf[2];
    if (rc < 1) return;
    rc += HTTP2_FRAME_HEADER_LEN;

    char *frame = (char *)scope_malloc(rc);
    if (!frame) return;
    scope_memmove(frame, buf, HTTP2_FRAME_HEADER_LEN);
    scope_memmove(frame + HTTP2_FRAME_HEADER_LEN, readBuf, rc - HTTP2_FRAME_HEADER_LEN);

    doProtocol((uint64_t)0, fd, frame, rc, TLSRX, BUF);
    funcprint("Scope: c_http2_client_read of %d\n", fd);

    scope_free(frame);
}

EXPORTON void *
go_http2_client_read(char *stackptr)
{
    return do_cfunc(stackptr, c_http2_client_read, g_go_schema->tap[INDEX_HOOK_HTTP2_CLIENT_READ].assembly_fn);
}

// Extract data from net/http.http2stickyErrWriter.Write (tls http2 client write)
static void
c_http2_client_write(char *sys_stack, char *g_stack)
{
    uint64_t tcpConn = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_client_write_tcpConn);
    if (!tcpConn) return;
    char *buf        = (char *)*(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_client_write_buf);
    if (!buf) return;
    uint64_t rc      = *(uint64_t *)(g_stack + g_go_schema->arg_offsets.c_http2_client_write_rc);
    if (rc < 1) return;

    int fd = getFDFromConn(tcpConn);
    
    doProtocol((uint64_t)0, fd, buf, rc, TLSTX, BUF);
    funcprint("Scope: c_http2_client_write of %d\n", fd);
}

EXPORTON void *
go_http2_client_write(char *stackptr)
{
    return do_cfunc(stackptr, c_http2_client_write, g_go_schema->tap[INDEX_HOOK_HTTP2_CLIENT_WRITE].assembly_fn);
}

extern void handleExit(void);
static void
c_exit(char *sys_stack)
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
    funcprint("c_exit\n");

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
    return do_cfunc(stackptr, c_exit, g_go_schema->tap[INDEX_HOOK_EXIT].assembly_fn);
}

EXPORTON void *
go_die(char *stackptr)
{
    return do_cfunc(stackptr, c_exit, g_go_schema->tap[INDEX_HOOK_DIE].assembly_fn);
}
