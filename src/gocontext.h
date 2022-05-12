#ifndef __GOTCONTEXT_H__
#define __GOTCONTEXT_H__
#include "scopeelf.h"

#define EXPORTON __attribute__((visibility("default")))

typedef struct {
    int c_write_fd;
    int c_write_buf;
    int c_write_rc;
    int c_getdents_dirfd;
    int c_getdents_rc;
    int c_unlinkat_dirfd;
    int c_unlinkat_pathname;
    int c_unlinkat_flags;
    int c_unlinkat_rc;
    int c_open_fd;
    int c_open_path;
    int c_close_fd;
    int c_close_rc;
    int c_read_fd;
    int c_read_buf;
    int c_read_rc;
    int c_socket_domain;
    int c_socket_type;
    int c_socket_sd;
    int c_accept4_fd;
    int c_accept4_addr;
    int c_accept4_addrlen;
    int c_accept4_sd_out;
    int c_tls_server_read_callee;
    int c_tls_server_read_connReader;
    int c_tls_server_read_buf;
    int c_tls_server_read_rc;
    int c_tls_server_write_callee;
    int c_tls_server_write_conn;
    int c_tls_server_write_buf;
    int c_tls_server_write_rc;
    int c_tls_client_write_callee;
    int c_tls_client_write_w_pc;
    int c_tls_client_write_buf;
    int c_tls_client_write_rc;
    int c_tls_client_read_callee;
    int c_tls_client_read_pc;
    int c_http2_client_write_callee;
    int c_http2_client_write_tcpConn;
    int c_http2_client_write_buf;
    int c_http2_client_write_rc;
} go_arg_offsets_t;

typedef struct {                  // Structure               Field       Offset
    int g_to_m;                   // "runtime.g"             "m"        "48"
    int m_to_tls;                 // "runtime.m"             "tls"      "136"
    int connReader_to_conn;       // "net/http.connReader"   "conn"     "0"
    int conn_to_tlsState;         // "net/http.conn"         "tlsState" "48"
    int persistConn_to_conn;      // "net/http.persistConn"  "conn"     "88"
    int persistConn_to_bufrd;     // "net/http.persistConn"  "br"       "104"
    int iface_data;               // "runtime.iface"         "data"     "8"
    int netfd_to_pd;              // "net.netFD"             "pfd"      "0"
    int pd_to_fd;                 // "internal/poll.FD"      "sysfd"    "16"
    int netfd_to_sysfd;           // "net.netFD"             "sysfd"    "16"
    int bufrd_to_buf;             // "bufio/Reader"          "buf"      "0"
    int conn_to_rwc;              // "net/http.conn"         "rwc"      "0"
    int persistConn_to_tlsState;  // "net/http.persistConn"  "tlsState" "96"
} go_struct_offsets_t;

typedef struct {
    // These are constants at build time
    char *   func_name;    // name of go function
    void *   assembly_fn;  // scope handler function (in assembly)

    // These are set at runtime.
    void *   return_addr;  // addr of where in go to resume after patch
    uint32_t frame_size;   // size of go stack frame
} tap_t;

typedef struct {
    go_arg_offsets_t arg_offsets;
    go_struct_offsets_t struct_offsets;
    tap_t tap[];
} go_schema_t;

// Go strings are not null delimited like c strings.
// Instead, go strings have structure which includes a length field.
typedef struct {
    char* str;  // 0x0 offset
    int   len;  // 0x8 offset
} gostring_t;

typedef void (*assembly_fn)(void);

extern go_schema_t *g_go_schema;
extern go_schema_t go_12_schema;
extern go_schema_t go_17_schema;
extern go_arg_offsets_t g_go_arg;
extern go_struct_offsets_t g_go_struct;
extern tap_t g_go_tap[];
extern unsigned long scope_fs;
extern uint64_t scope_stack;

extern int arch_prctl(int, unsigned long);
extern void initGoHook(elf_buf_t*);
extern void sysprint(const char *, ...) PRINTF_FORMAT(1, 2);
extern void *getSymbol(const char *, char *);

extern void go_hook_write(void);
extern void go_hook_open(void);
extern void go_hook_unlinkat(void);
extern void go_hook_getdents(void);
extern void go_hook_socket(void);
extern void go_hook_accept4(void);
extern void go_hook_read(void);
extern void go_hook_close(void);
extern void go_hook_tls_server_read(void);
extern void go_hook_tls_server_write(void);
extern void go_hook_tls_client_read(void);
extern void go_hook_tls_client_write(void);
extern void go_hook_http2_client_write(void);
extern void go_hook_exit(void);
extern void go_hook_die(void);

extern void go_hook_reg_write(void);
extern void go_hook_reg_open(void);
extern void go_hook_reg_unlinkat(void);
extern void go_hook_reg_getdents(void);
extern void go_hook_reg_socket(void);
extern void go_hook_reg_accept4(void);
extern void go_hook_reg_read(void);
extern void go_hook_reg_close(void);
extern void go_hook_reg_tls_server_read(void);
extern void go_hook_reg_tls_server_write(void);
extern void go_hook_reg_tls_client_read(void);
extern void go_hook_reg_tls_client_write(void);
extern void go_hook_reg_http2_client_write(void);

#endif // __GOTCONTEXT_H__
