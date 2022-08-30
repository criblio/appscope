#ifndef __GOTCONTEXT_H__
#define __GOTCONTEXT_H__
#include "scopeelf.h"

#define EXPORTON __attribute__((visibility("default")))

typedef struct {
    int c_syscall_write_fd;
    int c_syscall_write_buf;
    int c_syscall_openat_path;
    int c_syscall_unlinkat_dirfd;
    int c_syscall_unlinkat_pathname;
    int c_syscall_unlinkat_flags;
    int c_syscall_getdents64_dirfd;
    int c_syscall_socket_domain;
    int c_syscall_socket_type;
    int c_syscall_accept4_fd;
    int c_syscall_accept4_addr;
    int c_syscall_accept4_addrlen;
    int c_syscall_read_fd;
    int c_syscall_read_buf;
    int c_syscall_close_fd;
    int c_tls_server_read_connReader;
    int c_tls_server_read_buf;
    int c_tls_server_read_rc;
    int c_tls_server_write_conn;
    int c_tls_server_write_buf;
    int c_tls_server_write_rc;
    int c_tls_client_read_pc;
    int c_tls_client_write_w_pc;
    int c_tls_client_write_buf;
    int c_tls_client_write_rc;
    int c_http2_server_read_sc;
    int c_http2_server_write_sc;
    int c_http2_server_preface_callee;
    int c_http2_server_preface_sc;
    int c_http2_server_preface_rc;
    int c_http2_client_read_cc;
    int c_http2_client_write_tcpConn;
    int c_http2_client_write_buf;
    int c_http2_client_write_rc;
} go_arg_offsets_t;

typedef struct {                  // Structure                  Field      
    int g_to_m;                   // "runtime.g"                "m"        
    int m_to_tls;                 // "runtime.m"                "tls"      
    int connReader_to_conn;       // "net/http.connReader"      "conn"     
    int conn_to_tlsState;         // "net/http.conn"            "tlsState" 
    int persistConn_to_conn;      // "net/http.persistConn"     "conn"     
    int persistConn_to_bufrd;     // "net/http.persistConn"     "br"       
    int iface_data;               // "runtime.iface"            "data"     
    int netfd_to_pd;              // "net.netFD"                "pfd"      
    int pd_to_fd;                 // "internal/poll.FD"         "sysfd"    
    int netfd_to_sysfd;           // "net.netFD"                "sysfd"    
    int bufrd_to_buf;             // "bufio/Reader"             "buf"      
    int conn_to_rwc;              // "net/http.conn"            "rwc"      
    int persistConn_to_tlsState;  // "net/http.persistConn"     "tlsState" 
    int fr_to_readBuf;            // "net/http.http2Framer"     "readBuf" 
    int fr_to_writeBuf;           // "net/http.http2Framer"     "writeBuf" 
    int fr_to_headerBuf;          // "net/http.http2Framer"     "headerBuf" 
    int fr_to_rc;                 // "net/http.http2Framer"     "readBuf"     
    int cc_to_fr;                 // "net/http.http2ClientConn" "http2framer"
    int cc_to_tconn;              // "net/http.http2ClientConn" "tconn"
    int sc_to_fr;                 // "net/http.http2serverConn" "http2framer"
    int sc_to_conn;               // "net/http.http2serverConn" "conn"
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
extern go_schema_t go_8_schema;
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

extern void go_hook_tls_server_read(void);
extern void go_hook_tls_server_write(void);
extern void go_hook_tls_client_read(void);
extern void go_hook_tls_client_write(void);
extern void go_hook_http2_server_read(void);
extern void go_hook_http2_server_write(void);
extern void go_hook_http2_server_preface(void);
extern void go_hook_http2_client_read(void);
extern void go_hook_http2_client_write(void);
extern void go_hook_exit(void);
extern void go_hook_die(void);

extern void go_hook_reg_tls_server_read(void);
extern void go_hook_reg_tls_server_write(void);
extern void go_hook_reg_tls_client_read(void);
extern void go_hook_reg_tls_client_write(void);
extern void go_hook_reg_http2_server_read(void);
extern void go_hook_reg_http2_server_write(void);
extern void go_hook_reg_http2_server_preface(void);
extern void go_hook_reg_http2_client_read(void);
extern void go_hook_reg_http2_client_write(void);

extern void go_reg_syscall(void);
extern void go_reg_rawsyscall(void);
extern void go_reg_syscall6(void);

#endif // __GOTCONTEXT_H__
