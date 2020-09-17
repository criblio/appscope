#ifndef __GOTCONTEXT_H__
#define __GOTCONTEXT_H__
#include "scopeelf.h"

#define EXPORTON __attribute__((visibility("default")))

typedef struct {            // Structure                     Field              Offset
    int g_to_m;             // "runtime.g"                   "m"                "48"
    int m_to_tls;           // "runtime.m"                   "tls"              "136"
    int connReader_to_conn; // "net/http.connReader"         "conn"             "0"
    int conn_to_tlsState;   // "net/http.conn"               "tlsState"         "48"
    int conn_to_remoteAddr; // "net/http.conn"               "remoteAddr"       "32"
} go_offsets_t;

typedef struct {
    // These are constants at build time
    char *   func_name;    // name of go function
    void *   assembly_fn;  // scope handler function (in assembly)

    // These are set at runtime.
    void *   return_addr;  // addr of where in go to resume after patch
    uint32_t frame_size;   // size of go stack frame
} tap_t;

// Go strings are not null delimited like c strings.
// Instead, go strings have structure which includes a length field.
typedef struct {
    char* str;  // 0x0 offset
    int   len;  // 0x8 offset
} gostring_t;

typedef void (*assembly_fn)(void);

extern go_offsets_t g_go;
extern tap_t g_go_tap[];
extern unsigned long scope_fs;
extern uint64_t scope_stack;

extern int arch_prctl(int, unsigned long);
extern void initGoHook(elf_buf_t*);
extern void sysprint(const char *, ...);
extern void *getSymbol(const char *, char *);

extern void go_hook_write(void);
extern void go_hook_open(void);
extern void go_hook_socket(void);
extern void go_hook_accept4(void);
extern void go_hook_read(void);
extern void go_hook_close(void);
extern void go_hook_tls_read(void);
extern void go_hook_tls_write(void);
extern void go_hook_exit(void);
extern void go_hook_die(void);

#endif // __GOTCONTEXT_H__
