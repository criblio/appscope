#ifndef __WRAP_H__
#define __WRAP_H__

#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>
#include <ctype.h>

#include "dns.h"

#ifdef __MACOS__
#include "../os/macOS/os.h"
#elif defined (__LINUX__)
#include "../os/linux/os.h"
#endif

#define DEBUG 0
#define TRUE 1
#define FALSE 0
#define EXPORT __attribute__((visibility("default")))
#define EXPORTOFF  __attribute__((visibility("hidden")))
#define EXPORTON __attribute__((visibility("default")))
#define DNS_PORT 53

// Initial size of net array for state
#define NET_ENTRIES 1024
#define MAX_FDS 4096
#define PROTOCOL_STR 8
#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128
#define SCOPE_UNIX 99

#ifndef bool
typedef unsigned int bool;
#endif

// Several control types, used in several areas
enum control_type_t {
    LOCAL,
    REMOTE,
    PERIODIC,
    EVENT_BASED
};

enum metric_t {
    OPEN_PORTS,
    TCP_CONNECTIONS,
    ACTIVE_CONNECTIONS,
    PROC_CPU,
    PROC_MEM,
    PROC_THREAD,
    PROC_FD,
    PROC_CHILD,
    NETRX,
    NETTX,
    NETRX_PROC,
    NETTX_PROC,
    DNS
};

typedef struct net_info_t {
    int fd;
    int type;
    int addrType;
    bool network;
    bool listen;
    bool accept;
    char dnsName[MAX_HOSTNAME];
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
} net_info;

typedef struct interposed_funcs_t {
    void (*vsyslog)(int, const char *, va_list);
    pid_t (*fork)(void);
    int (*close)(int);
    int (*shutdown)(int, int);
    int (*socket)(int, int, int);
    int (*listen)(int, int);
    int (*bind)(int, const struct sockaddr *, socklen_t);
    int (*connect)(int, const struct sockaddr *, socklen_t);
    int (*accept)(int, struct sockaddr *, socklen_t *);
    int (*accept4)(int, struct sockaddr *, socklen_t *, int);
    int (*accept$NOCANCEL)(int, struct sockaddr *, socklen_t *);
    ssize_t (*read)(int, void *, size_t);
    ssize_t (*write)(int, const void *, size_t);
    ssize_t (*send)(int, const void *, size_t, int);
    int (*fcntl)(int, int, ...);
    int (*fcntl64)(int, int, ...);
    int (*dup)(int);
    int (*dup2)(int, int);
    int (*dup3)(int, int, int);
    ssize_t (*sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
    ssize_t (*sendmsg)(int, const struct msghdr *, int);
    ssize_t (*recv)(int, void *, size_t, int);
    ssize_t (*recvfrom)(int sockfd, void *buf, size_t len, int flags,
                                struct sockaddr *src_addr, socklen_t *addrlen);
    ssize_t (*recvmsg)(int, struct msghdr *, int);
    // macOS
    int (*close$NOCANCEL)(int);
    int (*close_nocancel)(int);
    int (*guarded_close_np)(int, void *);
    ssize_t (*__sendto_nocancel)(int, const void *, size_t, int,
                                 const struct sockaddr *, socklen_t);
    uint32_t (*DNSServiceQueryRecord)(void *, uint32_t, uint32_t, const char *,
                                      uint16_t, uint16_t, void *, void *);
} interposed_funcs;
    
static inline void
atomicAdd(int *ptr, int val) {
    (void)__sync_add_and_fetch(ptr, val);
}

static inline void
atomicSub(int *ptr, int val)
{
    if (ptr && (*ptr == 0)) return;
	(void)__sync_sub_and_fetch(ptr, val);
}

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);

#define GET_PORT(fd, type, which) ({                  \
        in_port_t port; \
        switch (type) { \
    case AF_INET: \
        if (which == LOCAL) {                                           \
            port = ((struct sockaddr_in *)&g_netinfo[fd].localConn)->sin_port; \
        } else {                                                        \
            port = ((struct sockaddr_in *)&g_netinfo[fd].remoteConn)->sin_port; \
        }                                                               \
        break;\
    case AF_INET6:\
        if (which == LOCAL) {                                           \
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].localConn)->sin6_port; \
        } else {                                                        \
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].remoteConn)->sin6_port; \
        } \
        break; \
    default: \
        port = (in_port_t)0; \
        break; \
    } \
        htons(port);})

// struct to hold the next 6 numeric (int/ptr etc) variadic arguments
// use LOAD_FUNC_ARGS_VALIST to populate this structure
struct FuncArgs{
    uint64_t arg[6]; // pick the first 6 args
};

#define LOAD_FUNC_ARGS_VALIST(a, lastNamedArg)  \
    do{                                     \
        va_list __args;                     \
        va_start(__args, lastNamedArg);     \
        a.arg[0] = va_arg(__args, uint64_t); \
        a.arg[1] = va_arg(__args, uint64_t); \
        a.arg[2] = va_arg(__args, uint64_t); \
        a.arg[3] = va_arg(__args, uint64_t); \
        a.arg[4] = va_arg(__args, uint64_t); \
        a.arg[5] = va_arg(__args, uint64_t); \
        va_end(__args);                     \
    }while(0)

#endif // __WRAP_H__
