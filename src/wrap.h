
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
#include <limits.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

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
#define FS_ENTRIES 1024
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
    CONNECTION_DURATION,
    PROC_CPU,
    PROC_MEM,
    PROC_THREAD,
    PROC_FD,
    PROC_CHILD,
    NETRX,
    NETTX,
    NETRX_PROC,
    NETTX_PROC,
    DNS,
    FS_DURATION,
    FS_SIZE_READ,
    FS_SIZE_WRITE,
    FS_OPEN,
    FS_CLOSE,
    FS_SEEK,
    FS_READ,
    FS_WRITE,
    FS_STAT,
};

// File types; stream or fd
enum fs_type_t {
    FD,
    STREAM
};

typedef struct metric_counters_t {
    int openPorts;
    int TCPConnections;
    int activeConnections;
    int netrx;
    int nettx;
} metric_counters;

typedef struct rtconfig_t {
    int numNinfo;
    int numFSInfo;
    bool tsc_invariant;
    bool tsc_rdtscp;
    uint64_t freq;
    char hostname[MAX_HOSTNAME];
    char procname[MAX_PROCNAME];
} rtconfig;

typedef struct thread_timing_t {
    unsigned interval;                   // in seconds
    time_t startTime; 
    bool once;
    pthread_t periodicTID;
} thread_timing;

typedef struct net_info_t {
    int fd;
    int type;
    int addrType;
    bool network;
    bool listen;
    bool accept;
    uint64_t startTime;
    uint64_t duration;
    char dnsName[MAX_HOSTNAME];
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
} net_info;

typedef struct fs_info_t {
    int fd;
    enum fs_type_t type;
    uint64_t startTime;
    uint64_t duration;
    char path[PATH_MAX];
} fs_info;

typedef struct interposed_funcs_t {
    void (*vsyslog)(int, const char *, va_list);
    pid_t (*fork)(void);
    int (*open)(const char *, int, ...);
    int (*open64)(const char *, int, ...);
    int (*openat)(int, const char *, int, ...);
    int (*openat64)(int, const char *, int, ...);
    FILE *(*fopen)(const char *, const char *);
    FILE *(*fopen64)(const char *, const char *);
    FILE *(*freopen)(const char *, const char *, FILE *);
    FILE *(*freopen64)(const char *, const char *, FILE *);
    int (*creat)(const char *, mode_t);
    int (*creat64)(const char *, mode_t);
    int (*close)(int);
    int (*fclose)(FILE *);
    int (*fcloseall)(void);
    ssize_t (*read)(int, void *, size_t);
    ssize_t (*readv)(int, const struct iovec *, int);
    size_t (*fread)(void *, size_t, size_t, FILE *);
    ssize_t (*pread)(int, void *, size_t, off_t);
    ssize_t (*pread64)(int, void *, size_t, off_t);
    ssize_t (*preadv)(int, const struct iovec *, int, off_t);
    ssize_t (*preadv2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*preadv64v2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*write)(int, const void *, size_t);
    ssize_t (*writev)(int, const struct iovec *, int);
    size_t (*fwrite)(const void *, size_t, size_t, FILE *);
    ssize_t (*pwrite)(int, const void *, size_t, off_t);
    ssize_t (*pwrite64)(int, const void *, size_t, off_t);
    ssize_t (*pwritev)(int, const struct iovec *, int, off_t);
    ssize_t (*pwritev2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*pwritev64v2)(int, const struct iovec *, int, off_t, int);
    off_t (*lseek)(int, off_t, int);
    off_t (*lseek64)(int, off_t, int);
    int (*fseeko)(FILE *, off_t, int);
    int (*fseeko64)(FILE *, off_t, int);
    long (*ftell)(FILE *);
    off_t (*ftello)(FILE *);
    off_t (*ftello64)(FILE *);
    int (*fgetpos)(FILE *, fpos_t *);
    int (*fsetpos)(FILE *, const fpos_t *);
    void (*rewind)(FILE *);
    int (*shutdown)(int, int);
    int (*socket)(int, int, int);
    int (*listen)(int, int);
    int (*bind)(int, const struct sockaddr *, socklen_t);
    int (*connect)(int, const struct sockaddr *, socklen_t);
    int (*accept)(int, struct sockaddr *, socklen_t *);
    int (*accept4)(int, struct sockaddr *, socklen_t *, int);
    int (*accept$NOCANCEL)(int, struct sockaddr *, socklen_t *);
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

/*
 * As described by Intel, this is not a traditional test-and-set operation,
 * but rather an atomic exchange operation.
 * It writes val into *ptr, and returns the previous contents of *ptr.
 */
static inline int
atomicSet(int *ptr, int val)
{
    return __sync_lock_test_and_set(ptr, val);
}

extern rtconfig g_cfg;
static inline uint64_t
getTime() {
    unsigned low, high;

    /*
     * Newer CPUs support a second TSC read instruction.
     * The new instruction, rdtscp, performes a serialization
     * instruction before calling RDTSC. Specifically, rdtscp
     * performs a cpuid instruction then an rdtsc. This is 
     * intended to flush the instruction pipeline befiore
     * calling rdtsc.
     *
     * A serializing instruction is used as the order of 
     * execution is not guaranteed. It's described as 
     * "Out ofOrder Execution". In some cases the read 
     * of the TSC can come before the instruction being 
     * measured. That scenario is not very likely for us
     * as we tend to measure functions as opposed to 
     * statements.
     *
     * If the rdtscp instruction is available, we use it.
     * It takes a bit longer to execute due to the extra
     * serialization instruction (cpuid). However, it's
     * supposed to be more accurate.
     */
    if (g_cfg.tsc_rdtscp == TRUE) {
        asm volatile("rdtscp" : "=a" (low), "=d" (high));
    } else {
        asm volatile("rdtsc" : "=a" (low), "=d" (high));
    }
    return ((uint64_t)low) | (((uint64_t)high) << 32);
}

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);

#define GET_PORT(fd, type, which) ({            \
     in_port_t port;                     \
     switch (type) {                     \
     case AF_INET:                                                      \
        if (which == LOCAL) {                                           \
            port = ((struct sockaddr_in *)&g_netinfo[fd].localConn)->sin_port; \
        } else {                                                        \
            port = ((struct sockaddr_in *)&g_netinfo[fd].remoteConn)->sin_port; \
        }                                                               \
        break;                                                          \
     case AF_INET6:                                                     \
        if (which == LOCAL) {                                           \
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].localConn)->sin6_port; \
        } else {                                                        \
            port = ((struct sockaddr_in6 *)&g_netinfo[fd].remoteConn)->sin6_port; \
        }                                                               \
        break;                                                          \
     default:                                                           \
         port = (in_port_t)0;                                           \
         break;                                                         \
     }                                                                  \
        htons(port);})

// struct to hold the next 6 numeric (int/ptr etc) variadic arguments
// use LOAD_FUNC_ARGS_VALIST to populate this structure
struct FuncArgs{
    uint64_t arg[6]; // pick the first 6 args
};

#define LOAD_FUNC_ARGS_VALIST(a, lastNamedArg)  \
    do{                                         \
        va_list __args;                         \
        va_start(__args, lastNamedArg);         \
        a.arg[0] = va_arg(__args, uint64_t);    \
        a.arg[1] = va_arg(__args, uint64_t);    \
        a.arg[2] = va_arg(__args, uint64_t);    \
        a.arg[3] = va_arg(__args, uint64_t);    \
        a.arg[4] = va_arg(__args, uint64_t);    \
        a.arg[5] = va_arg(__args, uint64_t);    \
        va_end(__args);                         \
    }while(0)

#ifdef __LINUX__
extern void *_dl_sym(void *, const char *, void *);
#define WRAP_CHECK(func, rc)                                           \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = _dl_sym(RTLD_NEXT, #func, func)) == NULL) {   \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return rc;                                                 \
       }                                                               \
    } 

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = _dl_sym(RTLD_NEXT, #func, func)) == NULL) {   \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return;                                                    \
       }                                                               \
    } 

#else
#define WRAP_CHECK(func, rc)                                           \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return rc;                                                 \
       }                                                               \
    } 

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return;                                                    \
       }                                                               \
    } 
#endif // __LINUX__

#endif // __WRAP_H__
