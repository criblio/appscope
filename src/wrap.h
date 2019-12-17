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
#include <sys/syscall.h>
#include <wchar.h>
#include <sys/poll.h>

#include <sys/stat.h>
#if defined(__LINUX__) && defined(__STATX__) && defined(STRUCT_STATX_MISSING_FROM_SYS_STAT_H)
#include <linux/stat.h>
#endif // __LINUX__ && __STATX__ && STRUCT_STATX_MISSING_FROM_SYS_STAT_H

#include <sys/statvfs.h>
#include <sys/param.h>
#include <sys/mount.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#ifdef __LINUX__
#include <sys/vfs.h>
#include <sys/prctl.h>
#endif 

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
#define PROTOCOL_STR 16
#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128
#define SCOPE_UNIX 99
#define DYN_CONFIG_PREFIX "scope"
#define MAXTRIES 10
#ifndef bool
typedef unsigned int bool;
#endif

/*
 * OK; this is not cool. But, we are holding off making structural changes right now 
 * We'll move things into a Linux only build. Until then we need these for the macOS build
 */
#ifdef __MACOS__
#ifndef off64_t
typedef uint64_t off64_t;
#endif
#ifndef fpos64_t
typedef uint64_t fpos64_t;
#endif
#ifndef statvfs64
struct statvfs64 {
    uint64_t x;
};
#endif
#endif // __MACOS__

#ifndef AF_NETLINK
#define AF_NETLINK 16
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
    NET_CONNECTIONS,
    CONNECTION_DURATION,
    PROC_CPU,
    PROC_MEM,
    PROC_THREAD,
    PROC_FD,
    PROC_CHILD,
    NETRX,
    NETTX,
    DNS,
    DNS_DURATION,
    FS_DURATION,
    FS_READ,
    FS_WRITE,
    FS_OPEN,
    FS_CLOSE,
    FS_SEEK,
    TOT_READ,
    TOT_WRITE,
    TOT_RX,
    TOT_TX,
    TOT_SEEK,
    TOT_STAT,
    TOT_OPEN,
    TOT_CLOSE,
    TOT_DNS,
    TOT_PORTS,
    TOT_TCP_CONN,
    TOT_UDP_CONN,
    TOT_OTHER_CONN,
    TOT_FS_DURATION,
    TOT_NET_DURATION,
    TOT_DNS_DURATION,
    NET_ERR_CONN,
    NET_ERR_RX_TX,
    NET_ERR_DNS,
    FS_ERR_OPEN_CLOSE,
    FS_ERR_STAT,
    FS_ERR_READ_WRITE,
};

// File types; stream or fd
enum fs_type_t {
    FD,
    STREAM
};

typedef struct metric_counters_t {
    uint64_t openPorts;
    uint64_t netConnectionsUdp;
    uint64_t netConnectionsTcp;
    uint64_t netConnectionsOther;
    uint64_t netrxBytes;
    uint64_t nettxBytes;
    uint64_t readBytes;
    uint64_t writeBytes;
    uint64_t numSeek;
    uint64_t numStat;
    uint64_t numOpen;
    uint64_t numClose;
    uint64_t numDNS;
    uint64_t fsDurationNum;
    uint64_t fsDurationTotal;
    uint64_t connDurationNum;
    uint64_t connDurationTotal;
    uint64_t dnsDurationNum;
    uint64_t dnsDurationTotal;
    uint64_t netConnectErrors;
    uint64_t netTxRxErrors;
    uint64_t netDNSErrors;
    uint64_t fsOpenCloseErrors;
    uint64_t fsRdWrErrors;
    uint64_t fsStatErrors;
} metric_counters;

typedef struct {
    struct {
        int open_close;
        int read_write;
        int stat;
        int seek;
        int error;
    } fs;
    struct {
        int open_close;
        int rx_tx;
        int dns;
        int error;
        int dnserror;
    } net;
} summary_t;

typedef struct rtconfig_t {
    int numNinfo;
    int numFSInfo;
    bool tsc_invariant;
    bool tsc_rdtscp;
    summary_t summarize;
    uint64_t freq;
    const char *cmddir;
    char hostname[MAX_HOSTNAME];
    char procname[MAX_PROCNAME];
    pid_t pid;
} rtconfig;

typedef struct thread_timing_t {
    unsigned interval;                   // in seconds
    time_t startTime; 
    bool once;
    pthread_t periodicTID;
} thread_timing;

typedef struct net_info_t {
    int active;
    int type;
    int addrType;
    uint64_t numTX;
    uint64_t numRX;
    uint64_t txBytes;
    uint64_t rxBytes;
    bool dnsSend;
    uint64_t startTime;
    uint64_t numDuration;
    uint64_t totalDuration;
    uint64_t uid;
    char dnsName[MAX_HOSTNAME];
    struct sockaddr_storage localConn;
    struct sockaddr_storage remoteConn;
} net_info;

typedef struct {
    uint64_t initial;
    uint64_t duration;
} elapsed_t;

typedef struct fs_info_t {
    int active;
    enum fs_type_t type;
    uint64_t numOpen;
    uint64_t numClose;
    uint64_t numSeek;
    uint64_t numRead;
    uint64_t numWrite;
    uint64_t readBytes;
    uint64_t writeBytes;
    uint64_t numDuration;
    uint64_t totalDuration;
    uint64_t uid;
    char path[PATH_MAX];
} fs_info;

typedef struct interposed_funcs_t {
    void (*vsyslog)(int, const char *, va_list);
    pid_t (*fork)(void);
    int (*open)(const char *, int, ...);
    int (*openat)(int, const char *, int, ...);
    FILE *(*fopen)(const char *, const char *);
    FILE *(*freopen)(const char *, const char *, FILE *);
    int (*creat)(const char *, mode_t);
    int (*close)(int);
    int (*fclose)(FILE *);
    int (*fcloseall)(void);
    ssize_t (*read)(int, void *, size_t);
    ssize_t (*readv)(int, const struct iovec *, int);
    size_t (*fread)(void *, size_t, size_t, FILE *);
    char *(*fgets)(char *, int, FILE *);
    int (*fscanf)(FILE *, const char *, ...);
    int (*fputs)(const char *, FILE *);
    int (*fputs_unlocked)(const char *, FILE *);
    int (*fputws)(const wchar_t *, FILE *);
    int (*fgetc)(FILE *);
    int (*fputc)(int, FILE *);
    int (*fputc_unlocked)(int, FILE *);
    wint_t (*fputwc)(wchar_t, FILE *);
    wint_t (*putwc)(wchar_t, FILE *);
    ssize_t (*getline)(char **, size_t *, FILE *);
    ssize_t (*getdelim)(char **, size_t *, int, FILE *);
    ssize_t (*pread)(int, void *, size_t, off_t);
    ssize_t (*write)(int, const void *, size_t);
    ssize_t (*writev)(int, const struct iovec *, int);
    size_t (*fwrite)(const void *, size_t, size_t, FILE *);
    ssize_t (*pwrite)(int, const void *, size_t, off_t);
    ssize_t (*sendfile)(int, int, off_t *, size_t);
    off_t (*lseek)(int, off_t, int);
    int (*fseek)(FILE *, long, int);
    int (*fseeko)(FILE *, off_t, int);
    long (*ftell)(FILE *);
    off_t (*ftello)(FILE *);
    int (*fgetpos)(FILE *, fpos_t *);
    int (*fsetpos)(FILE *, const fpos_t *);
    void (*rewind)(FILE *);
    int (*stat)(const char *, struct stat *);
    int (*lstat)(const char *, struct stat *);
    int (*fstat)(int, struct stat *);
    int (*statfs)(const char *, struct statfs *);
    int (*fstatfs)(int, struct statfs *);
    int (*statvfs)(const char *, struct statvfs *);
    int (*fstatvfs)(int, struct statvfs *);
    int (*fstatat)(int, const char *, struct stat *, int);
    int (*access)(const char *, int);
    int (*faccessat)(int, const char *, int, int);
    int (*fcntl)(int, int, ...);
    int (*dup)(int);
    int (*dup2)(int, int);
    int (*dup3)(int, int, int);
    int (*shutdown)(int, int);
    int (*socket)(int, int, int);
    int (*listen)(int, int);
    int (*bind)(int, const struct sockaddr *, socklen_t);
    int (*connect)(int, const struct sockaddr *, socklen_t);
    int (*accept)(int, struct sockaddr *, socklen_t *);
    int (*accept4)(int, struct sockaddr *, socklen_t *, int);
    ssize_t (*send)(int, const void *, size_t, int);
    ssize_t (*sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
    ssize_t (*sendmsg)(int, const struct msghdr *, int);
    ssize_t (*recv)(int, void *, size_t, int);
    ssize_t (*recvfrom)(int sockfd, void *buf, size_t len, int flags,
                                struct sockaddr *src_addr, socklen_t *addrlen);
    ssize_t (*recvmsg)(int, struct msghdr *, int);
    struct hostent *(*gethostbyname)(const char *);
    int (*gethostbyname_r)(const char *, struct hostent *, char *, size_t,
                          struct hostent **, int *);
    struct hostent *(*gethostbyname2)(const char *, int);
    int (*getaddrinfo)(const char *, const char *, const struct addrinfo *,
                       struct addrinfo **);
    // __LINUX__
    /*
     * We need to make these Linux only, but we're holding off until structiural changes are done.
     */
    int (*open64)(const char *, int, ...);
    int (*openat64)(int, const char *, int, ...);
    int (*__open_2)(const char *, int);
    int (*__open64_2)(const char *, int);
    int (*__openat_2)(int, const char *, int);
    FILE *(*fopen64)(const char *, const char *);
    FILE *(*freopen64)(const char *, const char *, FILE *);
    int (*creat64)(const char *, mode_t);
    ssize_t (*__read_chk)(int, void *, size_t, size_t);
    char *(*__fgets_chk)(char *, size_t, int, FILE *);
    char *(*fgets_unlocked)(char *, int, FILE *);
    wchar_t *(*fgetws)(wchar_t *, int, FILE *);
    wint_t (*fgetwc)(FILE *);
    wchar_t *(*__fgetws_chk)(wchar_t *, size_t, int, FILE *);
    size_t (*__fread_chk)(void *, size_t, size_t, size_t, FILE *);
    size_t (*fread_unlocked)(void *, size_t, size_t, FILE *);
    size_t (*__fread_unlocked_chk)(void *, size_t, size_t, size_t, FILE *);
    ssize_t (*__getdelim)(char **, size_t *, int, FILE *);
    ssize_t (*pread64)(int, void *, size_t, off_t);
    ssize_t (*preadv)(int, const struct iovec *, int, off_t);
    ssize_t (*preadv2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*preadv64v2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*__pread_chk)(int, void *, size_t, off_t, size_t);
    ssize_t (*pwrite64)(int, const void *, size_t, off_t);
    ssize_t (*pwritev)(int, const struct iovec *, int, off_t);
    ssize_t (*pwritev64)(int, const struct iovec *, int, off64_t);
    ssize_t (*pwritev2)(int, const struct iovec *, int, off_t, int);
    ssize_t (*pwritev64v2)(int, const struct iovec *, int, off_t, int);
    size_t (*fwrite_unlocked)(const void *, size_t, size_t, FILE *);
    ssize_t (*sendfile64)(int, int, off64_t *, size_t);
    off64_t (*lseek64)(int, off64_t, int);
    int (*fseeko64)(FILE *, off64_t, int);
    off64_t (*ftello64)(FILE *);
    int (*fgetpos64)(FILE *, fpos64_t *);
    int (*fsetpos64)(FILE *stream, const fpos64_t *pos);
    int (*statfs64)(const char *, struct statfs64 *);
    int (*fstatfs64)(int, struct statfs64 *);
    int (*statvfs64)(const char *, struct statvfs64 *);
    int (*fstatvfs64)(int, struct statvfs64 *);
    int (*fstatat64)(int, const char *, struct stat64 *, int);
    int (*__xstat)(int, const char *, struct stat *);
    int (*__xstat64)(int, const char *, struct stat64 *);
    int (*__fxstat)(int, int, struct stat *);
    int (*__fxstat64)(int, int, struct stat64 *);
    int (*__fxstatat)(int, int, const char *, struct stat *, int);
    int (*__fxstatat64)(int, int, const char *, struct stat64 *, int);
    int (*__lxstat)(int, const char *, struct stat *);
    int (*__lxstat64)(int, const char *, struct stat64 *);
    int (*fcntl64)(int, int, ...);
    long (*syscall)(long, ...);
    int (*prctl)(int, unsigned long, unsigned long, unsigned long, unsigned long);

#if defined(__LINUX__) && defined(__STATX__)
    int (*statx)(int, const char *, int, unsigned int, struct statx *);
#endif // __LINUX__ && __STATX__

#ifdef __MACOS__
    int (*accept$NOCANCEL)(int, struct sockaddr *, socklen_t *);
    int (*close$NOCANCEL)(int);
    int (*close_nocancel)(int);
    int (*guarded_close_np)(int, void *);
    ssize_t (*__sendto_nocancel)(int, const void *, size_t, int,
                                 const struct sockaddr *, socklen_t);
    int32_t (*DNSServiceQueryRecord)(void *, uint32_t, uint32_t, const char *,
                                      uint16_t, uint16_t, void *, void *);
#endif // __MACOS__
} interposed_funcs;

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
    }                                                                  \
    doThread();

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = _dl_sym(RTLD_NEXT, #func, func)) == NULL) {   \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return;                                                    \
       }                                                               \
    }                                                                  \
    doThread();

#else
#define WRAP_CHECK(func, rc)                                           \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return rc;                                                 \
       }                                                               \
    }                                                                  \
    doThread();

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLog("ERROR: "#func":NULL\n", -1, CFG_LOG_ERROR);      \
            return;                                                    \
       }                                                               \
    }                                                                  \
    doThread();
#endif // __LINUX__

#endif // __WRAP_H__
