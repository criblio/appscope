#ifndef __WRAP_H__
#define __WRAP_H__

#define _GNU_SOURCE
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <netdb.h>
#include <pthread.h>
#include <wchar.h>

#include <sys/param.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

#include "../contrib/tls/tls.h"

#define DEBUG 0
#define EXPORT __attribute__((visibility("default")))
#define EXPORTOFF  __attribute__((visibility("hidden")))
#define EXPORTON __attribute__((visibility("default")))

#define DYN_CONFIG_PREFIX "scope"
#define MAXTRIES 10

typedef struct nss_list_t {
    uint64_t id;
    PRIOMethods *ssl_methods;
    PRIOMethods *ssl_int_methods;
} nss_list;

typedef struct thread_timing_t {
    unsigned interval;                   // in seconds
    time_t startTime; 
    bool once;
    pthread_t periodicTID;
    const struct sigaction *act;
} thread_timing;

typedef struct {
    uint64_t initial;
    uint64_t duration;
} elapsed_t;

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
    int (*gethostbyname2_r)(const char *, int, struct hostent *, char *, size_t,
                            struct hostent **, int *);
    int (*getaddrinfo)(const char *, const char *, const struct addrinfo *,
                       struct addrinfo **);
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
    int (*nanosleep)(const struct timespec *, struct timespec *);
    int (*epoll_wait)(int, struct epoll_event *, int, int);
    int (*select)(int, fd_set *, fd_set *, fd_set *, struct timeval *);
    int (*sigsuspend)(const sigset_t *);
    int (*sigaction)(int, const struct sigaction *, struct sigaction *);

    int (*SSL_read)(SSL *, void *, int);
    int (*SSL_write)(SSL *, const void *, int);
    int (*SSL_get_fd)(const SSL *);
    ssize_t (*gnutls_record_recv)(gnutls_session_t, void *, size_t);
    ssize_t (*gnutls_record_send)(gnutls_session_t, const void *, size_t);
    ssize_t (*gnutls_record_recv_early_data)(gnutls_session_t, void *, size_t);
    ssize_t (*gnutls_record_recv_packet)(gnutls_session_t, gnutls_packet_t *);
    ssize_t (*gnutls_record_recv_seq)(gnutls_session_t, void *, size_t, unsigned char *);
    ssize_t (*gnutls_record_send2)(gnutls_session_t, const void *, size_t, size_t, unsigned);
    ssize_t (*gnutls_record_send_early_data)(gnutls_session_t, const void *, size_t);
    ssize_t (*gnutls_record_send_range)(gnutls_session_t, const void *, size_t, const gnutls_range_st *);
    PRFileDesc *(*SSL_ImportFD)(PRFileDesc *, PRFileDesc *);

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

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);

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

#endif // __WRAP_H__
