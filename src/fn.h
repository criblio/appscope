#ifndef __FN_H__
#define __FN_H__

#include <wchar.h>
#include <sys/statvfs.h>
#include <netdb.h>
#ifdef __linux__
#include <sys/epoll.h>
#include <poll.h>
#include <sys/vfs.h>
#include <linux/aio_abi.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#endif // __linux__
#include <signal.h>
#include "../contrib/tls/tls.h"
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <arpa/nameser.h>
#include <dirent.h>

#ifdef __linux__
#ifndef io_context_t
#define io_context_t unsigned long
#endif
#endif

#ifdef __APPLE__
#include <sys/mount.h>

#ifndef off64_t
typedef uint64_t off64_t;
#endif
#ifndef fpos64_t
typedef uint64_t fpos64_t;
#endif

#ifndef nfds_t
typedef unsigned int nfds_t;
#endif
#endif // __APPLE__


typedef struct {
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
    int (*unlink)(const char *); 
    int (*unlinkat)(int, const char *, int);   
    ssize_t (*read)(int, void *, size_t);
    ssize_t (*readv)(int, const struct iovec *, int);
    size_t (*fread)(void *, size_t, size_t, FILE *);
    char *(*fgets)(char *, int, FILE *);
    int (*fscanf)(FILE *, const char *, ...);
    int (*putchar)(int);
    int (*puts)(const char *);
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
    ssize_t (*__recv_chk)(int, void *, size_t, size_t, int);
    ssize_t (*recvfrom)(int, void *, size_t, int,
                                struct sockaddr *, socklen_t *);
    ssize_t (*__recvfrom_chk)(int, void *, size_t, size_t, int,
                                struct sockaddr *, socklen_t *);
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
    ssize_t (*__pread64_chk)(int, void *, size_t, off_t, size_t);
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
    int (*sigaction)(int, const struct sigaction *, struct sigaction *);
    void (*_exit)(int);
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
    gnutls_transport_ptr_t (*gnutls_transport_get_ptr)(gnutls_session_t);
    PRFileDesc *(*SSL_ImportFD)(PRFileDesc *, PRFileDesc *);
    void *(*dlopen)(const char *, int);
    int (*PR_FileDesc2NativeHandle)(PRFileDesc *);
    void (*PR_SetError)(PRErrorCode, PRInt32);
    int (*execve)(const char *, char * const *, char * const *);
    int (*poll)(struct pollfd *, nfds_t, int);
    int (*__poll_chk)(struct pollfd *, nfds_t, int, size_t);
    int (*select)(int, fd_set *, fd_set *, fd_set *, struct timeval *);
    int (*nanosleep)(const struct timespec *, struct timespec *);
    int	(*ns_initparse)(const unsigned char *, int, ns_msg *);
    int	(*ns_parserr)(ns_msg *, ns_sect, int, ns_rr *);
    size_t (*__stdout_write)(FILE *, const unsigned char *, size_t);
    size_t (*__stderr_write)(FILE *, const unsigned char *, size_t);
    int (*__fprintf_chk)(FILE *, int, const char *, ...);
    void *(*__memset_chk)(void *, int, size_t, size_t);
    void *(*__memcpy_chk)(void *, const void *, size_t, size_t);
    int (*__sprintf_chk)(char *, int, size_t, const char *, ...);
    long int (*__fdelt_chk)(long int);
#ifdef __linux__
    // Couldn't easily get struct definitions for these on mac
    int (*statvfs64)(const char *, struct statvfs64 *);
    int (*fstatvfs64)(int, struct statvfs64 *);
    int (*epoll_wait)(int, struct epoll_event *, int, int);
    int (*__overflow)(FILE *, int);
    ssize_t (*__write_libc)(int, const void *, size_t);
    ssize_t (*__write_pthread)(int, const void *, size_t);
    int (*epoll_pwait)(int, struct epoll_event *, int, int, const sigset_t *);
    int (*ppoll)(struct pollfd *, nfds_t, const struct timespec *, const sigset_t *);
    int (*__ppoll_chk)(struct pollfd *, nfds_t, const struct timespec *, const sigset_t *, size_t);
    int (*pause)(void);
    int (*sigsuspend)(const sigset_t *);
    int (*sigwaitinfo)(const sigset_t *, siginfo_t *);
    int (*sigtimedwait)(const sigset_t *, siginfo_t *, const struct timespec *);
    int (*pselect)(int, fd_set *, fd_set *, fd_set *, const struct timespec *, const sigset_t *);
    int (*msgsnd)(int, const void *, size_t, int);
    ssize_t (*msgrcv)(int, void *, size_t, long, int);
    int (*semop)(int, struct sembuf *, size_t);
    int (*semtimedop)(int, struct sembuf *, size_t, const struct timespec *);
    int (*clock_nanosleep)(clockid_t, int, const struct timespec *, struct timespec *);
    int (*usleep)(useconds_t);
    int (*io_getevents)(io_context_t, long, long, struct io_event *, struct timespec *);
    int (*sendmmsg)(int, struct mmsghdr *, unsigned int, int);
    int (*recvmmsg)(int, struct mmsghdr *, unsigned int, int, struct timespec *);
    int (*pthread_create)(pthread_t *, const pthread_attr_t *,
                          void *(*)(void *), void *);
    int (*getentropy)(void *, size_t);
    void (*__ctype_init)(void);
    int (*__register_atfork)(void (*) (void), void (*) (void), void (*) (void), void *);
    void (*uv__read)(void *);
    int (*uv_fileno)(const void *, int *);
    DIR *(*opendir)(const char *);
    int (*closedir)(DIR *);
    struct dirent *(*readdir)(DIR *);
    void *(*malloc)(size_t);
    void (*free)(void *);
    void *(*calloc)(size_t, size_t);
    void *(*realloc)(void *, size_t);
    size_t (*malloc_usable_size)(void *);
    char *(*strdup)(const char *s);
    void *(*mmap)(void *, size_t, int, int, int, off_t);
    int   (*munmap)(void *, size_t);
#endif // __linux__

#if defined(__linux__) && defined(__STATX__)
    int (*statx)(int, const char *, int, unsigned int, struct statx *);
#endif // __linux__ && __STATX__

#ifdef __APPLE__
    int (*accept$NOCANCEL)(int, struct sockaddr *, socklen_t *);
    int (*close$NOCANCEL)(int);
    int (*close_nocancel)(int);
    int (*guarded_close_np)(int, void *);
    ssize_t (*__sendto_nocancel)(int, const void *, size_t, int,
                                 const struct sockaddr *, socklen_t);
    int32_t (*DNSServiceQueryRecord)(void *, uint32_t, uint32_t, const char *,
                                      uint16_t, uint16_t, void *, void *);
#endif // __APPLE__

    // These functions are not interposed.  They're here because
    // we've seen applications override the weak glibc implementation,
    // where our library needs to use the glibc instance.
    // setenv was overridden in bash.
    int (*setenv)(const char *name, const char *value, int overwrite);

    // intended for shells and any app that has it's own setenv
    int (*app_setenv)(const char *name, const char *value, int overwrite);
} interposed_funcs_t;

extern interposed_funcs_t g_fn;

void initFn(void);
void initFn_musl(void);

#endif // __FN_H__
