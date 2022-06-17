#define _GNU_SOURCE
#include "scopestdlib.h"

#include <limits.h>
#include <malloc.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// Internal standard library references
extern void  scopelibc_init_vdso_ehdr(unsigned long);
extern void  scopelibc_lock_before_fork_op(void);
extern void  scopelibc_unlock_after_fork_op(int);

// Memory management handling operations
extern void*  scopelibc_memalign(size_t, size_t);
extern void*  scopelibc_malloc(size_t);
extern void*  scopelibc_calloc(size_t, size_t);
extern void*  scopelibc_realloc(void *, size_t);
extern void   scopelibc_free(void *);
extern void*  scopelibc_mmap(void *, size_t, int, int, int, off_t);
extern int    scopelibc_munmap(void *, size_t);
extern FILE*  scopelibc_open_memstream(char **, size_t *);
extern void*  scopelibc_memset(void *, int, size_t);
extern void*  scopelibc_memmove(void *, const void *, size_t);
extern int    scopelibc_memcmp(const void *, const void *, size_t);
extern int    scopelibc_mprotect(void *, size_t, int);
extern void*  scopelibc_memcpy(void *, const void *, size_t);
extern int    scopelibc_mlock(const void *, size_t);
extern int    scopelibc_msync(void *, size_t, int);
extern int    scopelibc_mincore(void *, size_t, unsigned char *);

// File handling operations
extern FILE*          scopelibc_fopen(const char *, const char *);
extern int            scopelibc_fclose(FILE *);
extern FILE*          scopelibc_fdopen(int, const char *);
extern int            scopelibc_close(int);
extern ssize_t        scopelibc_read(int, void *, size_t);
extern size_t         scopelibc_fread(void *, size_t, size_t, FILE *);
extern ssize_t        scopelibc_write(int, const void *, size_t);
extern size_t         scopelibc_fwrite(const void *, size_t, size_t, FILE *);
extern char *         scopelibc_fgets(char *, int, FILE *);
extern ssize_t        scopelibc_getline(char **, size_t *, FILE *);
extern int            scopelibc_puts(const char*);
extern int            scopelibc_setvbuf(FILE *, char *, int, size_t);
extern int            scopelibc_fflush(FILE *);
extern char*          scopelibc_dirname(char *);
extern DIR*           scopelibc_opendir(const char *);
extern struct dirent* scopelibc_readdir(DIR *);
extern int            scopelibc_closedir(DIR *);
extern int            scopelibc_access(const char *, int);
extern FILE*          scopelibc_fmemopen(void *, size_t, const char *);
extern long           scopelibc_ftell(FILE *);
extern int            scopelibc_fseek(FILE *, long, int);
extern int            scopelibc_unlink(const char *);
extern int            scopelibc_dup2(int, int);
extern char*          scopelibc_basename(char *);
extern int            scopelibc_stat(const char *, struct stat *);
extern int            scopelibc_chmod(const char *, mode_t);
extern int            scopelibc_fchmod(int, mode_t);
extern int            scopelibc_feof(FILE *);
extern int            scopelibc_fileno(FILE *);
extern int            scopelibc_flock(int, int);
extern int            scopelibc_fstat(int, struct stat *);
extern int            scopelibc_mkdir(const char *, mode_t);
extern int            scopelibc_chdir(const char *);
extern int            scopelibc_rmdir(const char *);
extern char*          scopelibc_get_current_dir_name(void);
extern char*          scopelibc_getcwd(char *, size_t);
extern int            scopelibc_lstat(const char *, struct stat *);
extern int            scopelibc_rename(const char *, const char *);
extern int            scopelibc_remove(const char *);
extern int            scopelibc_pipe2(int [2], int);
extern void           scopelibc_setbuf(FILE *, char *);

// String handling operations
extern char*               scopelibc_realpath(const char *, char *);
extern ssize_t             scopelibc_readlink(const char *, char *, size_t);
extern char*               scopelibc_strdup(const char *);
extern int                 scopelibc_vasprintf(char **, const char *, va_list);
extern size_t              scopelibc_strftime(char *, size_t, const char *, const struct tm *);
extern size_t              scopelibc_strlen(const char*);
extern size_t              scopelibc_strnlen(const char *, size_t);
extern char*               scopelibc_strerror(int);
extern int                 scopelibc_strerror_r(int, char *, size_t);
extern double              scopelibc_strtod(const char *, char **);
extern long                scopelibc_strtol(const char *, char **, int);
extern long long           scopelibc_strtoll(const char *, char **, int);
extern unsigned long       scopelibc_strtoul(const char *, char **, int);
extern unsigned long long  scopelibc_strtoull(const char *, char **, int);
extern char*               scopelibc_strchr(const char *, int);
extern char*               scopelibc_strrchr(const char *, int);
extern char*               scopelibc_strstr(const char *, const char *);
extern int                 scopelibc_vsnprintf(char *, size_t, const char *, va_list);
extern int                 scopelibc_vfprintf(FILE *, const char *, va_list);
extern int                 scopelibc_vprintf(const char *, va_list);
extern int                 scopelibc_vsscanf(const char *, const char *, va_list);
extern int                 scopelibc_strcmp(const char *, const char *);
extern int                 scopelibc_strncmp(const char *, const char *, size_t);
extern int                 scopelibc_strcasecmp(const char *, const char *);
extern char*               scopelibc_strcpy(char *, const char *);
extern char*               scopelibc_strncpy(char *, const char *, size_t);
extern char*               scopelibc_stpcpy(char *, const char *);
extern char*               scopelibc_stpncpy(char *, const char *, size_t);
extern size_t              scopelibc_strcspn(const char *, const char *);
extern char*               scopelibc_strcat(char *, const char *);
extern char*               scopelibc_strncat(char *, const char *, size_t);
extern char*               scopelibc_strpbrk(const char *, const char *);
extern char*               scopelibc_strcasestr(const char *, const char *);
extern char*               scopelibc_strtok(char *, const char *);
extern char*               scopelibc_strtok_r(char *, const char *, char **);
extern const char*         scopelibc_gai_strerror(int);

// Network handling operations
extern int             scopelibc_gethostname(char *, size_t);
extern int             scopelibc_getsockname(int, struct sockaddr *, socklen_t *);
extern int             scopelibc_getsockopt(int, int, int, void *, socklen_t *);
extern int             scopelibc_setsockopt(int, int, int, const void *, socklen_t);
extern int             scopelibc_socket(int, int, int);
extern int             scopelibc_bind(int, const struct sockaddr *, socklen_t);
extern int             scopelibc_accept(int, struct sockaddr *, socklen_t *);
extern int             scopelibc_connect(int, const struct sockaddr *, socklen_t);
extern int             scopelibc_listen(int, int);
extern void            scopelibc_rewind(FILE *);
extern ssize_t         scopelibc_send(int, const void *, size_t, int);
extern ssize_t         scopelibc_sendmsg(int, const struct msghdr *, int);
extern ssize_t         scopelibc_recv(int, void *, size_t, int);
extern ssize_t         scopelibc_recvmsg(int, struct msghdr *, int);
extern ssize_t         scopelibc_recvfrom(int, void *, size_t, int, struct sockaddr *, socklen_t *);
extern int             scopelibc_shutdown(int, int);
extern int             scopelibc_poll(struct pollfd *, nfds_t, int);
extern int             scopelibc_select(int, fd_set *, fd_set *, fd_set *, struct timeval *);
extern int             scopelibc_getaddrinfo(const char *, const char *, const struct addrinfo *, struct addrinfo **);
extern void            scopelibc_freeaddrinfo(struct addrinfo *);
extern int             scopelibc_copyaddrinfo(struct sockaddr *, socklen_t, struct addrinfo **);
extern int             scopelibc_getnameinfo(const struct sockaddr *, socklen_t, char *, socklen_t, char *, socklen_t, int);
extern int             scopelibc_getpeername(int, struct sockaddr *, socklen_t *);
extern struct hostent* scopelibc_gethostbyname(const char *);
extern const char*     scopelibc_inet_ntop(int, const void *, char *, socklen_t);
extern uint16_t        scopelibc_ntohs(uint16_t);
extern uint16_t        scopelibc_htons(uint16_t);

// Misc handling operations
extern int           scopelibc_atoi(const char *);
extern int           scopelibc_isspace(int);
extern int           scopelibc_isprint(int);
extern void          scopelibc_perror(const char*);
extern int           scopelibc_gettimeofday(struct timeval *, struct timezone *);
extern int           scopelibc_timer_create(clockid_t, struct sigevent *, timer_t *);
extern int           scopelibc_timer_settime(timer_t, int, const struct itimerspec *, struct itimerspec *);
extern int           scopelibc_timer_delete(timer_t);
extern time_t        scopelibc_time(time_t *);
extern struct tm*    scopelibc_localtime_r(const time_t *, struct tm *);
extern struct tm*    scopelibc_gmtime_r(const time_t *, struct tm *);
extern unsigned int  scopelibc_sleep(unsigned int);
extern int           scopelibc_usleep(useconds_t);
extern int           scopelibc_nanosleep(const struct timespec *, struct timespec *);
extern int           scopelibc_sigaction(int, const struct sigaction *, struct sigaction *);
extern int           scopelibc_sigemptyset(sigset_t *);
extern int           scopelibc_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
extern int           scopelibc_pthread_barrier_init(pthread_barrier_t *, const pthread_barrierattr_t *, unsigned);
extern int           scopelibc_pthread_barrier_destroy(pthread_barrier_t *);
extern int           scopelibc_pthread_barrier_wait(pthread_barrier_t *);
extern int           scopelibc_dlclose(void *);
extern int           scopelibc_ns_initparse(const unsigned char *, int, ns_msg *);
extern int           scopelibc_ns_parserr(ns_msg *, ns_sect, int, ns_rr *);
extern int           scopelibc_getgrgid_r(gid_t, struct group *, char *, size_t, struct group **);
extern int           scopelibc_getpwuid_r(uid_t, struct passwd *, char *, size_t, struct passwd **);
extern pid_t         scopelibc_getpid(void);
extern pid_t         scopelibc_getppid(void);
extern uid_t         scopelibc_getuid(void);
extern gid_t         scopelibc_getgid(void);
extern void*         scopelibc_dlopen(const char *, int);
extern int           scopelibc_dlclose(void *);
extern void*         scopelibc_dlsym(void *, const char *);
extern long          scopelibc_ptrace(int, pid_t, void *, void *);
extern pid_t         scopelibc_waitpid(pid_t, int *, int);
extern char*         scopelibc_getenv(const char *);
extern int           scopelibc_setenv(const char *, const char *, int);
extern struct lconv* scopelibc_localeconv(void);
extern int           scopelibc_shm_open(const char *, int, mode_t);
extern int           scopelibc_shm_unlink(const char *);
extern long          scopelibc_sysconf(int);
extern int           scopelibc_mkstemp(char *);
extern int           scopelibc_clock_gettime(clockid_t, struct timespec *);
extern int           scopelibc_getpagesize(void);
extern int           scopelibc_uname(struct utsname *);
extern int           scopelibc_arch_prctl(int, unsigned long);
extern int           scopelibc_getrusage(int , struct rusage *);
extern int           scopelibc_atexit(void (*)(void));
extern int           scopelibc_tcsetattr(int, int, const struct termios *);
extern int           scopelibc_tcgetattr(int, struct termios *);
extern void*         scopelibc_shmat(int, const void *, int);
extern int           scopelibc_shmdt(const void *);
extern int           scopelibc_shmget(key_t, size_t, int);
extern int           scopelibc_sched_getcpu(void);

static int g_go_static;

// TODO consider moving GOAppState API somewhere else
void
scopeSetGoAppStateStatic(int static_state) {
    g_go_static = static_state;
}

int
scopeGetGoAppStateStatic(void) {
    return g_go_static;
}

int
SCOPE_DlIteratePhdr(int (*callback) (struct dl_phdr_info *info, size_t size, void *data), void *data)
{
    // TODO provide implementation for static GO
    // We cannot use dl_iterate_phdr since it uses TLS
    // To retrieve information about go symbols we need to implement own
    return (!scopeGetGoAppStateStatic()) ? dl_iterate_phdr(callback, data) : 0;
}

// Internal library operations

void
scope_init_vdso_ehdr(void) {
    unsigned long ehdr = getauxval(AT_SYSINFO_EHDR);
    scopelibc_init_vdso_ehdr(ehdr);
}

void
scope_op_before_fork(void) {
    scopelibc_lock_before_fork_op();
}

void
scope_op_after_fork(int who) {
    scopelibc_unlock_after_fork_op(who);
}

// Memory management handling operations

void*
scope_memalign(size_t alignment, size_t size) {
    return scopelibc_memalign(alignment, size);
}

void*
scope_malloc(size_t size) {
    return scopelibc_malloc(size);
}

void*
scope_calloc(size_t nmemb, size_t size) {
    return scopelibc_calloc(nmemb, size);
}

void*
scope_realloc(void *ptr, size_t size) {
    return scopelibc_realloc(ptr, size);
}

void
scope_free(void *ptr) {
    scopelibc_free(ptr);
}

void*
scope_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return scopelibc_mmap(addr, length, prot, flags, fd, offset);
}

int
scope_munmap(void *addr, size_t length) {
    return scopelibc_munmap(addr, length);
}

FILE*
scope_open_memstream(char **ptr, size_t *sizeloc) {
    return scopelibc_open_memstream(ptr, sizeloc);
}

void*
scope_memset(void *s, int c, size_t n) {
    return scopelibc_memset(s, c, n);
}

void*
scope_memmove(void *dest, const void *src, size_t n) {
    return scopelibc_memmove(dest, src, n);
}

int
scope_memcmp(const void *s1, const void *s2, size_t n) {
    return scopelibc_memcmp(s1, s2, n);
}

int
scope_mprotect(void *addr, size_t len, int prot) {
    return scopelibc_mprotect(addr, len, prot);
}

void*
scope_memcpy(void *restrict dest, const void *restrict src, size_t n) {
    return scopelibc_memcpy(dest, src, n);
}

int
scope_mlock(const void *addr, size_t len) {
    return scopelibc_mlock(addr, len);
}

int
scope_msync(void *addr, size_t length, int flags) {
    return scopelibc_msync(addr, length, flags);
}

int
scope_mincore(void *addr, size_t length, unsigned char *vec) {
    return scopelibc_mincore(addr, length, vec);
}

// File handling operations

FILE*
scope_fopen( const char * filename, const char * mode) {
    return scopelibc_fopen(filename, mode);
}

int
scope_fclose(FILE * stream) {
    return scopelibc_fclose(stream);
}

FILE*
scope_fdopen(int fd, const char *mode) {
    return scopelibc_fdopen(fd, mode);
}

int
scope_close(int fd) {
    return scopelibc_close(fd);
}

ssize_t
scope_read(int fd, void *buf, size_t count) {
    return scopelibc_read(fd, buf, count);
}

size_t
scope_fread(void *restrict ptr, size_t size, size_t nmemb, FILE *restrict stream) {
    return scopelibc_fread(ptr, size, nmemb, stream);
}

ssize_t
scope_write(int fd, const void *buf, size_t count) {
    return scopelibc_write(fd, buf, count);
}

size_t
scope_fwrite(const void *restrict ptr, size_t size, size_t nmemb, FILE *restrict stream) {
    return scopelibc_fwrite(ptr, size, nmemb, stream);
}

char *
scope_fgets(char *restrict s, int n, FILE *restrict stream) {
    return scopelibc_fgets(s, n, stream);
}

ssize_t
scope_getline(char **restrict lineptr, size_t *restrict n, FILE *restrict stream) {
    return scopelibc_getline(lineptr, n, stream);
}

int
scope_puts(const char *s) {
    return scopelibc_puts(s);
}

int
scope_setvbuf(FILE *restrict stream, char *restrict buf, int type, size_t size) {
    return scopelibc_setvbuf(stream, buf, type, size);
}

int
scope_fflush(FILE *stream) {
    return scopelibc_fflush(stream);
}

char*
scope_dirname(char *path) {
    return scopelibc_dirname(path);
}

DIR*
scope_opendir(const char *name) {
    return scopelibc_opendir(name);
}
struct dirent*
scope_readdir(DIR *dirp) {
    return scopelibc_readdir(dirp);
}

int
scope_closedir(DIR *dirp) {
    return scopelibc_closedir(dirp);
}

int scope_access(const char *pathname, int mode) {
    return scopelibc_access(pathname, mode);
}

FILE*
scope_fmemopen(void *buf, size_t size, const char *mode) {
    return scopelibc_fmemopen(buf, size, mode);
}

long
scope_ftell(FILE *stream) {
    return scopelibc_ftell(stream);
}

int
scope_fseek(FILE *stream, long offset, int whence) {
    return scopelibc_fseek(stream, offset, whence);
}

int
scope_unlink(const char *pathname) {
    return scopelibc_unlink(pathname);
}

int
scope_dup2(int oldfd, int newfd) {
    return scopelibc_dup2(oldfd, newfd);
}

char*
scope_basename(char *path) {
    return scopelibc_basename(path);
}

int
scope_stat(const char *restrict pathname, struct stat *restrict statbuf) {
    return scopelibc_stat(pathname, statbuf);
}

int
scope_chmod(const char *path, mode_t mode) {
    return scopelibc_chmod(path, mode);
}

int
scope_fchmod(int fildes, mode_t mode) {
    return scopelibc_fchmod(fildes, mode);
}

int
scope_feof(FILE *stream) {
    return scopelibc_feof(stream);
}

int
scope_fileno(FILE *stream) {
    return scopelibc_fileno(stream);
}

int
scope_flock(int fd, int operation) {
    return scopelibc_flock(fd, operation);
}

int
scope_fstat(int fd, struct stat *buf) {
    return scopelibc_fstat(fd, buf);
}

int
scope_mkdir(const char *pathname, mode_t mode) {
    return scopelibc_mkdir(pathname, mode);
}

int
scope_chdir(const char *path) {
    return scopelibc_chdir(path);
}

int
scope_rmdir(const char *pathname) {
    return scopelibc_rmdir(pathname);
}

char*
scope_get_current_dir_name(void){
    return scopelibc_get_current_dir_name();
}

char*
scope_getcwd(char *buf, size_t size) {
    return scopelibc_getcwd(buf, size);
}

int
scope_lstat(const char *restrict path, struct stat *restrict buf) {
    return scopelibc_lstat(path, buf);
}

int
scope_rename(const char *oldpath, const char *newpath) {
    return scopelibc_rename(oldpath, newpath);
}

int
scope_remove(const char *pathname) {
    return scopelibc_remove(pathname);
}

int
scope_pipe2(int pipefd[2], int flags) {
    return scopelibc_pipe2(pipefd, flags);
}

void
scope_setbuf(FILE *restrict stream, char *restrict buf) {
    scopelibc_setbuf(stream, buf);
}

char*
scope_strcpy(char *restrict dest, const char *src) {
    return scopelibc_strcpy(dest, src);
}

char*
scope_strncpy(char *restrict dest, const char *restrict src, size_t n) {
    return scopelibc_strncpy(dest, src, n);
}

char*
scope_stpcpy(char *restrict dest, const char *restrict src) {
    return scopelibc_stpcpy(dest, src);
}

char*
scope_stpncpy(char *restrict dest, const char *restrict src, size_t n) {
    return scopelibc_stpncpy(dest, src, n);
}


// String handling operations

char*
scope_realpath(const char *restrict path, char *restrict resolved_path) {
    return scopelibc_realpath(path, resolved_path);
}

ssize_t
scope_readlink(const char *restrict pathname, char *restrict buf, size_t bufsiz) {
    return scopelibc_readlink(pathname, buf, bufsiz);
}

char*
scope_strdup(const char *s) {
    return scopelibc_strdup(s);
}

int
scope_vasprintf(char **strp, const char *fmt, va_list ap) {
    return scopelibc_vasprintf(strp, fmt, ap);
}

size_t
scope_strftime(char *restrict s, size_t max, const char *restrict format, const struct tm *restrict tm) {
    return scopelibc_strftime(s, max, format, tm);
}

size_t
scope_strlen(const char *s) {
    return scopelibc_strlen(s);
}

size_t
scope_strnlen(const char *s, size_t maxlen) {
    return scopelibc_strnlen(s, maxlen);
}

char*
scope_strerror(int errnum) {
    return scopelibc_strerror(errnum);
}

int
scope_strerror_r(int err, char *buf, size_t buflen) {
    return scopelibc_strerror_r(err, buf, buflen);
}

double
scope_strtod(const char *restrict nptr, char **restrict endptr) {
    return scopelibc_strtod(nptr, endptr);
}

long
scope_strtol(const char *restrict nptr, char **restrict endptr, int base) {
    return scopelibc_strtol(nptr, endptr, base);
}

long long
scope_strtoll(const char *restrict nptr, char **restrict endptr, int base) {
    return scopelibc_strtoll(nptr, endptr, base);
}

unsigned long
scope_strtoul(const char *restrict nptr, char **restrict endptr, int base) {
    return scopelibc_strtoul(nptr, endptr, base);
}

unsigned long long
scope_strtoull(const char *restrict nptr, char **restrict endptr, int base) {
    return scopelibc_strtoull(nptr, endptr, base);
}

char*
scope_strchr(const char *s, int c) {
    return scopelibc_strchr(s, c);
}

char*
scope_strrchr(const char *s, int c) {
    return scopelibc_strrchr(s, c);
}

char*
scope_strstr(const char *haystack, const char *needle) {
    return scopelibc_strstr(haystack, needle);
}

int
scope_vsnprintf(char *str, size_t size, const char *format, va_list ap) {
    return scopelibc_vsnprintf(str, size, format, ap);
}

int
scope_vfprintf(FILE *stream, const char *format, va_list ap) {
    return scopelibc_vfprintf(stream, format, ap);
}

int
scope_vprintf(const char *format, va_list ap) {
    return scopelibc_vprintf(format, ap);
}

int
scope_vsscanf(const char *str, const char *format, va_list ap) {
    return scopelibc_vsscanf(str, format, ap);
}

int
scope_strcmp(const char *s1, const char *s2) {
    return scopelibc_strcmp(s1, s2);
}

int
scope_strncmp(const char *s1, const char *s2, size_t n) {
    return scopelibc_strncmp(s1, s2, n);
}

int
scope_strcasecmp(const char *s1, const char *s2) {
    return scopelibc_strcasecmp(s1, s2);
}

size_t
scope_strcspn(const char *s, const char *reject) {
    return scopelibc_strcspn(s, reject);
}

char*
scope_strcat(char *restrict dest, const char *restrict src) {
    return scopelibc_strcat(dest, src);
}

char*
scope_strncat(char *restrict dest, const char *restrict src, size_t n) {
    return scopelibc_strncat(dest, src, n);
}

char*
scope_strpbrk(const char *s, const char *accept) {
    return scopelibc_strpbrk(s, accept);
}

char*
scope_strcasestr(const char * haystack, const char * needle) {
    return scopelibc_strcasestr(haystack, needle);
}

char*
scope_strtok(char *restrict str, const char *restrict delim) {
    return scopelibc_strtok(str, delim);
}

char*
scope_strtok_r(char *restrict str, const char *restrict delim, char **restrict saveptr) {
    return scopelibc_strtok_r(str, delim, saveptr);
}

const char*
scope_gai_strerror(int errcode) {
    return scopelibc_gai_strerror(errcode);
}


// Network handling operations

int
scope_gethostname(char *name, size_t len) {
    return scopelibc_gethostname(name, len);
}

int
scope_getsockname(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict addrlen) {
    return scopelibc_getsockname(sockfd, addr, addrlen);
}

int
scope_getsockopt(int sockfd, int level, int optname,  void *restrict optval, socklen_t *restrict optlen) {
    return scopelibc_getsockopt(sockfd, level, optname, optval, optlen);
}

int
scope_setsockopt(int sockfd, int level, int optname,  const void *restrict optval, socklen_t optlen) {
    return scopelibc_setsockopt(sockfd, level, optname, optval, optlen);
}

int
scope_socket(int domain, int type, int protocol) {
    return scopelibc_socket(domain, type, protocol);
}

int
scope_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    return scopelibc_bind(sockfd, addr, addrlen);
}

int
scope_accept(int sockfd, struct sockaddr *restrict addr, socklen_t *restrict addrlen) {
    return scopelibc_accept(sockfd, addr, addrlen);
}

int
scope_connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    return scopelibc_connect(sockfd, addr, addrlen);
}

int
scope_listen(int socket, int backlog) {
    return scopelibc_listen(socket, backlog);
}

void
scope_rewind(FILE *stream) {
    scopelibc_rewind(stream);
}

ssize_t
scope_send(int sockfd, const void *buf, size_t len, int flags) {
    return scopelibc_send(sockfd, buf, len, flags);
}

ssize_t
scope_sendmsg(int socket, const struct msghdr *message, int flags) {
    return scopelibc_sendmsg(socket, message, flags);
}

ssize_t
scope_recv(int sockfd, void *buf, size_t len, int flags) {
    return scopelibc_recv(sockfd, buf, len, flags);
}

ssize_t
scope_recvmsg(int socket, struct msghdr *message, int flags) {
    return scopelibc_recvmsg(socket, message, flags);
}

ssize_t
scope_recvfrom(int sockfd, void *restrict buf, size_t len, int flags, struct sockaddr *restrict src_addr, socklen_t *restrict addrlen) {
    return scopelibc_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
}

int
scope_shutdown(int sockfd, int how) {
    return scopelibc_shutdown(sockfd, how);
}

int
scope_poll(struct pollfd *fds, nfds_t nfds, int timeout) {
    return scopelibc_poll(fds, nfds, timeout);
}

int
scope_select(int nfds, fd_set *restrict readfds, fd_set *restrict writefds, fd_set *restrict exceptfds, struct timeval *restrict timeout) {
    return scopelibc_select(nfds, readfds, writefds, exceptfds, timeout);
}

int
scope_getaddrinfo(const char *restrict node, const char *restrict service, const struct addrinfo *restrict hints, struct addrinfo **restrict res) {
    return scopelibc_getaddrinfo(node, service, hints, res);
}

int
scope_copyaddrinfo(struct sockaddr *addr, socklen_t addrlen, struct addrinfo **restrict res) {
    return scopelibc_copyaddrinfo(addr, addrlen, res);
}

void
scope_freeaddrinfo(struct addrinfo *ai) {
    scopelibc_freeaddrinfo(ai);
}

int scope_getnameinfo(const struct sockaddr *restrict addr, socklen_t addrlen, char *restrict host, socklen_t hostlen, char *restrict serv, socklen_t servlen, int flags) {
    return scopelibc_getnameinfo(addr, addrlen, host, hostlen, serv, servlen, flags);
}

int
scope_getpeername(int fd, struct sockaddr *restrict addr, socklen_t *restrict len) {
    return scopelibc_getpeername(fd, addr, len);
}

struct hostent*
scope_gethostbyname(const char *name) {
    return scopelibc_gethostbyname(name);
}

const char*
scope_inet_ntop(int af, const void *restrict src, char *restrict dst, socklen_t size) {
    return scopelibc_inet_ntop(af, src, dst, size);
}

uint16_t
scope_ntohs(uint16_t netshort) {
    return scopelibc_ntohs(netshort);
}

uint16_t
scope_htons(uint16_t hostshort) {
    return scopelibc_htons(hostshort);
}

// Misc

int
scope_atoi(const char *nptr) {
    return scopelibc_atoi(nptr);
}

int
scope_isspace(int c) {
    return scopelibc_isspace(c);
}

int
scope_isprint(int c) {
    return scopelibc_isprint(c);
}

void
scope_perror(const char *s) {
    scopelibc_perror(s);
}

int
scope_gettimeofday(struct timeval *restrict tv, struct timezone *restrict tz) {
    return scopelibc_gettimeofday(tv, tz);
}

struct tm*
scope_localtime_r(const time_t *timep, struct tm *result) {
    return scopelibc_localtime_r(timep, result);
}

int
scope_timer_create(clockid_t clockid, struct sigevent *restrict sevp, timer_t *restrict timerid) {
    return scopelibc_timer_create(clockid, sevp, timerid);
}

int
scope_timer_settime(timer_t timerid, int flags, const struct itimerspec *restrict new_value, struct itimerspec *restrict old_value) {
    return scopelibc_timer_settime(timerid, flags, new_value, old_value);
}

int
scope_timer_delete(timer_t timerid) {
    return scopelibc_timer_delete(timerid);
}

time_t
scope_time(time_t *tloc) {
    return scopelibc_time(tloc);
}

struct tm*
scope_gmtime_r(const time_t *timep, struct tm *result) {
    return scopelibc_gmtime_r(timep, result);
}

unsigned int
scope_sleep(unsigned int seconds) {
    return scopelibc_sleep(seconds);
}

int
scope_usleep(useconds_t usec) {
    return scopelibc_usleep(usec);
}

int
scope_nanosleep(const struct timespec *req, struct timespec *rem) {
    return scopelibc_nanosleep(req, rem);
}

int
scope_sigaction(int signum, const struct sigaction *restrict act, struct sigaction *restrict oldact) {
    return scopelibc_sigaction(signum, act, oldact);
}

int
scope_sigemptyset(sigset_t * set) {
    return scopelibc_sigemptyset(set);
}

int
scope_pthread_create(pthread_t *restrict thread, const pthread_attr_t *restrict attr, void *(*start_routine)(void *), void *restrict arg) {
    return scopelibc_pthread_create(thread, attr, start_routine, arg);
}

int
scope_pthread_barrier_init(pthread_barrier_t *restrict barrier, const pthread_barrierattr_t *restrict attr, unsigned count) {
    return scopelibc_pthread_barrier_init(barrier, attr, count);
}

int
scope_pthread_barrier_destroy(pthread_barrier_t *barrier) {
    return scopelibc_pthread_barrier_destroy(barrier);
}

int
scope_pthread_barrier_wait(pthread_barrier_t *barrier) {
    return scopelibc_pthread_barrier_wait(barrier);
}

int
scope_ns_initparse(const u_char *msg, int msglen, ns_msg *handle) {
    return scopelibc_ns_initparse(msg, msglen, handle);
}

int
scope_ns_parserr(ns_msg *handle, ns_sect section, int rrnum, ns_rr *rr) {
    return scopelibc_ns_parserr(handle, section, rrnum, rr);
}

int
scope_getgrgid_r(gid_t gid, struct group *restrict grp, char *restrict buf, size_t buflen, struct group **restrict result) {
    return scopelibc_getgrgid_r(gid, grp, buf, buflen, result);
}

int
scope_getpwuid_r(uid_t uid, struct passwd *pwd, char *buf, size_t buflen, struct passwd **result) {
    return scopelibc_getpwuid_r(uid, pwd, buf, buflen, result);
}

pid_t
scope_getpid(void) {
    return scopelibc_getpid();
}

pid_t
scope_getppid(void) {
    return scopelibc_getppid();
}

uid_t
scope_getuid(void) {
    return scopelibc_getuid();
}

gid_t
scope_getgid(void) {
    return scopelibc_getgid();
}

void*
scope_dlopen(const char *filename, int flags) {
    return scopelibc_dlopen(filename, flags);
}

int
scope_dlclose(void *handle) {
    return scopelibc_dlclose(handle);
}

void*
scope_dlsym(void *restrict handle, const char *restrict symbol) {
    return scopelibc_dlsym(handle, symbol);
}

long
scope_ptrace(int request, pid_t pid, void *addr, void *data) {
    return scopelibc_ptrace(request, pid, addr, data);
}

pid_t
scope_waitpid(pid_t pid, int *status, int options) {
    return scopelibc_waitpid(pid, status, options);
}

char*
scope_getenv(const char *name) {
    return scopelibc_getenv(name);
}

int
scope_setenv(const char *name, const char *value, int overwrite) {
    return scopelibc_setenv(name, value, overwrite);
}

struct lconv *
scope_localeconv(void) {
    return scopelibc_localeconv();
}

int
scope_shm_open(const char *name, int oflag, mode_t mode) {
    return scopelibc_shm_open(name, oflag, mode);
}

int
scope_shm_unlink(const char *name) {
    return scopelibc_shm_unlink(name);
}

long
scope_sysconf(int name) {
    return scopelibc_sysconf(name);
}

int
scope_mkstemp(char *template) {
    return scopelibc_mkstemp(template);
}

int
scope_clock_gettime(clockid_t clk_id, struct timespec *tp) {
    return scopelibc_clock_gettime(clk_id, tp);
}

int
scope_getpagesize(void) {
    return scopelibc_getpagesize();
}

int
scope_uname(struct utsname *buf) {
    return scopelibc_uname(buf);
}

int
scope_arch_prctl(int code, unsigned long addr) {
#if defined(__x86_64)
    return scopelibc_arch_prctl(code, addr);
#else
    //arch_prctl is supported only on Linux/x86-64 for 64-bit
    return -1;
#endif
}

int
scope_getrusage(int who, struct rusage *usage) {
    return scopelibc_getrusage(who, usage);
}

int
scope_atexit(void (*atexit_func)(void)) {
    return scopelibc_atexit(atexit_func);
}

int
scope_tcsetattr(int fildes, int optional_actions, const struct termios *termios_p) {
    return scopelibc_tcsetattr(fildes, optional_actions, termios_p);
}

int
scope_tcgetattr(int fildes, struct termios *termios_p) {
    return scopelibc_tcgetattr(fildes, termios_p);
}

void*
scope_shmat(int shmid, const void *shmaddr, int shmflg) {
    return scopelibc_shmat(shmid, shmaddr, shmflg);
}

int
scope_shmdt(const void *shmaddr) {
    return scopelibc_shmdt(shmaddr);
}

int
scope_shmget(key_t key, size_t size, int shmflg) {
    return scopelibc_shmget(key, size, shmflg);
}

int
scope_sched_getcpu(void) {
    return scopelibc_sched_getcpu();
}

int
scope___snprintf_chk(char *str, size_t maxlen, int flag, size_t slen, const char * format, ...)
{
    int ret;
    va_list ap;
    va_start(ap, format);
    ret = scope_vsnprintf(str, slen, format, ap);
    va_end(ap);
    return ret;
}

int
scope___vfprintf_chk(FILE *fp, int flag, const char *format, va_list ap)
{
    return scope_vfprintf(fp, format, ap);
}

int
scope___vsnprintf_chk(char *s, size_t maxlen, int flag, size_t slen, const char *format, va_list args)
{
    return scope_vsnprintf(s, slen, format, args);
}

int
scope__iso99_sscanf(const char *restrict s, const char *restrict fmt, ...)
{
    int ret;
    va_list ap;
    va_start(ap, fmt);
    ret = scope_vsscanf(s, fmt, ap);
    va_end(ap);
    return ret;
}

unsigned short **
scope___ctype_b_loc (void)
{
    return scopelibc___ctype_b_loc();
}

int32_t **
scope___ctype_tolower_loc(void)
{
    return scopelibc___ctype_tolower_loc();
}
