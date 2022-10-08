#ifndef __SCOPE_STDLIB_H__
#define __SCOPE_STDLIB_H__

#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <dirent.h>
#include <grp.h>
#include <link.h>
#include <locale.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

extern int  scopelibc_fcntl(int, int, ... /* arg */);
extern int  scopelibc_open(const char *, int, ...);
extern long scopelibc_syscall(long, ...);
extern int  scopelibc_printf(const char *, ...);
extern int  scopelibc_dprintf(int, const char *, ...);
extern int  scopelibc_fprintf(FILE *, const char *, ...);
extern int  scopelibc_snprintf(char *, size_t, const char *, ...);
extern int  scopelibc_sscanf(const char *, const char *, ...);
extern int  scopelibc_fscanf(FILE *, const char *, ...);
extern int  scopelibc_sprintf(char *, const char *, ...);
extern int  scopelibc_asprintf(char **, const char *, ...);
extern int  scopelibc_errno_val;
extern FILE scopelibc___stdin_FILE;
extern FILE scopelibc___stdout_FILE;
extern FILE scopelibc___stderr_FILE;
extern unsigned short ** scopelibc___ctype_b_loc(void);
extern int32_t ** scopelibc___ctype_tolower_loc(void);

#define scope_fcntl    scopelibc_fcntl
#define scope_open     scopelibc_open
#define scope_syscall  scopelibc_syscall
#define scope_printf   scopelibc_printf
#define scope_dprintf  scopelibc_dprintf
#define scope_fprintf  scopelibc_fprintf
#define scope_snprintf scopelibc_snprintf
#define scope_sscanf   scopelibc_sscanf
#define scope_fscanf   scopelibc_fscanf
#define scope_sprintf  scopelibc_sprintf
#define scope_asprintf scopelibc_asprintf
#define scope_errno    scopelibc_errno_val
#define scope_stdin    (&scopelibc___stdin_FILE)
#define scope_stdout   (&scopelibc___stdout_FILE)
#define scope_stderr   (&scopelibc___stderr_FILE)

/*
 * Notes on the use of errno.
 * There are 2 errno values; errno and scope_errno.
 *
 * errno is set by the application. It should be
 * used, for the most part, by interposed functions where
 * results of application behavior needs to be checked. It
 * should only ever be set by libscope in specific cases.
 *
 * scope_errno is used by the internal libc.
 * This value is not thread specific, thread safe, as we avoid the
 * use of the %fs register and TLS behavior with the internal libc.
 *
 * Use scope_errno only for functions called from the periodic
 * thread, during libscope constructor, from ldscope or from ldscopedyn.
 *
 * Other functions, primarily those called from interposed functions, can
 * not safely reference scope_errno.
 */
// Other
extern void scopeSetGoAppStateStatic(int);
extern int scopeGetGoAppStateStatic(void);

// Custom operations

void  scope_init_vdso_ehdr(void);
void  scope_op_before_fork(void);
void  scope_op_after_fork(int);

// Memory management handling operations
void* scope_memalign(size_t, size_t);
void* scope_malloc(size_t);
void* scope_calloc(size_t, size_t);
void* scope_realloc(void *, size_t);
void  scope_free(void *);
void* scope_mmap(void *, size_t, int, int, int, off_t);
int   scope_munmap(void *, size_t);
FILE* scope_open_memstream(char **, size_t *);
void* scope_memset(void *, int, size_t);
void* scope_memmove(void *, const void *, size_t);
int   scope_memcmp(const void *, const void *, size_t);
int   scope_mprotect(void *, size_t, int);
void* scope_memcpy(void *, const void *, size_t);
int   scope_mlock(const void *, size_t);
int   scope_msync(void *, size_t, int);
int   scope_mincore(void *, size_t, unsigned char *);

// File handling operations
FILE*          scope_fopen(const char *, const char *);
int            scope_fclose(FILE *);
FILE*          scope_fdopen(int, const char *);
int            scope_close(int);
ssize_t        scope_read(int, void *, size_t);
size_t         scope_fread(void *, size_t, size_t, FILE *);
ssize_t        scope_write(int, const void *, size_t);
size_t         scope_fwrite(const void *, size_t, size_t, FILE *);
char *         scope_fgets(char *, int, FILE *);
ssize_t        scope_getline(char **, size_t *, FILE *);
int            scope_puts(const char *);
int            scope_setvbuf(FILE *, char *, int, size_t);
int            scope_fflush(FILE *);
char*          scope_dirname(char *);
DIR*           scope_opendir(const char *);
struct dirent* scope_readdir(DIR *);
int            scope_closedir(DIR *);
int            scope_access(const char *, int);
FILE*          scope_fmemopen(void *, size_t, const char *);
long           scope_ftell(FILE *);
int            scope_fseek(FILE *, long, int);
off_t          scope_lseek(int, off_t, int);
int            scope_unlink(const char *);
int            scope_dup2(int, int);
char*          scope_basename(char *);
int            scope_stat(const char *, struct stat *);
int            scope_chmod(const char *, mode_t);
int            scope_fchmod(int, mode_t);
int            scope_feof(FILE *);
int            scope_fileno(FILE *);
int            scope_flock(int, int);
int            scope_fstat(int, struct stat *);
int            scope_mkdir(const char *, mode_t);
int            scope_chdir(const char *);
int            scope_rmdir(const char *);
char*          scope_get_current_dir_name(void);
char*          scope_getcwd(char *, size_t);
int            scope_lstat(const char *, struct stat *);
int            scope_rename(const char *, const char *);
int            scope_remove(const char *);
int            scope_pipe2(int [2], int);
void           scope_setbuf(FILE *, char *);

// String handling operations
char*              scope_realpath(const char *, char *);
ssize_t            scope_readlink(const char *, char *, size_t);
char*              scope_strdup(const char *);
int                scope_vasprintf(char **, const char *, va_list);
size_t             scope_strftime(char *, size_t, const char *, const struct tm *);
size_t             scope_strlen(const char *);
size_t             scope_strnlen(const char *, size_t);
char *             scope_strerror(int);
int                scope_strerror_r(int, char *, size_t);
double             scope_strtod(const char *, char **);
long               scope_strtol(const char *, char **, int);
long long          scope_strtoll(const char *, char **, int);
unsigned long      scope_strtoul(const char *, char **, int);
unsigned long long scope_strtoull(const char *, char **, int);
char*              scope_strchr(const char *, int);
char*              scope_strrchr(const char *, int);
char*              scope_strstr(const char *, const char *);
int                scope_vsnprintf(char *, size_t, const char *, va_list);
int                scope_vfprintf(FILE *, const char *, va_list);
int                scope_vprintf(const char *, va_list);
int                scope_strcmp(const char *, const char *);
int                scope_strncmp(const char *, const char *, size_t);
int                scope_strcasecmp(const char *, const char *);
char*              scope_strcpy(char *, const char *);
char*              scope_strncpy(char *, const char *, size_t);
char*              scope_stpcpy(char *, const char *);
char*              scope_stpcpy(char *, const char *);
char*              scope_stpncpy(char *, const char *, size_t);
size_t             scope_strcspn(const char *, const char *);
char*              scope_strcat(char *, const char *);
char*              scope_strncat(char *, const char *, size_t);
char*              scope_strpbrk(const char *, const char *);
char*              scope_strcasestr(const char *, const char *);
char*              scope_strtok(char *, const char *);
char*              scope_strtok_r(char *, const char *, char **);
const char*        scope_gai_strerror(int);

// Network handling operations
int             scope_gethostname(char *, size_t);
int             scope_getsockname(int, struct sockaddr *, socklen_t *);
int             scope_getsockopt(int, int, int, void *, socklen_t *);
int             scope_setsockopt(int, int, int, const void *, socklen_t);
int             scope_socket(int, int, int);
int             scope_accept(int, struct sockaddr *, socklen_t *);
int             scope_bind(int, const struct sockaddr *, socklen_t);
int             scope_connect(int, const struct sockaddr *, socklen_t);
int             scope_listen(int, int);
void            scope_rewind(FILE *);
ssize_t         scope_send(int, const void *, size_t, int);
ssize_t         scope_sendmsg(int, const struct msghdr *, int);
ssize_t         scope_recv(int, void *, size_t, int);
ssize_t         scope_recvmsg(int, struct msghdr *, int);
ssize_t         scope_recvfrom(int, void *, size_t, int, struct sockaddr *, socklen_t *);
int             scope_shutdown(int, int);
int             scope_poll(struct pollfd *, nfds_t, int);
int             scope_select(int, fd_set *, fd_set *, fd_set *, struct timeval *);
int             scope_getaddrinfo(const char *, const char *, const struct addrinfo *, struct addrinfo **);
int             scope_copyaddrinfo(struct sockaddr *, socklen_t, struct addrinfo **);
void            scope_freeaddrinfo(struct addrinfo *);
int             scope_getnameinfo(const struct sockaddr *, socklen_t, char *, socklen_t, char *, socklen_t, int);
int             scope_getpeername(int, struct sockaddr *, socklen_t *);
struct hostent* scope_gethostbyname(const char *);
const char*     scope_inet_ntop(int, const void *, char *, socklen_t);
uint16_t        scope_ntohs(uint16_t);
uint16_t        scope_htons(uint16_t);

// Misc
int           scope_atoi(const char *);
int           scope_isspace(int);
int           scope_isprint(int);
int           scope_isdigit(int);
void          scope_perror(const char *);
int           scope_gettimeofday(struct timeval *, struct timezone *);
int           scope_timer_create(clockid_t, struct sigevent *, timer_t *);
int           scope_timer_settime(timer_t, int, const struct itimerspec *, struct itimerspec *);
int           scope_timer_delete(timer_t);
time_t        scope_time(time_t *);
struct tm*    scope_localtime_r(const time_t *, struct tm *);
struct tm*    scope_gmtime_r(const time_t *, struct tm *);
unsigned int  scope_sleep(unsigned int);
int           scope_usleep(useconds_t);
int           scope_nanosleep(const struct timespec *, struct timespec *);
int           scope_sigaction(int, const struct sigaction *, struct sigaction *);
int           scope_sigemptyset(sigset_t *);
int           scope_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int           scope_pthread_barrier_init(pthread_barrier_t *, const pthread_barrierattr_t *, unsigned);
int           scope_pthread_barrier_destroy(pthread_barrier_t *);
int           scope_pthread_barrier_wait(pthread_barrier_t *);;
int           scope_ns_initparse(const unsigned char *, int, ns_msg *);
int           scope_ns_parserr(ns_msg *, ns_sect, int, ns_rr *);
int           scope_getgrgid_r(gid_t, struct group *, char *, size_t, struct group **);
int           scope_getpwuid_r(uid_t, struct passwd *, char *, size_t, struct passwd **);
pid_t         scope_getpid(void);
pid_t         scope_getppid(void);
uid_t         scope_getuid(void);
uid_t         scope_geteuid(void);
gid_t         scope_getegid(void);
int           scope_seteuid(uid_t);
int           scope_setegid(gid_t);
gid_t         scope_getgid(void);
void*         scope_dlopen(const char *, int);
void*         scope_dlsym(void *, const char *);
int           scope_dlclose(void *);
long          scope_ptrace(int, pid_t, void *, void *);
pid_t         scope_waitpid(pid_t, int *, int);
char*         scope_getenv(const char *);
int           scope_setenv(const char *, const char *, int);
struct lconv* scope_localeconv(void);
int           scope_shm_open(const char *, int, mode_t);
int           scope_shm_unlink(const char *);
long          scope_sysconf(int);
int           scope_mkstemp(char *);
int           scope_clock_gettime(clockid_t, struct timespec *);
int           scope_getpagesize(void);
int           scope_uname(struct utsname *);
int           scope_arch_prctl(int, unsigned long);
int           scope_getrusage(int, struct rusage *);
int           scope_atexit(void (*)(void));
int           scope_tcsetattr(int, int, const struct termios *);
int           scope_tcgetattr(int, struct termios *);
void*         scope_shmat(int, const void *, int);
int           scope_shmdt(const void *);
int           scope_shmget(key_t, size_t, int);
int           scope_sched_getcpu(void);
int           scope_ftruncate(int, off_t);
int           scope_rand(void);
void          scope_srand(unsigned int);
int           scope_setns(int, int);
int           scope_chown(const char *, uid_t, gid_t);
int           scope_fchown(int, uid_t, gid_t);


#endif // __SCOPE_STDLIB_H__
