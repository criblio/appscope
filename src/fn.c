#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <link.h>

#include "dbg.h"
#include "fn.h"

#define GETADDR(val, sym)                              \
    ares.in_symbol = sym;                              \
    if (dl_iterate_phdr(getAddr, &ares)) {             \
        val = ares.out_addr;                           \
    } else if ((val = dlsym(RTLD_NEXT, sym))) {        \
        DBG(NULL);                                     \
    } else if ((val = dlsym(RTLD_DEFAULT, sym))) {     \
        DBG(NULL);                                     \
    } else {                                           \
        val = NULL;                                    \
    }

interposed_funcs_t g_fn;

typedef struct
{
    char *in_symbol;
    void *out_addr;
} addresult_t;

static int
getAddr(struct dl_phdr_info *info, size_t size, void *data)
{
    addresult_t *ares = (addresult_t *)data;

    ares->out_addr = NULL;

    if (strstr(info->dlpi_name, "librt.so") == NULL) return 0;

    // can we find the symbol in this object
    void *handle = g_fn.dlopen(info->dlpi_name, RTLD_NOW);
    if (!handle) return 0;

    void *addr = dlsym(handle, ares->in_symbol);
    dlclose(handle);

    // if we don't find addr, keep going
    if (!addr) {
        return 0;
    }

    // We found the symbol in a lib that is not libscope
    ares->out_addr = addr;
    return 1;
}

void
initFn(void)
{
    addresult_t ares;

    g_fn.dlopen = dlsym(RTLD_NEXT, "dlopen");
    if (!g_fn.dlopen) g_fn.dlopen = dlsym(RTLD_DEFAULT, "dlopen");

    GETADDR(g_fn.vsyslog, "vsyslog");
    GETADDR(g_fn.fork, "fork");
    GETADDR(g_fn.open, "open");
    GETADDR(g_fn.openat, "openat");
    GETADDR(g_fn.fopen, "fopen");
    GETADDR(g_fn.freopen, "freopen");
    GETADDR(g_fn.creat, "creat");
    GETADDR(g_fn.close, "close");
    GETADDR(g_fn.fclose, "fclose");
    GETADDR(g_fn.fcloseall, "fcloseall");
    GETADDR(g_fn.read, "read");
    GETADDR(g_fn.pread, "pread");
    GETADDR(g_fn.readv, "readv");
    GETADDR(g_fn.fread, "fread");
    GETADDR(g_fn.__fread_chk, "__fread_chk");
    GETADDR(g_fn.fread_unlocked, "fread_unlocked");
    GETADDR(g_fn.fgets, "fgets");
    GETADDR(g_fn.__fgets_chk, "__fgets_chk");
    GETADDR(g_fn.fgets_unlocked, "fgets_unlocked");
    GETADDR(g_fn.fgetws, "fgetws");
    GETADDR(g_fn.__fgetws_chk, "__fgetws_chk");
    GETADDR(g_fn.fgetwc, "fgetwc");
    GETADDR(g_fn.fgetc, "fgetc");
    GETADDR(g_fn.fscanf, "fscanf");
    GETADDR(g_fn.fputc, "fputc");
    GETADDR(g_fn.fputc_unlocked, "fputc_unlocked");
    GETADDR(g_fn.fputwc, "fputwc");
    GETADDR(g_fn.putwc, "putwc");
    GETADDR(g_fn.getline, "getline");
    GETADDR(g_fn.__getdelim, "__getdelim");
    GETADDR(g_fn.write, "write");
    GETADDR(g_fn.pwrite, "pwrite");
    GETADDR(g_fn.writev, "writev");
    GETADDR(g_fn.fwrite, "fwrite");
    GETADDR(g_fn.sendfile, "sendfile");
    GETADDR(g_fn.putchar, "putchar");
    GETADDR(g_fn.puts, "puts");
    GETADDR(g_fn.fputs, "fputs");
    GETADDR(g_fn.fputs_unlocked, "fputs_unlocked");
    GETADDR(g_fn.fputws, "fputws");
    GETADDR(g_fn.lseek, "lseek");
    GETADDR(g_fn.fseek, "fseek");
    GETADDR(g_fn.fseeko, "fseeko");
    GETADDR(g_fn.ftell, "ftell");
    GETADDR(g_fn.ftello, "ftello");
    GETADDR(g_fn.ftello, "ftello");
    GETADDR(g_fn.fgetpos, "fgetpos");
    GETADDR(g_fn.fsetpos, "fsetpos");
    GETADDR(g_fn.fsetpos64, "fsetpos64");
    GETADDR(g_fn.stat, "stat");
    GETADDR(g_fn.lstat, "lstat");
    GETADDR(g_fn.fstat, "fstat");
    GETADDR(g_fn.fstatat, "fstatat");
    GETADDR(g_fn.statfs, "statfs");
    GETADDR(g_fn.fstatfs, "fstatfs");
    GETADDR(g_fn.statvfs, "statvfs");
    GETADDR(g_fn.fstatvfs, "fstatvfs");
    GETADDR(g_fn.access, "access");
    GETADDR(g_fn.faccessat, "faccessat");
    GETADDR(g_fn.rewind, "rewind");
    GETADDR(g_fn.fcntl, "fcntl");
    GETADDR(g_fn.fcntl64, "fcntl64");
    GETADDR(g_fn.dup, "dup");
    GETADDR(g_fn.dup2, "dup2");
    GETADDR(g_fn.dup3, "dup3");
    GETADDR(g_fn.socket, "socket");
    GETADDR(g_fn.shutdown, "shutdown");
    GETADDR(g_fn.listen, "listen");
    GETADDR(g_fn.accept, "accept");
    GETADDR(g_fn.accept4, "accept4");
    GETADDR(g_fn.bind, "bind");
    GETADDR(g_fn.connect, "connect");
    GETADDR(g_fn.send, "send");
    GETADDR(g_fn.sendto, "sendto");
    GETADDR(g_fn.sendmsg, "sendmsg");
    GETADDR(g_fn.recv, "recv");
    GETADDR(g_fn.recvfrom, "recvfrom");
    GETADDR(g_fn.recvmsg, "recvmsg");
    GETADDR(g_fn.gethostbyname, "gethostbyname");
    GETADDR(g_fn.gethostbyname2, "gethostbyname2");
    GETADDR(g_fn.getaddrinfo, "getaddrinfo");
    GETADDR(g_fn.sigaction, "sigaction");
    GETADDR(g_fn.execve, "execve");
    GETADDR(g_fn.poll, "poll");
    GETADDR(g_fn.select, "select");
    GETADDR(g_fn.ns_initparse, "ns_initparse");
    GETADDR(g_fn.ns_parserr, "ns_parserr");
    GETADDR(g_fn.__stdout_write, "__stdout_write");
    GETADDR(g_fn.__stderr_write, "__stderr_write");
    GETADDR(g_fn.__fprintf_chk, "__fprintf_chk");
    GETADDR(g_fn.__memset_chk, "__memset_chk");
    GETADDR(g_fn.__memcpy_chk, "__memcpy_chk");
    GETADDR(g_fn.__sprintf_chk, "__sprintf_chk");
    GETADDR(g_fn.__fdelt_chk, "__fdelt_chk");
    GETADDR(g_fn.open64, "open64");
    GETADDR(g_fn.openat64, "openat64");
    GETADDR(g_fn.__open_2, "__open_2");
    GETADDR(g_fn.__open64_2, "__open64_2");
    GETADDR(g_fn.__openat_2, "__openat_2");
    GETADDR(g_fn.fopen64, "fopen64");
    GETADDR(g_fn.freopen64, "freopen64");
    GETADDR(g_fn.creat64, "creat64");
    GETADDR(g_fn.pread64, "pread64");
    GETADDR(g_fn.preadv, "preadv");
    GETADDR(g_fn.preadv2, "preadv2");
    GETADDR(g_fn.preadv64v2, "preadv64v2");
    GETADDR(g_fn.__pread_chk, "__pread_chk");
    GETADDR(g_fn.__read_chk, "__read_chk");
    GETADDR(g_fn.__fread_unlocked_chk, "__fread_unlocked_chk");
    GETADDR(g_fn.pwrite64, "pwrite64");
    GETADDR(g_fn.pwritev, "pwritev");
    GETADDR(g_fn.pwritev64, "pwritev64");
    GETADDR(g_fn.pwritev2, "pwritev2");
    GETADDR(g_fn.pwritev64v2, "pwritev64v2");
    GETADDR(g_fn.fwrite_unlocked, "fwrite_unlocked");
    GETADDR(g_fn.sendfile64, "sendfile64");
    GETADDR(g_fn.lseek64, "lseek64");
    GETADDR(g_fn.fseeko64, "fseeko64");
    GETADDR(g_fn.ftello64, "ftello64");
    GETADDR(g_fn.statfs64, "statfs64");
    GETADDR(g_fn.fstatfs64, "fstatfs64");
    GETADDR(g_fn.fstatvfs64, "fstatvfs64");
    GETADDR(g_fn.fgetpos64, "fgetpos64");
    GETADDR(g_fn.statvfs64, "statvfs64");
    GETADDR(g_fn.__lxstat, "__lxstat");
    GETADDR(g_fn.__lxstat64, "__lxstat64");
    GETADDR(g_fn.__xstat, "__xstat");
    GETADDR(g_fn.__xstat64, "__xstat64");
    GETADDR(g_fn.__fxstat, "__fxstat");
    GETADDR(g_fn.__fxstat64, "__fxstat64");
    GETADDR(g_fn.__fxstatat, "__fxstatat");
    GETADDR(g_fn.__fxstatat64, "__fxstatat64");
    GETADDR(g_fn.gethostbyname_r, "gethostbyname_r");
    GETADDR(g_fn.gethostbyname2_r, "gethostbyname2_r");
    GETADDR(g_fn.syscall, "syscall");
    GETADDR(g_fn.prctl, "prctl");
    GETADDR(g_fn._exit, "_exit");
    GETADDR(g_fn.SSL_read, "SSL_read");
    GETADDR(g_fn.SSL_write, "SSL_write");
    GETADDR(g_fn.SSL_get_fd, "SSL_get_fd");
    GETADDR(g_fn.gnutls_record_recv, "gnutls_record_recv");
    GETADDR(g_fn.gnutls_record_send, "gnutls_record_send");
    GETADDR(g_fn.gnutls_record_recv_early_data, "gnutls_record_recv_early_data");
    GETADDR(g_fn.gnutls_record_recv_packet, "gnutls_record_recv_packet");
    GETADDR(g_fn.gnutls_record_recv_seq, "gnutls_record_recv_seq");
    GETADDR(g_fn.gnutls_record_send2, "gnutls_record_send2");
    GETADDR(g_fn.gnutls_record_send_early_data, "gnutls_record_send_early_data");
    GETADDR(g_fn.gnutls_record_send_range, "gnutls_record_send_range");
    GETADDR(g_fn.gnutls_transport_get_ptr, "gnutls_transport_get_ptr");
    GETADDR(g_fn.SSL_ImportFD, "SSL_ImportFD");
    //GETADDR(g_fn.dlopen, "dlopen");
    GETADDR(g_fn.PR_FileDesc2NativeHandle, "PR_FileDesc2NativeHandle");
    GETADDR(g_fn.PR_SetError, "PR_SetError");
    GETADDR(g_fn.__overflow, "__overflow");
    GETADDR(g_fn.sendmmsg, "sendmmsg");
    GETADDR(g_fn.recvmmsg, "recvmmsg");
    GETADDR(g_fn.pthread_create, "pthread_create");
    GETADDR(g_fn.getentropy, "getentropy");
    GETADDR(g_fn.__ctype_init, "__ctype_init");
    GETADDR(g_fn.__register_atfork, "__register_atfork");
    GETADDR(g_fn.epoll_wait, "epoll_wait");
    GETADDR(g_fn.nanosleep, "nanosleep");
    GETADDR(g_fn.sigsuspend, "sigsuspend");
    GETADDR(g_fn.pause, "pause");
    GETADDR(g_fn.sigwaitinfo, "sigwaitinfo");
    GETADDR(g_fn.sigtimedwait, "sigtimedwait");
    GETADDR(g_fn.epoll_pwait, "epoll_pwait");
    GETADDR(g_fn.ppoll, "ppoll");
    GETADDR(g_fn.pselect, "pselect");
    GETADDR(g_fn.msgsnd, "msgsnd");
    GETADDR(g_fn.msgrcv, "msgrcv");
    GETADDR(g_fn.semop, "semop");
    GETADDR(g_fn.semtimedop, "semtimedop");
    GETADDR(g_fn.clock_nanosleep, "clock_nanosleep");
    GETADDR(g_fn.usleep, "usleep");
    GETADDR(g_fn.io_getevents, "io_getevents");
    GETADDR(g_fn.setenv, "setenv");
#ifdef __STATX__
    GETADDR(g_fn.statx, "statx");
#endif
#if 0
    g_fn.vsyslog = dlsym(handle, "vsyslog");
    g_fn.fork = dlsym(handle, "fork");
    g_fn.open = dlsym(handle, "open");
    g_fn.openat = dlsym(handle, "openat");
    g_fn.fopen = dlsym(handle, "fopen");
    g_fn.freopen = dlsym(handle, "freopen");
    g_fn.creat = dlsym(handle, "creat");
    g_fn.close = dlsym(handle, "close");
    g_fn.fclose = dlsym(handle, "fclose");
    g_fn.fcloseall = dlsym(handle, "fcloseall");
    g_fn.read = dlsym(handle, "read");
    g_fn.pread = dlsym(handle, "pread");
    g_fn.readv = dlsym(handle, "readv");
    g_fn.fread = dlsym(handle, "fread");
    g_fn.__fread_chk = dlsym(handle, "__fread_chk");
    g_fn.fread_unlocked = dlsym(handle, "fread_unlocked");
    g_fn.fgets = dlsym(handle, "fgets");
    g_fn.__fgets_chk = dlsym(handle, "__fgets_chk");
    g_fn.fgets_unlocked = dlsym(handle, "fgets_unlocked");
    g_fn.fgetws = dlsym(handle, "fgetws");
    g_fn.__fgetws_chk = dlsym(handle, "__fgetws_chk");
    g_fn.fgetwc = dlsym(handle, "fgetwc");
    g_fn.fgetc = dlsym(handle, "fgetc");
    g_fn.fscanf = dlsym(handle, "fscanf");
    g_fn.fputc = dlsym(handle, "fputc");
    g_fn.fputc_unlocked = dlsym(handle, "fputc_unlocked");
    g_fn.fputwc = dlsym(handle, "fputwc");
    g_fn.putwc = dlsym(handle, "putwc");
    g_fn.getline = dlsym(handle, "getline");
    g_fn.getdelim = dlsym(handle, "getdelim");
    g_fn.__getdelim = dlsym(handle, "__getdelim");
    g_fn.write = dlsym(handle, "write");
    g_fn.pwrite = dlsym(handle, "pwrite");
    g_fn.writev = dlsym(handle, "writev");
    g_fn.fwrite = dlsym(handle, "fwrite");
    g_fn.sendfile = dlsym(handle, "sendfile");
    g_fn.putchar = dlsym(handle, "putchar");
    g_fn.puts = dlsym(handle, "puts");
    g_fn.fputs = dlsym(handle, "fputs");
    g_fn.fputs_unlocked = dlsym(handle, "fputs_unlocked");
    g_fn.fputws = dlsym(handle, "fputws");
    g_fn.lseek = dlsym(handle, "lseek");
    g_fn.fseek = dlsym(handle, "fseek");
    g_fn.fseeko = dlsym(handle, "fseeko");
    g_fn.ftell = dlsym(handle, "ftell");
    g_fn.ftello = dlsym(handle, "ftello");
    g_fn.fgetpos = dlsym(handle, "fgetpos");
    g_fn.fsetpos = dlsym(handle, "fsetpos");
    g_fn.fsetpos64 = dlsym(handle, "fsetpos64");
    g_fn.stat = dlsym(handle, "stat");
    g_fn.lstat = dlsym(handle, "lstat");
    g_fn.fstat = dlsym(handle, "fstat");
    g_fn.fstatat = dlsym(handle, "fstatat");
    g_fn.statfs = dlsym(handle, "statfs");
    g_fn.fstatfs = dlsym(handle, "fstatfs");
    g_fn.statvfs = dlsym(handle, "statvfs");
    g_fn.fstatvfs = dlsym(handle, "fstatvfs");
    g_fn.access = dlsym(handle, "access");
    g_fn.faccessat = dlsym(handle, "faccessat");
    g_fn.rewind = dlsym(handle, "rewind");
    g_fn.fcntl = dlsym(handle, "fcntl");
    g_fn.fcntl64 = dlsym(handle, "fcntl64");
    g_fn.dup = dlsym(handle, "dup");
    g_fn.dup2 = dlsym(handle, "dup2");
    g_fn.dup3 = dlsym(handle, "dup3");
    g_fn.socket = dlsym(handle, "socket");
    g_fn.shutdown = dlsym(handle, "shutdown");
    g_fn.listen = dlsym(handle, "listen");
    g_fn.accept = dlsym(handle, "accept");
    g_fn.accept4 = dlsym(handle, "accept4");
    g_fn.bind = dlsym(handle, "bind");
    g_fn.connect = dlsym(handle, "connect");
    g_fn.send = dlsym(handle, "send");
    g_fn.sendto = dlsym(handle, "sendto");
    g_fn.sendmsg = dlsym(handle, "sendmsg");
    g_fn.recv = dlsym(handle, "recv");
    g_fn.recvfrom = dlsym(handle, "recvfrom");
    g_fn.recvmsg = dlsym(handle, "recvmsg");
    g_fn.gethostbyname = dlsym(handle, "gethostbyname");
    g_fn.gethostbyname2 = dlsym(handle, "gethostbyname2");
    g_fn.getaddrinfo = dlsym(handle, "getaddrinfo");
    g_fn.sigaction = dlsym(handle, "sigaction");
    g_fn.execve = dlsym(handle, "execve");
    g_fn.poll = dlsym(handle, "poll");
    g_fn.select = dlsym(handle, "select");
    g_fn.ns_initparse = dlsym(handle, "ns_initparse");
    g_fn.ns_parserr = dlsym(handle, "ns_parserr");
    g_fn.__stdout_write = dlsym(handle, "__stdio_write");
    g_fn.__stderr_write = dlsym(handle, "__stdio_write");
    // added for openssl on libmusl
    g_fn.__fprintf_chk = dlsym(handle, "__fprintf_chk");
    g_fn.__memset_chk = dlsym(handle, "__memset_chk");
    g_fn.__memcpy_chk = dlsym(handle, "__memcpy_chk");
    g_fn.__sprintf_chk = dlsym(handle, "__sprintf_chk");
    g_fn.__fdelt_chk = dlsym(handle, "__fdelt_chk");
#ifdef __MACOS__
    g_fn.close$NOCANCEL = dlsym(handle, "close$NOCANCEL");
    g_fn.close_nocancel = dlsym(handle, "close_nocancel");
    g_fn.guarded_close_np = dlsym(handle, "guarded_close_np");
    g_fn.accept$NOCANCEL = dlsym(handle, "accept$NOCANCEL");
    g_fn.__sendto_nocancel = dlsym(handle, "__sendto_nocancel");
    g_fn.DNSServiceQueryRecord = dlsym(handle, "DNSServiceQueryRecord");
#endif // __MACOS__

#ifdef __LINUX__
    g_fn.open64 = dlsym(handle, "open64");
    g_fn.openat64 = dlsym(handle, "openat64");
    g_fn.__open_2 = dlsym(handle, "__open_2");
    g_fn.__open64_2 = dlsym(handle, "__open64_2");
    g_fn.__openat_2 = dlsym(handle, "__openat_2");
    g_fn.fopen64 = dlsym(handle, "fopen64");
    g_fn.freopen64 = dlsym(handle, "freopen64");
    g_fn.creat64 = dlsym(handle, "creat64");
    g_fn.pread64 = dlsym(handle, "pread64");
    g_fn.preadv = dlsym(handle, "preadv");
    g_fn.preadv2 = dlsym(handle, "preadv2");
    g_fn.preadv64v2 = dlsym(handle, "preadv64v2");
    g_fn.__pread_chk = dlsym(handle, "__pread_chk");
    g_fn.__read_chk = dlsym(handle, "__read_chk");
    g_fn.__fread_unlocked_chk = dlsym(handle, "__fread_unlocked_chk");
    g_fn.pwrite64 = dlsym(handle, "pwrite64");
    g_fn.pwritev = dlsym(handle, "pwritev");
    g_fn.pwritev64 = dlsym(handle, "pwritev64");
    g_fn.pwritev2 = dlsym(handle, "pwritev2");
    g_fn.pwritev64v2 = dlsym(handle, "pwritev64v2");
    g_fn.fwrite_unlocked = dlsym(handle, "fwrite_unlocked");
    g_fn.sendfile64 = dlsym(handle, "sendfile64");
    g_fn.lseek64 = dlsym(handle, "lseek64");
    g_fn.fseeko64 = dlsym(handle, "fseeko64");
    g_fn.ftello64 = dlsym(handle, "ftello64");
    g_fn.statfs64 = dlsym(handle, "statfs64");
    g_fn.fstatfs64 = dlsym(handle, "fstatfs64");
    g_fn.fstatvfs64 = dlsym(handle, "fstatvfs64");
    g_fn.fgetpos64 = dlsym(handle, "fgetpos64");
    g_fn.statvfs64 = dlsym(handle, "statvfs64");
    g_fn.__lxstat = dlsym(handle, "__lxstat");
    g_fn.__lxstat64 = dlsym(handle, "__lxstat64");
    g_fn.__xstat = dlsym(handle, "__xstat");
    g_fn.__xstat64 = dlsym(handle, "__xstat64");
    g_fn.__fxstat = dlsym(handle, "__fxstat");
    g_fn.__fxstat64 = dlsym(handle, "__fxstat64");
    g_fn.__fxstatat = dlsym(handle, "__fxstatat");
    g_fn.__fxstatat64 = dlsym(handle, "__fxstatat64");
    g_fn.gethostbyname_r = dlsym(handle, "gethostbyname_r");
    g_fn.gethostbyname2_r = dlsym(handle, "gethostbyname2_r");
    g_fn.syscall = dlsym(handle, "syscall");
    g_fn.prctl = dlsym(handle, "prctl");
    g_fn._exit = dlsym(handle, "_exit");
    g_fn.SSL_read = dlsym(handle, "SSL_read");
    g_fn.SSL_write = dlsym(handle, "SSL_write");
    g_fn.SSL_get_fd = dlsym(handle, "SSL_get_fd");
    g_fn.gnutls_record_recv = dlsym(handle, "gnutls_record_recv");
    g_fn.gnutls_record_send = dlsym(handle, "gnutls_record_send");
    g_fn.gnutls_record_recv_early_data = dlsym(handle, "gnutls_record_recv_early_data");
    g_fn.gnutls_record_recv_packet = dlsym(handle, "gnutls_record_recv_packet");
    g_fn.gnutls_record_recv_seq = dlsym(handle, "gnutls_record_recv_seq");
    g_fn.gnutls_record_send2 = dlsym(handle, "gnutls_record_send2");
    g_fn.gnutls_record_send_early_data = dlsym(handle, "gnutls_record_send_early_data");
    g_fn.gnutls_record_send_range = dlsym(handle, "gnutls_record_send_range");
    g_fn.gnutls_transport_get_ptr = dlsym(handle, "gnutls_transport_get_ptr");
    g_fn.SSL_ImportFD = dlsym(handle, "SSL_ImportFD");
    g_fn.dlopen = dlsym(handle, "dlopen");
    g_fn.PR_FileDesc2NativeHandle = dlsym(handle, "PR_FileDesc2NativeHandle");
    g_fn.PR_SetError = dlsym(handle, "PR_SetError");
    g_fn.__overflow = dlsym(handle, "__overflow");
    g_fn.sendmmsg = dlsym(handle, "sendmmsg");
    g_fn.recvmmsg = dlsym(handle, "recvmmsg");
    g_fn.pthread_create = dlsym(handle, "pthread_create");
    g_fn.getentropy = dlsym(handle, "getentropy");
    g_fn.__ctype_init = dlsym(handle, "__ctype_init");
    g_fn.__register_atfork = dlsym(handle, "__register_atfork");
#ifdef __STATX__
    g_fn.statx = dlsym(handle, "statx");
#endif // __STATX__

    // functions that can't be restarted, needed for stopTimer()
    // plus poll & select which are used for linux & macos
    g_fn.epoll_wait = dlsym(handle, "epoll_wait");
    g_fn.nanosleep = dlsym(handle, "nanosleep");
    g_fn.sigsuspend = dlsym(handle, "sigsuspend");
    g_fn.pause = dlsym(handle, "pause");
    g_fn.sigwaitinfo = dlsym(handle, "sigwaitinfo");
    g_fn.sigtimedwait = dlsym(handle, "sigtimedwait");
    g_fn.epoll_pwait = dlsym(handle, "epoll_pwait");
    g_fn.ppoll = dlsym(handle, "ppoll");
    g_fn.pselect = dlsym(handle, "pselect");
    g_fn.msgsnd = dlsym(handle, "msgsnd");
    g_fn.msgrcv = dlsym(handle, "msgrcv");
    g_fn.semop = dlsym(handle, "semop");
    g_fn.semtimedop = dlsym(handle, "semtimedop");
    g_fn.clock_nanosleep = dlsym(handle, "clock_nanosleep");
    g_fn.usleep = dlsym(handle, "usleep");
    g_fn.io_getevents = dlsym(handle, "io_getevents");

    // These functions are not interposed.  They're here because
    // we've seen applications override the weak glibc implementation,
    // where our library needs to use the glibc instance.
    // setenv was overriden in bash.
    g_fn.setenv = dlsym(handle, "setenv");

#endif // __LINUX__
#endif // 0
}
