#define _GNU_SOURCE
#include <dlfcn.h>
#include "fn.h"

interposed_funcs_t g_fn;

void
initFn(void)
{
    g_fn.vsyslog = dlsym(RTLD_NEXT, "vsyslog");
    g_fn.fork = dlsym(RTLD_NEXT, "fork");
    g_fn.open = dlsym(RTLD_NEXT, "open");
    g_fn.openat = dlsym(RTLD_NEXT, "openat");
    g_fn.fopen = dlsym(RTLD_NEXT, "fopen");
    g_fn.freopen = dlsym(RTLD_NEXT, "freopen");
    g_fn.creat = dlsym(RTLD_NEXT, "creat");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.fclose = dlsym(RTLD_NEXT, "fclose");
    g_fn.fcloseall = dlsym(RTLD_NEXT, "fcloseall");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.pread = dlsym(RTLD_NEXT, "pread");
    g_fn.readv = dlsym(RTLD_NEXT, "readv");
    g_fn.fread = dlsym(RTLD_NEXT, "fread");
    g_fn.__fread_chk = dlsym(RTLD_NEXT, "__fread_chk");
    g_fn.fread_unlocked = dlsym(RTLD_NEXT, "fread_unlocked");
    g_fn.fgets = dlsym(RTLD_NEXT, "fgets");
    g_fn.__fgets_chk = dlsym(RTLD_NEXT, "__fgets_chk");
    g_fn.fgets_unlocked = dlsym(RTLD_NEXT, "fgets_unlocked");
    g_fn.fgetws = dlsym(RTLD_NEXT, "fgetws");
    g_fn.__fgetws_chk = dlsym(RTLD_NEXT, "__fgetws_chk");
    g_fn.fgetwc = dlsym(RTLD_NEXT, "fgetwc");
    g_fn.fgetc = dlsym(RTLD_NEXT, "fgetc");
    g_fn.fscanf = dlsym(RTLD_NEXT, "fscanf");
    g_fn.fputc = dlsym(RTLD_NEXT, "fputc");
    g_fn.fputc_unlocked = dlsym(RTLD_NEXT, "fputc_unlocked");
    g_fn.fputwc = dlsym(RTLD_NEXT, "fputwc");
    g_fn.putwc = dlsym(RTLD_NEXT, "putwc");
    g_fn.getline = dlsym(RTLD_NEXT, "getline");
    g_fn.getdelim = dlsym(RTLD_NEXT, "getdelim");
    g_fn.__getdelim = dlsym(RTLD_NEXT, "__getdelim");
    g_fn.write = dlsym(RTLD_NEXT, "write");
    g_fn.pwrite = dlsym(RTLD_NEXT, "pwrite");
    g_fn.writev = dlsym(RTLD_NEXT, "writev");
    g_fn.fwrite = dlsym(RTLD_NEXT, "fwrite");
    g_fn.sendfile = dlsym(RTLD_NEXT, "sendfile");
    g_fn.putchar = dlsym(RTLD_NEXT, "putchar");
    g_fn.puts = dlsym(RTLD_NEXT, "puts");
    g_fn.fputs = dlsym(RTLD_NEXT, "fputs");
    g_fn.fputs_unlocked = dlsym(RTLD_NEXT, "fputs_unlocked");
    g_fn.fputws = dlsym(RTLD_NEXT, "fputws");
    g_fn.lseek = dlsym(RTLD_NEXT, "lseek");
    g_fn.fseek = dlsym(RTLD_NEXT, "fseek");
    g_fn.fseeko = dlsym(RTLD_NEXT, "fseeko");
    g_fn.ftell = dlsym(RTLD_NEXT, "ftell");
    g_fn.ftello = dlsym(RTLD_NEXT, "ftello");
    g_fn.fgetpos = dlsym(RTLD_NEXT, "fgetpos");
    g_fn.fsetpos = dlsym(RTLD_NEXT, "fsetpos");
    g_fn.fsetpos64 = dlsym(RTLD_NEXT, "fsetpos64");
    g_fn.stat = dlsym(RTLD_NEXT, "stat");
    g_fn.lstat = dlsym(RTLD_NEXT, "lstat");
    g_fn.fstat = dlsym(RTLD_NEXT, "fstat");
    g_fn.fstatat = dlsym(RTLD_NEXT, "fstatat");
    g_fn.statfs = dlsym(RTLD_NEXT, "statfs");
    g_fn.fstatfs = dlsym(RTLD_NEXT, "fstatfs");
    g_fn.statvfs = dlsym(RTLD_NEXT, "statvfs");
    g_fn.fstatvfs = dlsym(RTLD_NEXT, "fstatvfs");
    g_fn.access = dlsym(RTLD_NEXT, "access");
    g_fn.faccessat = dlsym(RTLD_NEXT, "faccessat");
    g_fn.rewind = dlsym(RTLD_NEXT, "rewind");
    g_fn.fcntl = dlsym(RTLD_NEXT, "fcntl");
    g_fn.fcntl64 = dlsym(RTLD_NEXT, "fcntl64");
    g_fn.dup = dlsym(RTLD_NEXT, "dup");
    g_fn.dup2 = dlsym(RTLD_NEXT, "dup2");
    g_fn.dup3 = dlsym(RTLD_NEXT, "dup3");
    g_fn.socket = dlsym(RTLD_NEXT, "socket");
    g_fn.shutdown = dlsym(RTLD_NEXT, "shutdown");
    g_fn.listen = dlsym(RTLD_NEXT, "listen");
    g_fn.accept = dlsym(RTLD_NEXT, "accept");
    g_fn.accept4 = dlsym(RTLD_NEXT, "accept4");
    g_fn.bind = dlsym(RTLD_NEXT, "bind");
    g_fn.connect = dlsym(RTLD_NEXT, "connect");
    g_fn.send = dlsym(RTLD_NEXT, "send");
    g_fn.sendto = dlsym(RTLD_NEXT, "sendto");
    g_fn.sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    g_fn.recv = dlsym(RTLD_NEXT, "recv");
    g_fn.recvfrom = dlsym(RTLD_NEXT, "recvfrom");
    g_fn.recvmsg = dlsym(RTLD_NEXT, "recvmsg");
    g_fn.gethostbyname = dlsym(RTLD_NEXT, "gethostbyname");
    g_fn.gethostbyname2 = dlsym(RTLD_NEXT, "gethostbyname2");
    g_fn.getaddrinfo = dlsym(RTLD_NEXT, "getaddrinfo");
    g_fn.sigaction = dlsym(RTLD_NEXT, "sigaction");
    g_fn.execve = dlsym(RTLD_NEXT, "execve");
    g_fn.poll = dlsym(RTLD_NEXT, "poll");
    g_fn.select = dlsym(RTLD_NEXT, "select");
#ifdef __MACOS__
    g_fn.close$NOCANCEL = dlsym(RTLD_NEXT, "close$NOCANCEL");
    g_fn.close_nocancel = dlsym(RTLD_NEXT, "close_nocancel");
    g_fn.guarded_close_np = dlsym(RTLD_NEXT, "guarded_close_np");
    g_fn.accept$NOCANCEL = dlsym(RTLD_NEXT, "accept$NOCANCEL");
    g_fn.__sendto_nocancel = dlsym(RTLD_NEXT, "__sendto_nocancel");
    g_fn.DNSServiceQueryRecord = dlsym(RTLD_NEXT, "DNSServiceQueryRecord");
#endif // __MACOS__

#ifdef __LINUX__
    g_fn.open64 = dlsym(RTLD_NEXT, "open64");
    g_fn.openat64 = dlsym(RTLD_NEXT, "openat64");
    g_fn.__open_2 = dlsym(RTLD_NEXT, "__open_2");
    g_fn.__open64_2 = dlsym(RTLD_NEXT, "__open64_2");
    g_fn.__openat_2 = dlsym(RTLD_NEXT, "__openat_2");
    g_fn.fopen64 = dlsym(RTLD_NEXT, "fopen64");
    g_fn.freopen64 = dlsym(RTLD_NEXT, "freopen64");
    g_fn.creat64 = dlsym(RTLD_NEXT, "creat64");
    g_fn.pread64 = dlsym(RTLD_NEXT, "pread64");
    g_fn.preadv = dlsym(RTLD_NEXT, "preadv");
    g_fn.preadv2 = dlsym(RTLD_NEXT, "preadv2");
    g_fn.preadv64v2 = dlsym(RTLD_NEXT, "preadv64v2");
    g_fn.__pread_chk = dlsym(RTLD_NEXT, "__pread_chk");
    g_fn.__read_chk = dlsym(RTLD_NEXT, "__read_chk");
    g_fn.__fread_unlocked_chk = dlsym(RTLD_NEXT, "__fread_unlocked_chk");
    g_fn.pwrite64 = dlsym(RTLD_NEXT, "pwrite64");
    g_fn.pwritev = dlsym(RTLD_NEXT, "pwritev");
    g_fn.pwritev64 = dlsym(RTLD_NEXT, "pwritev64");
    g_fn.pwritev2 = dlsym(RTLD_NEXT, "pwritev2");
    g_fn.pwritev64v2 = dlsym(RTLD_NEXT, "pwritev64v2");
    g_fn.fwrite_unlocked = dlsym(RTLD_NEXT, "fwrite_unlocked");
    g_fn.sendfile64 = dlsym(RTLD_NEXT, "sendfile64");
    g_fn.lseek64 = dlsym(RTLD_NEXT, "lseek64");
    g_fn.fseeko64 = dlsym(RTLD_NEXT, "fseeko64");
    g_fn.ftello64 = dlsym(RTLD_NEXT, "ftello64");
    g_fn.statfs64 = dlsym(RTLD_NEXT, "statfs64");
    g_fn.fstatfs64 = dlsym(RTLD_NEXT, "fstatfs64");
    g_fn.fstatvfs64 = dlsym(RTLD_NEXT, "fstatvfs64");
    g_fn.fgetpos64 = dlsym(RTLD_NEXT, "fgetpos64");
    g_fn.statvfs64 = dlsym(RTLD_NEXT, "statvfs64");
    g_fn.__lxstat = dlsym(RTLD_NEXT, "__lxstat");
    g_fn.__lxstat64 = dlsym(RTLD_NEXT, "__lxstat64");
    g_fn.__xstat = dlsym(RTLD_NEXT, "__xstat");
    g_fn.__xstat64 = dlsym(RTLD_NEXT, "__xstat64");
    g_fn.__fxstat = dlsym(RTLD_NEXT, "__fxstat");
    g_fn.__fxstat64 = dlsym(RTLD_NEXT, "__fxstat64");
    g_fn.__fxstatat = dlsym(RTLD_NEXT, "__fxstatat");
    g_fn.__fxstatat64 = dlsym(RTLD_NEXT, "__fxstatat64");
    g_fn.gethostbyname_r = dlsym(RTLD_NEXT, "gethostbyname_r");
    g_fn.gethostbyname2_r = dlsym(RTLD_NEXT, "gethostbyname2_r");
    g_fn.syscall = dlsym(RTLD_NEXT, "syscall");
    g_fn.prctl = dlsym(RTLD_NEXT, "prctl");
    g_fn._exit = dlsym(RTLD_NEXT, "_exit");
    g_fn.SSL_read = dlsym(RTLD_NEXT, "SSL_read");
    g_fn.SSL_write = dlsym(RTLD_NEXT, "SSL_write");
    g_fn.SSL_get_fd = dlsym(RTLD_NEXT, "SSL_get_fd");
    g_fn.gnutls_record_recv = dlsym(RTLD_NEXT, "gnutls_record_recv");
    g_fn.gnutls_record_send = dlsym(RTLD_NEXT, "gnutls_record_send");
    g_fn.gnutls_record_recv_early_data = dlsym(RTLD_NEXT, "gnutls_record_recv_early_data");
    g_fn.gnutls_record_recv_packet = dlsym(RTLD_NEXT, "gnutls_record_recv_packet");
    g_fn.gnutls_record_recv_seq = dlsym(RTLD_NEXT, "gnutls_record_recv_seq");
    g_fn.gnutls_record_send2 = dlsym(RTLD_NEXT, "gnutls_record_send2");
    g_fn.gnutls_record_send_early_data = dlsym(RTLD_NEXT, "gnutls_record_send_early_data");
    g_fn.gnutls_record_send_range = dlsym(RTLD_NEXT, "gnutls_record_send_range");
    g_fn.gnutls_transport_get_ptr = dlsym(RTLD_NEXT, "gnutls_transport_get_ptr");
    g_fn.SSL_ImportFD = dlsym(RTLD_NEXT, "SSL_ImportFD");
    g_fn.dlopen = dlsym(RTLD_NEXT, "dlopen");
    g_fn.PR_FileDesc2NativeHandle = dlsym(RTLD_NEXT, "PR_FileDesc2NativeHandle");
    g_fn.PR_SetError = dlsym(RTLD_NEXT, "PR_SetError");
    g_fn.__overflow = dlsym(RTLD_NEXT, "__overflow");
    g_fn.sendmmsg = dlsym(RTLD_NEXT, "sendmmsg");
    g_fn.recvmmsg = dlsym(RTLD_NEXT, "recvmmsg");
    g_fn.getentropy = dlsym(RTLD_NEXT, "getentropy");
#ifdef __STATX__
    g_fn.statx = dlsym(RTLD_NEXT, "statx");
#endif // __STATX__

    // functions that can't be restarted, needed for stopTimer()
    // plus poll & select which are used for linux & macos
    g_fn.epoll_wait = dlsym(RTLD_NEXT, "epoll_wait");
    g_fn.nanosleep = dlsym(RTLD_NEXT, "nanosleep");
    g_fn.sigsuspend = dlsym(RTLD_NEXT, "sigsuspend");
    g_fn.pause = dlsym(RTLD_NEXT, "pause");
    g_fn.sigwaitinfo = dlsym(RTLD_NEXT, "sigwaitinfo");
    g_fn.sigtimedwait = dlsym(RTLD_NEXT, "sigtimedwait");
    g_fn.epoll_pwait = dlsym(RTLD_NEXT, "epoll_pwait");
    g_fn.ppoll = dlsym(RTLD_NEXT, "ppoll");
    g_fn.pselect = dlsym(RTLD_NEXT, "pselect");
    g_fn.msgsnd = dlsym(RTLD_NEXT, "msgsnd");
    g_fn.msgrcv = dlsym(RTLD_NEXT, "msgrcv");
    g_fn.semop = dlsym(RTLD_NEXT, "semop");
    g_fn.semtimedop = dlsym(RTLD_NEXT, "semtimedop");
    g_fn.clock_nanosleep = dlsym(RTLD_NEXT, "clock_nanosleep");
    g_fn.usleep = dlsym(RTLD_NEXT, "usleep");
    g_fn.io_getevents = dlsym(RTLD_NEXT, "io_getevents");

    // These functions are not interposed.  They're here because
    // we've seen applications override the weak glibc implementation,
    // where our library needs to use the glibc instance.
    // setenv was overriden in bash.
    g_fn.setenv = dlsym(RTLD_NEXT, "setenv");

#endif // __LINUX__
}
