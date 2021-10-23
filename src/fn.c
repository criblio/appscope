#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <link.h>

#include "dbg.h"
#include "utils.h"
#include "fn.h"

static int g_ismusl = FALSE;

/*
 * We need to adjust to how ld.so, libdl, dlopen, dlsym together
 * resolve symbols. Several use cases have revealed different
 * behavior as symbols are resolved. We have been able to resolve
 * symbols for g_fn using the RTLD_NEXT handle with dlsym(). That
 * has worked in all preload cases. When we added an attach feature
 * it was revealed that symbols were not resolved as expected in
 * several cases. The context in which the library is loaded and
 * therefore, the context in which the constructor is run, is quite
 * different as compared to process load time when preload is applied.
 * Resolving symbols at load time versus run time, or preload versus
 * postload, is, in many cases, quite different. In the attach case
 * the lib constructor is run after all required libs have been loaded.
 * In the post load attach case we will see ld.so resolve symbols from
 * libscope where they need to be resolved by system libs; libc, libpthred,
 * libdl, et al.
 *
 * It becomes important to note that the handle obtained by dlopen() and used
 * with dlsym() is a starting point. That is, the symbol search can be made to
 * start at a specific object, but it is not limited to the symbols defined by
 * that object. In fact, the starting point defined by a handle defines a
 * hierarchical walk of the symbols defined by ld.so. For example, a handle
 * returned by dlopen("libdl.so") will search symbols in libdl.so. If the symbol
 * is not resolved the search will continue to include shared objects that are
 * required by libdl.so and then those used by the executable at large.
 *
 * In the libscope.so case there are several dependent libs; libc, libdl, librt,
 * libpthread, ld.so. In general, a hierarchy exists, not quite literal, but in
 * practical terms, libc is not dependent on other libs. Likewise, libpthread is
 * dependent on libc and librt is dependent on libpthread and libc. We make use
 * of this dependency by starting a symbol search at librt. The search order becomes
 * librt, libpthread, libc, executable. The intent is that a symbol will be resolved
 * by librt or libpthread before libc. There are duplicate symbols in libs. For example,
 * the function read() is defined in libpthread and libc. The idea being that some
 * symbols in libc are utilized for single threaded apps where the overhead of locking
 * is not needed. The use of weak symbol definitions allow the duplication to work. We
 * want to resolve to thread safe implementations where they exist. Therefore, if a symbol
 * could be resolved in libpthread and libc, we'd like to resolve from libpthread.
 *
 * In general, what we've done is walk all objects in an executable using dl_iterate_phdr().
 * We use a call back function that causes a symbol search that starts at librt. This is
 * accomplished with a dlopen("librt") and a dlsym() with the resulting handle.In a
 * GNU libc distro this resolves most symbols from libpthread and libc.
 *
 * A musl libc distro doesn't contain a lib hierarchy such as that found in a GNU libc
 * based distro. There is, for the most part, a single lib that includes all supported
 * POSIX definitions in a musl libc distro. Therefore, the act of walking all shared
 * objects, starting with librt doesn't function the same in a musl libc instance.
 * Symbols aren't resolved in this case. We add a second clause to symbol resolution
 * to use the RTLD_NEXT handle with dlsym() in order to resolve symbols in the musl libc
 * case. This works well for preload and postload cases with the exception of static execs.
 *
 * On a musl libc distro symbols from a static executable cannot be resolved with the
 * RTLD_NEXT handle. Presumably, there is no next in the static exec. It makes sense in
 * that all of the functions are fully resolved and resident in the exec. Therefore, starting
 * a symbol search from an object after the exec could be seen as problematic. This can be
 * confusing as the use of the RTLD_NEXT handle in static exec works with ld-linux.so, the
 * GNU loader.
 *
 * In order to effectively support static execs we add a clause to symbol resolution to
 * detect a static exec and use the RTLD_DEFAULT handle with dlsym(). While this is not
 * strictly required in a GNU libc distro, it does appear to be correct behavior. We see
 * that ld-linux.so resolves symbols as expected on static execs with RTLD_DEFAULT.
 *
 * It's important to note that the use of RTLD_DEFAULT in a dynamic exec generally
 * results in symbols being resolved from libscope. Therefore, we need to use this
 * handle in a static exec case only. It makes sense in that RTLD_DEFAULT causes a
 * symbol search to begin with the current object. When executed from the libscope
 * constructor, this results in symbols that we want to locate in libpthread or libc
 * to be resolved from libscope. That doesn't work.
*/
#define GETADDR(val, sym)                                \
    ares.in_symbol = sym;                                \
    if (checkEnv("SCOPE_EXEC_TYPE", "static") == TRUE) { \
        val = dlsym(RTLD_DEFAULT, sym);                  \
        /* GNU ld.so */                                  \
    } else if (dl_iterate_phdr(getAddr, &ares)) {        \
        val = ares.out_addr;                             \
        /* musl ld.so when loading an app from exec */   \
    } else if ((val = dlsym(RTLD_NEXT, sym))) {          \
        DBG(NULL);                                       \
        /* musl ld.so when attaching */                  \
    } else if (g_ismusl && (val = dlsym(RTLD_DEFAULT, sym))) {       \
        DBG(NULL);                                       \
    } else {                                             \
        val = NULL;                                      \
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
initFn_musl(void)
{
    // For use in GETADDR macro...
    g_ismusl = TRUE;

    initFn();
}

void
initFn(void)
{
    addresult_t ares;

    g_fn.dlopen = dlsym(RTLD_NEXT, "dlopen");
    if (!g_fn.dlopen) g_fn.dlopen = dlsym(RTLD_DEFAULT, "dlopen");

    g_fn.SSL_read = dlsym(RTLD_NEXT, "SSL_read");
    g_fn.SSL_write = dlsym(RTLD_NEXT, "SSL_write");
    g_fn.SSL_get_fd = dlsym(RTLD_NEXT, "SSL_get_fd");

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
    g_fn.app_setenv = dlsym(RTLD_DEFAULT, "setenv");
#ifdef __STATX__
    GETADDR(g_fn.statx, "statx");
#endif
}
