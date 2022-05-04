#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/un.h>

#include "dbg.h"
#include "scopetypes.h"
#include "os.h"
#include "scopestdlib.h"
#include "fn.h"
#include "transport.h"

// Yuck.  Avoids naming conflict between our src/wrap.c and libssl.a
#define SSL_read SCOPE_SSL_read
#define SSL_write SCOPE_SSL_write
#include "openssl/ssl.h"
#include "openssl/err.h"
#undef SSL_read
#undef SSL_write

struct _transport_t
{
    cfg_transport_t type;
    int (*getaddrinfo)(const char *, const char *,
                       const struct addrinfo *,
                       struct addrinfo **);
    int (*origGetaddrinfo)(const char *, const char *,
                           const struct addrinfo *,
                           struct addrinfo **);
    union {
        struct {
            int sock;
            int pending_connect;
            uint64_t connect_attempts;
            net_fail_t failure_reason;
            char *host;
            char *port;
            struct sockaddr_storage gai_addr;
            struct {
                // Configuration
                unsigned enable;
                unsigned validateserver;
                char *cacertpath;
                // Operational params
                SSL_CTX *ctx;
                SSL *ssl;
            } tls;
            struct {
                int entries;
                struct addrinfo *next;
                struct addrinfo *list;
            } addr;
        } net;
        struct {
            int sock;
            char *path; // original configured path... for error reporting
            struct sockaddr_un addr;
            int addr_len;
        } local; // aka "unix".  Can't use "unix" because it's a macro name
        struct {
            char *path;
            FILE *stream;
            int stdout;  // Flag to indicate that stream is stdout
            int stderr;  // Flag to indicate that stream is stderr
            cfg_buffer_t buf_policy;
        } file;
    };
};

// This is *not* realtime safe; it's shared between all transports in a
// process.  It's used by scopeGetaddrinfo() to avoid a bug seen in
// node.js processes.  See transportReconnect() below for details.
static struct addrinfo *g_cached_addr = NULL;

static void (*handleExit_fn)(void) = NULL;

static transport_t*
newTransport()
{
    transport_t *t;

    t = scope_calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    t->getaddrinfo = scope_getaddrinfo;
    t->origGetaddrinfo = t->getaddrinfo;  // store a copy
    return t;
}

/*
 * Some apps require that a set of fds, usually low numbers, 0-20,
 * must exist. Therefore, we don't want to allow the kernel to
 * give us the next available fd. We need to place the fd in a
 * range that is likely not to affect an app. 
 *
 * We look for an available fd starting at a relatively high
 * range and work our way down until we find one we can get.
 * Then, we force the use of the available fd. 
 */
static int
placeDescriptor(int fd, transport_t *t)
{
    // next_fd_to_try avoids reusing file descriptors.
    // Without this, we've had problems where the buffered stream for
    // g_log has it's fd closed and reopened by another transport which
    // causes the mis-routing of data.
    static int next_fd_to_try = DEFAULT_FD;
    if (next_fd_to_try < DEFAULT_MIN_FD) {
        next_fd_to_try = DEFAULT_FD;
    }

    int i, dupfd;

    for (i = next_fd_to_try; i >= DEFAULT_MIN_FD; i--) {
        if ((scope_fcntl(i, F_GETFD) == -1) && (scope_errno == EBADF)) {

            // This fd is available, try to dup it
            if ((dupfd = scope_dup2(fd, i)) == -1) continue;
            scope_close(fd);

            // Set close on exec. (dup2 does not preserve FD_CLOEXEC)
            int flags = scope_fcntl(dupfd, F_GETFD, 0);
            if (scope_fcntl(dupfd, F_SETFD, flags | FD_CLOEXEC) == -1) {
                DBG("%d", dupfd);
            }

            next_fd_to_try = dupfd - 1;
            return dupfd;
        }
    }
    DBG("%d", t->type);
    scope_close(fd);
    return -1;
}

cfg_transport_t
transportType(transport_t *trans)
{
    if (!trans) return (cfg_transport_t)-1;

    return trans->type;
}

int
transportConnection(transport_t *trans)
{
    if (!trans) return -1;
    switch(trans->type) {
        case CFG_UDP:
        case CFG_TCP:
            if (trans->net.sock != -1) {
                return trans->net.sock;
            }
            return trans->net.pending_connect;
        case CFG_UNIX:
        case CFG_EDGE:
            return trans->local.sock;
        case CFG_FILE:
            if (trans->file.stream) {
                return scope_fileno(trans->file.stream);
            } else {
                return -1;
            }
        case CFG_SYSLOG:
        case CFG_SHM:
            break;
        default:
            DBG(NULL);
    }

    return -1;
}

int
transportNeedsConnection(transport_t *trans)
{
    if (!trans) return FALSE;
    switch (trans->type) {
        case CFG_UDP:
        case CFG_TCP:
            if ((trans->net.sock == -1) ||
                (trans->net.tls.enable && !trans->net.tls.ssl)) return TRUE;
            if (osNeedsConnect(trans->net.sock)) {
                DBG("fd:%d, tls:%d", trans->net.sock, trans->net.tls.enable);
                if (trans->net.tls.enable) {
                    scopeLogInfo("fd:%d tls session closed remotely", trans->net.sock);
                } else {
                    scopeLogInfo("fd:%d tcp connection closed remotely", trans->net.sock);
                }
                transportDisconnect(trans);
                return TRUE;
            }
            return FALSE;
        case CFG_FILE:
            // This checks to see if our file descriptor has been
            // closed by our process.  (errno == EBADF) Stream buffering
            // makes it harder to know when this has happened.
            if ((trans->file.stream) &&
                (scope_fcntl(scope_fileno(trans->file.stream), F_GETFD) == -1)) {
                DBG(NULL);
                transportDisconnect(trans);
            }
            return (trans->file.stream == NULL);
        case CFG_UNIX:
        case CFG_EDGE:
            if (trans->local.sock == -1) return TRUE;
            if (osNeedsConnect(trans->local.sock)) {
                DBG("fd:%d %s", trans->local.sock, trans->local.path);
                transportDisconnect(trans);
                return TRUE;
            }
            return FALSE;
        case CFG_SYSLOG:
        case CFG_SHM:
            break;
        default:
            DBG(NULL);
    }
    return FALSE;
}

static void
loadRootCertFile(transport_t *trans)
{
    char *cafile = trans->net.tls.cacertpath;

    // If the configuration provides a cacertpath, use it.
    // Otherwise, find a distro-specific root cert file.
    if (!cafile) {
        int i;
        // Based off of this: https://golang.org/src/crypto/x509/root_linux.go
        const char* const rootFileList[] = {
            "/etc/ssl/certs/ca-certificates.crt",                // Debian/Ubuntu/Gentoo etc.
            "/etc/pki/tls/certs/ca-bundle.crt",                  // Fedora/RHEL 6
            "/etc/ssl/ca-bundle.pem",                            // OpenSUSE
            "/etc/pki/tls/cacert.pem",                           // OpenELEC
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", // CentOS/RHEL 7
            "/etc/ssl/cert.pem"                                  // Alpine Linux
        };

        for (i=0; i<sizeof(rootFileList)/sizeof(char*); ++i) {
            if (!scope_access(rootFileList[i], R_OK)) {
                cafile = (char*)rootFileList[i];
                break;
            }
        }
    }

    long loc_rv = SSL_CTX_load_verify_locations(trans->net.tls.ctx, cafile, NULL);
    if (trans->net.tls.validateserver && !loc_rv) {
        char err[256] = {0};
        ERR_error_string_n(ERR_peek_last_error() , err, sizeof(err));
        scopeLogInfo("fd:%d error setting tls cacertpath: \"%s\" : %s", trans->net.sock, cafile, err);
        // We're not treating this as a hard error at this point.
        // Let the process proceed; validation below will likely fail
        // and might provide more meaningful info.
    }
}

static void
shutdownTlsSession(transport_t *trans)
{
    if (trans->net.tls.ssl) {
        int ret = SSL_shutdown(trans->net.tls.ssl);
        if (ret < 0) {
            // protocol error occurred
            int ssl_err = SSL_get_error(trans->net.tls.ssl, ret);
            scopeLogInfo("Client SSL_shutdown failed: ssl_err=%d\n", ssl_err);
        }
        SSL_free(trans->net.tls.ssl);
        trans->net.tls.ssl = NULL;
    }

    if (trans->net.tls.ctx) {
        SSL_CTX_free(trans->net.tls.ctx);
        trans->net.tls.ctx = NULL;
    }

    if (trans->net.sock != -1) {
        scope_shutdown(trans->net.sock, SHUT_RDWR);
        scope_close(trans->net.sock);
        trans->net.sock = -1;
    }
}

static void
handle_tls_destroy(void)
{
    scopeLogInfo("detected beginning of process exit sequence");

    if (handleExit_fn) handleExit_fn();
}

void
transportRegisterForExitNotification(void (*fn)(void))
{
    // remember what to call when OPENSSL is being destructed.
    handleExit_fn = fn;

    // register so handle_tls_destroy() is called as the process exits.
    if (!OPENSSL_atexit(handle_tls_destroy)) {
        DBG(NULL);
    }

    // This ensures that where TLS is not enabled we will get our exit
    // handler called. It's safe to call the handler more than once.
    atexit(fn);
}

static int
establishTlsSession(transport_t *trans)
{
    if (!trans || trans->net.sock == -1) return FALSE;
    scopeLogInfo("fd:%d establishing tls session", trans->net.sock);

    static int init_called = FALSE;
    if (!init_called) {
        OPENSSL_init_ssl(OPENSSL_INIT_NO_ATEXIT, NULL);
        init_called = TRUE;
    }

    trans->net.tls.ctx = SSL_CTX_new(TLS_method());
    if (!trans->net.tls.ctx) {
        char err[256] = {0};
        ERR_error_string_n(ERR_peek_last_error() , err, sizeof(err));
        scopeLogInfo("fd:%d error creating tls context: %s", trans->net.sock, err);
        trans->net.failure_reason = TLS_CONTEXT_FAIL;
        goto err;
    }

    loadRootCertFile(trans);

    trans->net.tls.ssl = SSL_new(trans->net.tls.ctx);
    if (!trans->net.tls.ssl) {
        char err[256] = {0};
        ERR_error_string_n(ERR_peek_last_error() , err, sizeof(err));
        scopeLogInfo("fd:%d error creating tls session: %s", trans->net.sock, err);
        trans->net.failure_reason = TLS_SESSION_FAIL;
        goto err;
    }

    if (!SSL_set_fd(trans->net.tls.ssl, trans->net.sock)) {
        char err[256] = {0};
        ERR_error_string_n(ERR_peek_last_error() , err, sizeof(err));
        scopeLogInfo("fd:%d error setting tls on socket: %d : %s", trans->net.sock, trans->net.sock, err);
        trans->net.failure_reason = TLS_SOCKET_FAIL;
        goto err;
    }

    ERR_clear_error(); // to make SSL_get_error reliable
    int con_rv = SSL_connect(trans->net.tls.ssl);
    if (con_rv != 1) {
        char err[256] = {0};
        int ssl_err = SSL_get_error(trans->net.tls.ssl, con_rv);
        ERR_error_string_n(ssl_err, err, sizeof(err));
        scopeLogInfo("fd:%d error establishing tls connection: %s", trans->net.sock, err);
        if (ssl_err == SSL_ERROR_SSL || ssl_err == SSL_ERROR_SYSCALL) {
            ERR_error_string_n(ERR_peek_last_error() , err, sizeof(err));
            scopeLogInfo("fd:%d error establishing tls connection: %s %d", trans->net.sock, err, errno);
            trans->net.failure_reason = TLS_CONN_FAIL;
        }
        goto err;
    }

    // This improves the delivery but we're unsure of what the cost is
    // in terms of network usage.
    // See https://github.com/criblio/appscope/issues/781
    //
    // BIO_set_tcp_ndelay(trans->net.sock, TRUE);

    if (trans->net.tls.validateserver) {
        // Just test that we received a server cert
        X509* cert = SSL_get_peer_certificate(trans->net.tls.ssl);
        if (cert) {
            X509_free(cert);  // Looks good.  Free it immediately
        } else {
            scopeLogInfo("fd:%d error accessing peer certificate for tls server validation",
                                                  trans->net.sock);
            trans->net.failure_reason = TLS_CERT_FAIL;
            goto err;
        }

        long ver_rc = SSL_get_verify_result(trans->net.tls.ssl);
        if (ver_rc != X509_V_OK) {
            const char *err = X509_verify_cert_error_string(ver_rc);
            scopeLogInfo("fd:%d tls server validation failed : \"%s\"", trans->net.sock, err);
            trans->net.failure_reason = TLS_VERIFY_FAIL;
            goto err;
        }
    }

    scopeLogInfo("fd:%d tls session established", trans->net.sock);
    return TRUE;
err:
    shutdownTlsSession(trans);
    return FALSE;
}

int
transportDisconnect(transport_t *trans)
{
    if (!trans) return 0;
    switch (trans->type) {
        case CFG_UDP:
        case CFG_TCP:
            // appropriate for both tls and non-tls connections...
            shutdownTlsSession(trans);
            if (trans->net.pending_connect != -1) {
                scope_close(trans->net.pending_connect);
                trans->net.pending_connect = -1;
            }
            break;
        case CFG_FILE:
            if (!trans->file.stdout && !trans->file.stderr) {
                if (trans->file.stream) scope_fclose(trans->file.stream);
            }
            trans->file.stream = NULL;
            break;
        case CFG_UNIX:
        case CFG_EDGE:
            if (trans->local.sock != -1) {
                scope_close(trans->local.sock);
                trans->local.sock = -1;
            }
            break;
        case CFG_SYSLOG:
        case CFG_SHM:
            break;
        default:
            DBG(NULL);
    }
    return 0;
}


// We've observed that node.js processes can hang from spinlocks
// in glibc's getaddrinfo:
//
//      #0  __lll_lock_wait_private ()
//                 at ../sysdeps/unix/sysv/linux/x86_64/lowlevellock.S:95
//      #1  in get_locked_global () at resolv_conf.c:90
//      #2  resolv_conf_get_1 () at resolv_conf.c:200
//      #3  __resolv_conf_get () at resolv_conf.c:359
//      #4  in context_alloc () at resolv_context.c:137
//      #5  context_get (preinit=false) at resolv_context.c:181
//      #6  __GI___resolv_context_get () at resolv_context.c:195
//      #7  in gaih_inet () at ../sysdeps/posix/getaddrinfo.c:767
//      #8  in __GI_getaddrinfo () at ../sysdeps/posix/getaddrinfo.c:2300
//      #9  in socketConnectionStart () at src/transport.c:339
//      #10 in transportConnect () at src/transport.c:514
//      #11 in transportCreateTCP () at src/transport.c:549
//      #12 in initTransport (cfg=0x5446bb0, t=CFG_CTL) at src/cfgutils.c:1516
//      #13 in initCtl (cfg=0x5446bb0) at src/cfgutils.c:1609
//      #14 in doReset () at src/wrap.c:637
//      #15 in fork () at src/wrap.c:3361
//      #16 in uv_spawn () at ../deps/uv/src/unix/process.c:489
//
// Here's our own version that returns an address from a previously
// successful connection.  Look ma, no spinlocks!  See transportReconnect()
// below for more info.
static int
scopeGetaddrinfo(const char *node, const char *service,
                  const struct addrinfo *hints,
                  struct addrinfo **res)
{
    if (!res) return 1;
    *res = g_cached_addr;
    return (g_cached_addr) ? 0 : 1; // 0 is successful
}

static struct addrinfo *
getExistingConnectionAddr(transport_t *trans)
{
    struct addrinfo *ai = NULL;
    if (transportNeedsConnection(trans) || trans->type != CFG_TCP) goto exit;

    // Clear the address value
    socklen_t addrsize = sizeof(trans->net.gai_addr);
    struct sockaddr *addr = (struct sockaddr*)&trans->net.gai_addr;
    scope_memset(addr, 0, addrsize);

    // lookup the address
    if (scope_getpeername(trans->net.sock, addr, &addrsize)) {
        DBG(NULL);
        goto exit;
    }

    int res = scope_copyaddrinfo(addr, addrsize, &ai);
    if (res) {
        DBG(NULL);
    }

exit:
    return ai;
}


// This is expected to be called by child processes that
// may have inherited connected transports from their parent
// processes.  i.e. fork()->doReset() path
// As a caution, because of its use of g_cached_addr, it's
// *not* reentrant.
int
transportReconnect(transport_t *trans)
{
    if (!trans) return 0;

    switch (trans->type) {
        case CFG_TCP:
            // Since TCP is connection-oriented, we want to disconnect
            // and reconnect so child processes can have distinct
            // connections from their parents.  However, we can't use
            // glibc's getaddrinfo lest we introduce hangs in node.js
            // processes.  So, if a transport has an existing connection,
            // grab the address from that connection and substitute in our
            // own getaddrinfo for this situation.

            g_cached_addr = getExistingConnectionAddr(trans);

            // We want to close the socket we got from our parent process, but don't
            // want to send a close notification for the SSL session.  If we sent the
            // close notification, it has the side effect of closing our parent process's
            // ssl session.
            if (trans->net.tls.enable && trans->net.tls.ssl) {
                SSL_set_quiet_shutdown(trans->net.tls.ssl, TRUE);
            }

            transportDisconnect(trans);          // Never keep the parents connection.
            if (g_cached_addr) {
                trans->getaddrinfo = scopeGetaddrinfo;
                transportConnect(trans);         // Will use g_cached_addr
                trans->getaddrinfo = trans->origGetaddrinfo;
            }

            break;
        case CFG_UNIX:
        case CFG_UDP:
        case CFG_FILE:
        case CFG_SYSLOG:
        case CFG_SHM:
        case CFG_EDGE:
            // Everything else is a no-op.  These can all share
            // the parent's transport.
            break;
        default:
            DBG(NULL);
    }
    return 0;
}

static int
setSocketBlocking(transport_t *trans, int sock, bool block)
{
    if (!trans) return 0;

    int current_flags = scope_fcntl(sock, F_GETFL, NULL);
    if (current_flags < 0) return FALSE;

    int desired_flags;
    if (block) {
        desired_flags = current_flags & ~O_NONBLOCK;
    } else {
        desired_flags = current_flags | O_NONBLOCK;
    }

    // We're successful; the flag is as desired
    if (current_flags == desired_flags) return TRUE;

    // fcntl returns 0 if successful
    return (scope_fcntl(sock, F_SETFL, desired_flags) == 0);
}

static int
socketConnectIsPending(transport_t *trans)
{
    return (!trans || trans->net.pending_connect < 0) ? FALSE : TRUE;
}

static int
checkPendingSocketStatus(transport_t *trans)
{
    if (!trans || trans->net.pending_connect == -1) return 0;
    int rc;
    struct timeval tv = {0};

    fd_set pending_results;
    FD_ZERO(&pending_results);
    FD_SET(trans->net.pending_connect, &pending_results);
    rc = scope_select(FD_SETSIZE, NULL, &pending_results, NULL, &tv);
    if (rc < 0) {
        if (scope_errno == EINTR) {
          return 0;
        }
        DBG(NULL);
        transportDisconnect(trans);
        return 0;
    } else if (rc == 0) {
        // No new status is available
        return 0;
    }

    // If we can't get socket status, or the status is an error, close the
    // socket that failed to connect and remove it from the pending list.
    int opt;
    socklen_t optlen = sizeof(opt);
    if ((scope_getsockopt(trans->net.pending_connect, SOL_SOCKET, SO_ERROR, (void*)(&opt), &optlen) < 0)
            || opt) {
        scopeLogInfo("fd:%d connect failed", trans->net.pending_connect);

        scope_close(trans->net.pending_connect);
        trans->net.pending_connect = -1;
        return 0;
    }

    // We have a connection
    scopeLogInfo("fd:%d connect successful", trans->net.pending_connect);

    // Move this descriptor up out of the way
    trans->net.sock = placeDescriptor(trans->net.pending_connect, trans);

    // Remove the pending status from the transport
    trans->net.pending_connect = -1;

    // If the placeDescriptor call failed, we're done
    if (trans->net.sock == -1) return 0;

    // Set the TCP socket to blocking
    if ((trans->type == CFG_TCP) && !setSocketBlocking(trans, trans->net.sock, TRUE)) {
        DBG("%d %s %s", trans->net.sock, trans->net.host, trans->net.port);
    }

    // Set TCP_QUICKACK
#if defined(TCP_QUICKACK) && (defined(IPPROTO_TCP) || defined(SOL_TCP))
    if (trans->type == CFG_TCP) {
        int opt;
        int on = TRUE;

#ifdef SOL_TCP
        opt=SOL_TCP;
#else
#ifdef IPPROTO_TCP
        opt=IPPROTO_TCP;
#endif
#endif
        if (scope_setsockopt(trans->net.sock, opt, TCP_QUICKACK, &on, sizeof(on))) {
            DBG("%d %s %s", trans->net.sock, trans->net.host, trans->net.port);
        }
    }
#endif


    // We have a connected socket!  Woot!
    trans->net.connect_attempts = 0;
    trans->net.failure_reason = NO_FAIL;
    
    // Do the tls stuff for this connection as needed.
    if (trans->net.tls.enable) {
        // when successful, we'll have a connected tls socket.
        // when not, this will cleanup, disconnecting the socket.
        establishTlsSession(trans);
    }

    return 1;
}

static void
freeAddressList(transport_t *trans)
{
    if (!trans || !trans->net.addr.list) return;

    scope_freeaddrinfo(trans->net.addr.list);
    trans->net.addr.entries = 0;
    trans->net.addr.list = NULL;
    trans->net.addr.next = NULL;
}

static int
getAddressList(transport_t *trans)
{
    // Don't leak; clean up any prior data
    freeAddressList(trans);

    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;     // IPv4 or IPv6

    switch (trans->type) {
        case CFG_UDP:
            hints.ai_socktype = SOCK_DGRAM;  // For UDP
            hints.ai_protocol = IPPROTO_UDP; // For UDP
            break;
        case CFG_TCP:
            hints.ai_socktype = SOCK_STREAM; // For TCP
            hints.ai_protocol = IPPROTO_TCP; // For TCP
            break;
        default:
            DBG(NULL);
            return 0;
    }

    char *type = (trans->type == CFG_UDP) ? "udp" : "tcp";
    scopeLogInfo("getting DNS info for %s %s:%s", type, trans->net.host, trans->net.port);

    if (trans->getaddrinfo(trans->net.host,
                           trans->net.port,
                           &hints, &addr_list)) return 0;

    // Count how many addrs we got back
    struct addrinfo* addr;
    for (addr = addr_list; addr; addr = addr->ai_next) {
        trans->net.addr.entries++;
    }
    trans->net.addr.list = addr_list;
    trans->net.addr.next = addr_list; // next is initially the first element

    return trans->net.addr.entries;
}

static struct addrinfo *
getNextAddressListEntry(transport_t *trans)
{
    if (!trans || !trans->net.addr.list || !trans->net.addr.next) return NULL;

    // record the next value to return, and advance it for subsequent calls
    struct addrinfo *next = trans->net.addr.next;
    trans->net.addr.next = next->ai_next;

    return next;
}


static int
socketConnectionStart(transport_t *trans)
{
    trans->net.connect_attempts++;

    // Get a list of addresses to try if we don't have a current list
    // or have exhausted the entries in a current list.
    if (!trans->net.addr.list || !trans->net.addr.next) {
        getAddressList(trans);
    }

    // try the next address in the address list
    struct addrinfo* addr;
    while ((addr = getNextAddressListEntry(trans))) {
        int sock;
        sock = scope_socket(addr->ai_family,
                           addr->ai_socktype,
                           addr->ai_protocol);

        if (sock == -1) continue;

        // Set the socket to close on exec
        int flags = scope_fcntl(sock, F_GETFD, 0);
        if (scope_fcntl(sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
            DBG("%d %s %s", sock, trans->net.host, trans->net.port);
        }

        // Connect will hang in some cases; start by setting non-blocking
        if (!setSocketBlocking(trans, sock, FALSE)) {
            DBG("%d %s %s", sock, trans->net.host, trans->net.port);
            transportDisconnect(trans);
            continue;
        }

        void *addrptr = NULL;
        unsigned short *portptr = NULL;
        if (addr->ai_family == AF_INET) {
            struct sockaddr_in *addr4_ptr;
            addr4_ptr = (struct sockaddr_in *)addr->ai_addr;
            addrptr = &addr4_ptr->sin_addr;
            portptr = &addr4_ptr->sin_port;
        } else if (addr->ai_family == AF_INET6) {
            struct sockaddr_in6 *addr6_ptr;
            addr6_ptr = (struct sockaddr_in6 *)addr->ai_addr;
            addrptr = &addr6_ptr->sin6_addr;
            portptr = &addr6_ptr->sin6_port;
        } else {
            DBG("%d %s %s %d", sock, trans->net.host, trans->net.port, addr->ai_family);
            scope_close(sock);
            continue;
        }
        char addrstr[INET6_ADDRSTRLEN];
        scope_inet_ntop(addr->ai_family, addrptr, addrstr, sizeof(addrstr));
        unsigned short port = scope_ntohs(*portptr);
        scope_errno = 0;
        if (scope_connect(sock, addr->ai_addr, addr->ai_addrlen) == -1) {

            if (scope_errno != EINPROGRESS) {
                scopeLogInfo("fd:%d connect to %s:%d failed", sock, addrstr, port);
                trans->net.failure_reason = CONN_FAIL;

                // We could create a sock, but not connect.  Clean up.
                scope_close(sock);
                continue;
            }

            scopeLogInfo("fd:%d connect to %s:%d is pending", sock, addrstr, port);
            trans->net.failure_reason = CONN_FAIL;

            trans->net.pending_connect = sock;
            break;  // replace w/continue for a shotgun start.
        }


        if (trans->type == CFG_UDP) {
            scopeLogInfo("fd:%d connect to %s:%d was successful", sock, addrstr, port);

            // connect on udp sockets normally succeeds immediately.
            trans->net.sock = placeDescriptor(sock, trans);
            if (trans->net.sock != -1) break;
        } else {
            DBG(NULL); // with non-blocking tcp sockets, we always expect -1
        }
    }

    return (trans->net.sock != -1);
}

static int
transportConnectFile(transport_t *t)
{
    // if stdout/stderr, set stream and skip everything else in the function.
    if (t->file.stdout) {
        t->file.stream = scope_stdout;
        return 1;
    } else if (t->file.stderr) {
        t->file.stream = scope_stderr;
        return 1;
    }

    int fd;
    fd = scope_open(t->file.path, O_CREAT|O_WRONLY|O_APPEND|O_CLOEXEC, 0666);
    if (fd == -1) {
        DBG("%s", t->file.path);
        transportDisconnect(t);
        return 0;
    }

    // Move this descriptor up out of the way
    if ((fd = placeDescriptor(fd, t)) == -1) {
        transportDisconnect(t);
        return 0;
    }

    // Needed because umask affects open permissions
    if (scope_fchmod(fd, 0666) == -1) {
        DBG("%d %s", fd, t->file.path);
    }

    FILE *f;
    if (!(f = scope_fdopen(fd, "a"))) {
        transportDisconnect(t);
        return 0;
    }
    t->file.stream = f;

    // Fully buffer the output unless we're told not to.
    // I expect line buffering to be useful when we're debugging crashes or
    // or if many applications are configured to write to the same files.
    int buf_mode = _IOFBF;
    switch (t->file.buf_policy) {
        case CFG_BUFFER_FULLY:
            buf_mode = _IOFBF;
            break;
        case CFG_BUFFER_LINE:
            buf_mode = _IOLBF;
            break;
        default:
            DBG("%d", t->file.buf_policy);
    }
    if (scope_setvbuf(t->file.stream, NULL, buf_mode, BUFSIZ)) {
        DBG(NULL);
    }

    return (t->file.stream != NULL);
}

#define EDGE_PATH_DOCKER "/var/run/appscope/appscope.sock"
#define EDGE_PATH_DEFAULT "/opt/cribl/state/appscope.sock"
#define READ_AND_WRITE (R_OK|W_OK)
static char*
edgePath(void){
    // 1) If EDGE_PATH_DOCKER can be accessed, return that.
    if (scope_access(EDGE_PATH_DOCKER, READ_AND_WRITE) == 0) {
        return scope_strdup(EDGE_PATH_DOCKER);
    }

    // 2) If CRIBL_HOME is defined and can be accessed,
    //    return $CRIBL_HOME/state/appscope.sock
    const char *cribl_home = getenv("CRIBL_HOME");
    if (cribl_home) {
        char *new_path = NULL;
        if (scope_asprintf(&new_path, "%s/%s", cribl_home, "state/appscope.sock") > 0) {
            if (scope_access(new_path, READ_AND_WRITE) == 0) {
                return new_path;
            }
            scope_free(new_path);
        }
    }

    // 3) If EDGE_PATH_DEFAULT can be accessed, return it
    if (scope_access(EDGE_PATH_DEFAULT, READ_AND_WRITE) == 0) {
        return scope_strdup(EDGE_PATH_DEFAULT);
    }

    return NULL;
}

int
transportConnect(transport_t *trans)
{
    if (!trans) return 1;

    // We're already connected.  Do nothing.
    if (!transportNeedsConnection(trans)) return 1;

    switch (trans->type) {
        case CFG_UDP:
        case CFG_TCP:
            if (!socketConnectIsPending(trans)) {
                // socketConnectionStart can directly connect (udp).
                // If it does, we're done.
                if (socketConnectionStart(trans)) return 1;
            }
            // Check to see if the a pending connection has been successful.
            return checkPendingSocketStatus(trans);
        case CFG_FILE:
            return transportConnectFile(trans);
        case CFG_EDGE:
            // Edge path needs to be recomputed on every connection attempt.
            if (trans->local.path) scope_free(trans->local.path);
            trans->local.path = edgePath();
            if (!trans->local.path) return 0;

            int pathlen = scope_strlen(trans->local.path);
            if (pathlen >= sizeof(trans->local.addr.sun_path)) return 0;

            scope_memset(&trans->local.addr, 0, sizeof(trans->local.addr));
            trans->local.addr.sun_family = AF_UNIX;
            scope_strncpy(trans->local.addr.sun_path, trans->local.path, pathlen);
            trans->local.addr_len = pathlen + sizeof(sa_family_t) + 1;

            // Keep going!  (no break or return here!)
            // CFG_EDGE uses CFG_UNIX's connection logic.
        case CFG_UNIX:
            if ((trans->local.sock = scope_socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
                DBG("%d %s", trans->local.sock, trans->local.path);
                return 0;
            }

            // Set close on exec
            int flags = scope_fcntl(trans->local.sock, F_GETFD, 0);
            if (scope_fcntl(trans->local.sock, F_SETFD, flags | FD_CLOEXEC) == -1) {
                DBG("%d %s", trans->local.sock, trans->local.path);
            }

            if (scope_connect(trans->local.sock, (const struct sockaddr *)&trans->local.addr,
                             trans->local.addr_len) == -1) {
                scopeLogInfo("fd:%d (%s) connect failed", trans->local.sock, trans->local.path);
                scope_close(trans->local.sock);
                trans->local.sock = -1;
                return 0;
            }

            // We have a connection
            scopeLogInfo("fd:%d connect successful", trans->local.sock);

            // Move this descriptor up out of the way
            trans->local.sock = placeDescriptor(trans->local.sock, trans);
            break;
        default:
            DBG(NULL);
    }

    return 1;
}

transport_t *
transportCreateTCP(const char *host, const char *port, unsigned int enable,
                      unsigned int validateserver, const char *cacertpath)
{
    transport_t* trans = NULL;

    if (!host || !port) return trans;

    trans = newTransport();
    if (!trans) return trans;

    trans->type = CFG_TCP;
    trans->net.sock = -1;
    trans->net.pending_connect = -1;
    trans->net.host = scope_strdup(host);
    trans->net.port = scope_strdup(port);
    trans->net.tls.enable = enable;
    trans->net.tls.validateserver = validateserver;
    trans->net.tls.cacertpath = (cacertpath) ? scope_strdup(cacertpath) : NULL;

    if (!trans->net.host || !trans->net.port) {
        DBG(NULL);
        transportDestroy(&trans);
        return trans;
    }

    transportConnect(trans);

    return trans;
}

transport_t*
transportCreateUdp(const char* host, const char* port)
{
    transport_t* t = NULL;

    if (!host || !port) return t;

    t = newTransport();
    if (!t) return t;

    t->type = CFG_UDP;
    t->net.sock = -1;
    t->net.pending_connect = -1;
    t->net.host = scope_strdup(host);
    t->net.port = scope_strdup(port);

    if (!t->net.host || !t->net.port) {
        DBG(NULL);
        transportDestroy(&t);
        return t;
    }

    transportConnect(t);

    return t;
}

transport_t*
transportCreateFile(const char* path, cfg_buffer_t buf_policy)
{
    transport_t *t;

    if (!path) return NULL;
    t = newTransport();
    if (!t) return NULL; 

    t->type = CFG_FILE;
    t->file.path = scope_strdup(path);
    if (!t->file.path) {
        DBG("%s", path);
        transportDestroy(&t);
        return t;
    }
    t->file.buf_policy = buf_policy;

    // See if path is "stdout" or "stderr"
    t->file.stdout = !scope_strcmp(path, "stdout");
    t->file.stderr = !scope_strcmp(path, "stderr");

    transportConnect(t);

    return t;
}

transport_t *
transportCreateUnix(const char *path)
{
    transport_t *trans = NULL;

    if (!path) goto err;

    int pathlen = scope_strlen(path);
    if (pathlen >= sizeof(trans->local.addr.sun_path)) goto err;

    if (!(trans = newTransport())) goto err;

    trans->type = CFG_UNIX;
    trans->local.sock = -1;
    if (!(trans->local.path = scope_strdup(path))) goto err;

    scope_memset(&trans->local.addr, 0, sizeof(trans->local.addr));
    trans->local.addr.sun_family = AF_UNIX;
    scope_strncpy(trans->local.addr.sun_path, path, pathlen);
    trans->local.addr_len = pathlen + sizeof(sa_family_t);
    if (path[0] == '@') {
        // The socket is abstract
        trans->local.addr.sun_path[0] = 0;
    } else {
        // Abstract socket addresses don't include a trailing null
        // delimiter but filesystem sockets do.
        trans->local.addr_len += 1; 
    }

    transportConnect(trans);

    return trans;

err:
    DBG("%s %p", path, trans);
    transportDestroy(&trans);
    return trans;
}

transport_t*
transportCreateEdge(void)
{
    transport_t *trans = NULL;

    if (!(trans = newTransport())) goto err;

    trans->type = CFG_EDGE;
    trans->local.sock = -1;
    trans->local.path = NULL;

    transportConnect(trans);

    return trans;

err:
    DBG("%p", trans);
    transportDestroy(&trans);
    return trans;

}

transport_t*
transportCreateSyslog(void)
{
    transport_t* t = scope_calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    t->type = CFG_SYSLOG;

    return t;
}

transport_t*
transportCreateShm()
{
    transport_t* t = scope_calloc(1, sizeof(transport_t));
    if (!t) {
        DBG(NULL);
        return NULL;
    }

    t->type = CFG_SHM;

    return t;
}

void
transportDestroy(transport_t **transport)
{
    if (!transport || !*transport) return;

    transport_t *trans = *transport;
    switch (trans->type) {
        case CFG_UDP:
        case CFG_TCP:
            transportDisconnect(trans);
            if (trans->net.host) scope_free(trans->net.host);
            if (trans->net.port) scope_free(trans->net.port);
            if (trans->net.tls.cacertpath) scope_free(trans->net.tls.cacertpath);
            freeAddressList(trans);
            break;
        case CFG_UNIX:
        case CFG_EDGE:
            if (trans->local.path) scope_free(trans->local.path);
            transportDisconnect(trans);
            break;
        case CFG_FILE:
            if (trans->file.path) scope_free(trans->file.path);
            if (!trans->file.stdout && !trans->file.stderr) {
                // if stdout/stderr, we didn't open stream, so don't close it
                if (trans->file.stream) scope_fclose(trans->file.stream);
            }
            break;
        case CFG_SYSLOG:
            break;
        case CFG_SHM:
            break;
        default:
            DBG("%d", trans->type);
    }
    scope_free(trans);
    *transport = NULL;
}

static int
tcpSendPlain(transport_t *trans, const char *msg, size_t len)
{
    if (!trans || transportNeedsConnection(trans)) return -1;

    int flags = 0;
#ifdef __linux__
    flags |= MSG_NOSIGNAL;
#endif

    size_t bytes_to_send = len;
    size_t bytes_sent = 0;
    int rc;

    while (bytes_to_send > 0) {
        if (g_ismusl == TRUE) {
            rc = scope_syscall(SYS_sendto, trans->net.sock, &msg[bytes_sent], bytes_to_send, flags, NULL, 0);
        } else {
            rc = scope_send(trans->net.sock, &msg[bytes_sent], bytes_to_send, flags);
        }

        if (rc <= 0) break;

        if (rc != bytes_to_send) {
            DBG("rc = %d, bytes_to_send = %zu", rc, bytes_to_send);
        }

        bytes_sent += rc;
        bytes_to_send -= rc;
    }

    if (rc < 0) {
        switch (scope_errno) {
        case EBADF:
        case EPIPE:
            DBG(NULL);
            transportDisconnect(trans);
            transportConnect(trans);
            return -1;
        default:
            DBG(NULL);
        }
    }
    return 0;
}

static int
tcpSendTls(transport_t *trans, const char *msg, size_t len)
{
    if (!trans || transportNeedsConnection(trans)) return -1;

    size_t bytes_to_send = len;
    size_t bytes_sent = 0;
    int rc = 0;
    int err = 0;

    while (bytes_to_send > 0) {

        rc = 0;
        ERR_clear_error(); // to make SSL_get_error reliable
        rc = SCOPE_SSL_write(trans->net.tls.ssl, &msg[bytes_sent], bytes_to_send);
        if (rc <= 0) {
            err = SSL_get_error(trans->net.tls.ssl, rc);
        }

        if (rc <= 0) {
            DBG("%d", err);
            transportDisconnect(trans);
            transportConnect(trans);
            return -1;
        }

        if (rc != bytes_to_send) {
            DBG("rc = %d, bytes_to_send = %zu", rc, bytes_to_send);
        }

        bytes_sent += rc;
        bytes_to_send -= rc;
    }

    return 0;
}

int
transportSend(transport_t *trans, const char *msg, size_t len)
{
    if (!trans || !msg) return -1;

    switch (trans->type) {
        case CFG_UDP:
            if (trans->net.sock != -1) {
                int rc;
                if (g_ismusl == TRUE) {
                    rc = scope_syscall(SYS_sendto, trans->net.sock, msg, len, 0, NULL, 0);
                } else {
                    rc = scope_send(trans->net.sock, msg, len, 0);
                }

                if (rc < 0) {
                    switch (scope_errno) {
                    case EBADF:
                        DBG(NULL);
                        transportDisconnect(trans);
                        transportConnect(trans);
                        return -1;
                    case EWOULDBLOCK:
                        DBG(NULL);
                        break;
                    default:
                        DBG(NULL);
                    }
                }
            }
            break;
        case CFG_TCP:
            if (trans->net.tls.enable) {
                return tcpSendTls(trans, msg, len);
            } else {
                return tcpSendPlain(trans, msg, len);
            }
            break;
        case CFG_FILE:
            if (trans->file.stream) {
                size_t msg_size = len;
                int bytes = scope_fwrite(msg, 1, msg_size, trans->file.stream);
                if (bytes != msg_size) {
                    if (scope_errno == EBADF) {
                        DBG("%d %d", bytes, msg_size);
                        transportDisconnect(trans);
                        transportConnect(trans);
                        return -1;
                    }
                    DBG("%d %d", bytes, msg_size);
                    return -1;
                }
            }
            break;
        case CFG_UNIX:
        case CFG_EDGE:
            if (trans->local.sock != -1) {
                int flags = 0;
#ifdef __linux__
                flags |= MSG_NOSIGNAL;
#endif
                int rc;
                if (g_ismusl == TRUE) {
                    rc = scope_syscall(SYS_sendto, trans->local.sock, msg, len, flags, NULL, 0);
                } else {
                    rc = scope_send(trans->local.sock, msg, len, flags);
                }

                if (rc < 0) {
                    switch (scope_errno) {
                    case EBADF:
                    case EPIPE:
                        DBG(NULL);
                        transportDisconnect(trans);
                        transportConnect(trans);
                        return -1;
                    case EWOULDBLOCK:
                        DBG(NULL);
                        break;
                    default:
                        DBG(NULL);
                    }
                }
            }
            break;
        case CFG_SYSLOG:
        case CFG_SHM:
            return -1;
        default:
            DBG("%d", trans->type);
            return -1;
    }
     return 0;
}

int
transportFlush(transport_t* t)
{
    if (!t) return -1;

    switch (t->type) {
        case CFG_UDP:
        case CFG_TCP:
            break;
        case CFG_FILE:
            if (scope_fflush(t->file.stream) == EOF) {
                DBG(NULL);
            }
            break;
        case CFG_UNIX:
        case CFG_EDGE:
        case CFG_SYSLOG:
        case CFG_SHM:
            return -1;
        default:
            DBG("%d", t->type);
            return -1;
    }
    return 0;
}

uint64_t
transportConnectAttempts(transport_t* t)
{
    return t->net.connect_attempts;
}

net_fail_t
transportFailureReason(transport_t *t)
{
    return t->net.failure_reason;
}

