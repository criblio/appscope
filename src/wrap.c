#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/poll.h>

#ifdef __linux__
#include <sys/prctl.h>
#ifdef __GO__
#include <asm/prctl.h>
#endif
#endif
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <libgen.h>
#include <sys/resource.h>
#include <setjmp.h>

#include "atomic.h"
#include "bashmem.h"
#include "cfg.h"
#include "cfgutils.h"
#include "com.h"
#include "dbg.h"
#include "dns.h"
#include "fn.h"
#include "os.h"
#include "plattime.h"
#include "report.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include "state.h"
#include "utils.h"
#include "wrap.h"
#include "runtimecfg.h"
#include "javaagent.h"
#include "inject.h"
#include "../contrib/libmusl/musl.h"

#define SSL_FUNC_READ "SSL_read"
#define SSL_FUNC_WRITE "SSL_write"

static thread_timing g_thread = {0};
static config_t *g_staticfg = NULL;
static log_t *g_prevlog = NULL;
static mtc_t *g_prevmtc = NULL;
static ctl_t *g_prevctl = NULL;
static bool g_replacehandler = FALSE;
static const char *g_cmddir;
static list_t *g_nsslist;
static uint64_t reentrancy_guard = 0ULL;
static rlim_t g_max_fds = 0;
static bool g_ismusl = FALSE;

typedef int (*ssl_rdfunc_t)(SSL *, void *, int);
typedef int (*ssl_wrfunc_t)(SSL *, const void *, int);

__thread int g_getdelim = 0;
__thread int g_ssl_fd = -1;

// Forward declaration
static void *periodic(void *);
static void doConfig(config_t *);
static void threadNow(int);
static void uv__read_hook(void *);

#ifdef __linux__
extern int arch_prctl(int, unsigned long);
extern unsigned long scope_fs;

extern void initGoHook(elf_buf_t*);

static got_list_t hook_list[] = {
    {"SSL_read", SSL_read, &g_fn.SSL_read},
    {"SSL_write", SSL_write, &g_fn.SSL_write},
    {NULL, NULL, NULL}
};

static got_list_t inject_hook_list[] = {
    {"sigaction",   NULL, &g_fn.sigaction},
    {"open",        NULL, &g_fn.open},
    {"openat",      NULL, &g_fn.openat},
    {"fopen",       NULL, &g_fn.fopen},
    {"freopen",     NULL, &g_fn.freopen},
    {"nanosleep",   NULL, &g_fn.nanosleep},
    {"select",      NULL, &g_fn.select},
    {"sigsuspend",  NULL, &g_fn.sigsuspend},
    {"epoll_wait",  NULL, &g_fn.epoll_wait},
    {"poll",        NULL, &g_fn.poll},
    {"pause",       NULL, &g_fn.pause},
    {"sigwaitinfo", NULL, &g_fn.sigwaitinfo},
    {"sigtimedwait", NULL, &g_fn.sigtimedwait},
    {"epoll_pwait", NULL, &g_fn.epoll_pwait},
    {"ppoll",       NULL, &g_fn.ppoll},
    {"pselect",     NULL, &g_fn.pselect},
    {"msgsnd",      NULL, &g_fn.msgsnd},
    {"msgrcv",      NULL, &g_fn.msgrcv},
    {"semop",       NULL, &g_fn.semop},
    {"semtimedop",  NULL, &g_fn.semtimedop},
    {"clock_nanosleep", NULL, &g_fn.clock_nanosleep},
    {"usleep", NULL, &g_fn.usleep},
    {"io_getevents", NULL, &g_fn.io_getevents},
    {"open64", NULL, &g_fn.open64},
    {"openat64", NULL, &g_fn.openat64},
    {"__open_2", NULL, &g_fn.__open_2},
    {"__open64_2", NULL, &g_fn.__open64_2},
    {"__openat_2", NULL, &g_fn.__openat_2},
    {"creat64", NULL, &g_fn.creat64},
    {"fopen64", NULL, &g_fn.fopen64},
    {"freopen64", NULL, &g_fn.freopen64},
    {"pread64", NULL, &g_fn.pread64},
    {"preadv", NULL, &g_fn.preadv},
    {"preadv2", NULL, &g_fn.preadv2},
    {"preadv64v2", NULL, &g_fn.preadv64v2},
    {"__pread_chk", NULL, &g_fn.__pread_chk},
    {"__read_chk", NULL, &g_fn.__read_chk},
    {"__fread_unlocked_chk", NULL, &g_fn.__fread_unlocked_chk},
    {"pwrite64", NULL, &g_fn.pwrite64},
    {"pwritev", NULL, &g_fn.pwritev},
    {"pwritev64", NULL, &g_fn.pwritev64},
    {"pwritev2", NULL, &g_fn.pwritev2},
    {"pwritev64v2", NULL, &g_fn.pwritev64v2},
    {"lseek64", NULL, &g_fn.lseek64},
    {"fseeko64", NULL, &g_fn.fseeko64},
    {"ftello64", NULL, &g_fn.ftello64},
    {"statfs64", NULL, &g_fn.statfs64},
    {"fstatfs64", NULL, &g_fn.fstatfs64},
    {"fsetpos64", NULL, &g_fn.fsetpos64},
    {"__xstat", NULL, &g_fn.__xstat},
    {"__xstat64", NULL, &g_fn.__xstat64},
    {"__lxstat", NULL, &g_fn.__lxstat},
    {"__lxstat64", NULL, &g_fn.__lxstat64},
    {"__fxstat", NULL, &g_fn.__fxstat},
    {"__fxstatat", NULL, &g_fn.__fxstatat},
    {"__fxstatat64", NULL, &g_fn.__fxstatat64},
    {"statfs", NULL, &g_fn.statfs},
    {"fstatfs", NULL, &g_fn.fstatfs},
    {"statvfs", NULL, &g_fn.statvfs},
    {"statvfs64", NULL, &g_fn.statvfs64},
    {"fstatvfs", NULL, &g_fn.fstatvfs},
    {"fstatvfs64", NULL, &g_fn.fstatvfs64},
    {"access", NULL, &g_fn.access},
    {"faccessat", NULL, &g_fn.faccessat},
    {"gethostbyname_r", NULL, &g_fn.gethostbyname_r},
    {"gethostbyname2_r", NULL, &g_fn.gethostbyname2_r},
    {"fstatat", NULL, &g_fn.fstatat},
    {"prctl", NULL, &g_fn.prctl},
    {"execve", NULL, &g_fn.execve},
    {"syscall", NULL, &g_fn.syscall},
    {"sendfile", NULL, &g_fn.sendfile},
    {"sendfile64", NULL, &g_fn.sendfile64},
    {"SSL_read", NULL, &g_fn.SSL_read},
    {"SSL_write", NULL, &g_fn.SSL_write},
    {"gnutls_record_recv", NULL, &g_fn.gnutls_record_recv},
    {"gnutls_record_recv_early_data", NULL, &g_fn.gnutls_record_recv_early_data},
    {"gnutls_record_recv_packet", NULL, &g_fn.gnutls_record_recv_packet},
    {"gnutls_record_recv_seq", NULL, &g_fn.gnutls_record_recv_seq},
    {"gnutls_record_send", NULL, &g_fn.gnutls_record_send},
    {"gnutls_record_send2", NULL, &g_fn.gnutls_record_send2},
    {"gnutls_record_send_early_data", NULL, &g_fn.gnutls_record_send_early_data},
    {"gnutls_record_send_range", NULL, &g_fn.gnutls_record_send_range},
    {"dlopen", NULL, &g_fn.dlopen},
    {"_exit", NULL, &g_fn._exit},
    {"close", NULL, &g_fn.close},
    {"fclose", NULL, &g_fn.fclose},
    {"fcloseall", NULL, &g_fn.fcloseall},
    {"lseek", NULL, &g_fn.lseek},
    {"fseek", NULL, &g_fn.fseek},
    {"fseeko", NULL, &g_fn.fseeko},
    {"ftell", NULL, &g_fn.ftell},
    {"ftello", NULL, &g_fn.ftello},
    {"rewind", NULL, &g_fn.rewind},
    {"fsetpos", NULL, &g_fn.fsetpos},
    {"fgetpos", NULL, &g_fn.fgetpos},
    {"fgetpos64", NULL, &g_fn.fgetpos64},
    {"write", NULL, &g_fn.write},
    {"pwrite", NULL, &g_fn.pwrite},
    {"writev", NULL, &g_fn.writev},
    {"fwrite", NULL, &g_fn.fwrite},
    {"puts", NULL, &g_fn.puts},
    {"putchar", NULL, &g_fn.putchar},
    {"fputs", NULL, &g_fn.fputs},
    {"fputs_unlocked", NULL, &g_fn.fputs_unlocked},
    {"read", NULL, &g_fn.read},
    {"readv", NULL, &g_fn.readv},
    {"pread", NULL, &g_fn.pread},
    {"fread", NULL, &g_fn.fread},
    {"__fread_chk", NULL, &g_fn.__fread_chk},
    {"fgets", NULL, &g_fn.fgets},
    {"__fgets_chk", NULL, &g_fn.__fgets_chk},
    {"fgets_unlocked", NULL, &g_fn.fgets_unlocked},
    {"__fgetws_chk", NULL, &g_fn.__fgetws_chk},
    {"fgetws", NULL, &g_fn.fgetws},
    {"fgetwc", NULL, &g_fn.fgetwc},
    {"fgetc", NULL, &g_fn.fgetc},
    {"fputc", NULL, &g_fn.fputc},
    {"fputc_unlocked", NULL, &g_fn.fputc_unlocked},
    {"putwc", NULL, &g_fn.putwc},
    {"fputwc", NULL, &g_fn.fputwc},
    {"fscanf", NULL, &g_fn.fscanf},
    {"getline", NULL, &g_fn.getline},
    {"getdelim", NULL, &g_fn.getdelim},
    {"__getdelim", NULL, &g_fn.__getdelim},
    {"fcntl", NULL, &g_fn.fcntl},
    {"fcntl64", NULL, &g_fn.fcntl64},
    {"dup", NULL, &g_fn.dup},
    {"dup2", NULL, &g_fn.dup2},
    {"dup3", NULL, &g_fn.dup3},
    {"vsyslog", NULL, &g_fn.vsyslog},
    {"fork", NULL, &g_fn.fork},
    {"socket", NULL, &g_fn.socket},
    {"shutdown", NULL, &g_fn.shutdown},
    {"listen", NULL, &g_fn.listen},
    {"accept", NULL, &g_fn.accept},
    {"accept4", NULL, &g_fn.accept4},
    {"bind", NULL, &g_fn.bind},
    {"connect", NULL, &g_fn.connect},
    {"send", NULL, &g_fn.send},
    {"sendto", NULL, &g_fn.sendto},
    {"sendmsg", NULL, &g_fn.sendmsg},
    {"recv", NULL, &g_fn.recv},
    {"recvfrom", NULL, &g_fn.recvfrom},
    {"recvmsg", NULL, &g_fn.recvmsg},
    {NULL, NULL, NULL}
};

typedef struct
{
    char *in_symbol;
    void *out_addr;
    int   after_scope;
} param_t;

// When used with dl_iterate_phdr(), this has a similar result to
// scope_dlsym(RTLD_NEXT, ).  But, unlike scope_dlsym(RTLD_NEXT, )
// it will never return a symbol in our library.  This is
// particularly useful for finding symbols in shared libraries
// that are dynamically loaded after our constructor has run.
// Not to point fingers, but I'm looking at you python.
//
static void *scope_dlsym(void *, const char *, void *);

static int
findSymbol(struct dl_phdr_info *info, size_t size, void *data)
{
    param_t* param = (param_t*)data;

    // Don't bother looking inside libraries until after we've seen our library.
    if (!param->after_scope) {
        param->after_scope = (strstr(info->dlpi_name, "libscope") != NULL) ||
            (strstr(info->dlpi_name, "/proc/") != NULL);
        return 0;
    }

    // Now start opening libraries and looking for param->in_symbol
    void *handle = g_fn.dlopen(info->dlpi_name, RTLD_NOW);
    if (!handle) return 0;
    void *addr = dlsym(handle, param->in_symbol);
    dlclose(handle);

    // if we don't find addr, keep going
    if (!addr)  return 0;

    // We found an addr, and it's not in our library!  Return it!
    param->out_addr = addr;
    return 1;
}

#define WRAP_CHECK(func, rc)                                           \
    if (g_fn.func == NULL ) {                                          \
       if (!g_ctl) {                                                   \
         if ((g_fn.func = scope_dlsym(RTLD_NEXT, #func, func)) == NULL) {  \
             scopeLogError("ERROR: "#func":NULL\n");         \
             return rc;                                                \
         }                                                             \
       } else {                                                        \
        param_t param = {.in_symbol = #func, .out_addr = NULL,         \
                         .after_scope = FALSE};                        \
        if (!dl_iterate_phdr(findSymbol, &param)) {                    \
            scopeLogError("ERROR: "#func":NULL\n");          \
            return rc;                                                 \
        }                                                              \
        g_fn.func = param.out_addr;                                    \
       }                                                               \
    }                                                                  \
    doThread();

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
       if (!g_ctl) {                                                   \
         if ((g_fn.func = scope_dlsym(RTLD_NEXT, #func, func)) == NULL) {  \
             scopeLogError("ERROR: "#func":NULL\n");         \
             return;                                                   \
         }                                                             \
       } else {                                                        \
        param_t param = {.in_symbol = #func, .out_addr = NULL,         \
                         .after_scope = FALSE};                        \
        if (!dl_iterate_phdr(findSymbol, &param)) {                    \
            scopeLogError("ERROR: "#func":NULL\n");          \
            return;                                                    \
        }                                                              \
        g_fn.func = param.out_addr;                                    \
      }                                                                \
    }                                                                  \
    doThread();

#define SYMBOL_LOADED(func) ({                                         \
    int retval;                                                        \
    if (g_fn.func == NULL) {                                           \
        param_t param = {.in_symbol = #func, .out_addr = NULL,         \
                         .after_scope = FALSE};                        \
        if (dl_iterate_phdr(findSymbol, &param)) {                     \
            g_fn.func = param.out_addr;                                \
        }                                                              \
    }                                                                  \
    retval = (g_fn.func != NULL);                                      \
    retval;                                                            \
})

#else

#define WRAP_CHECK(func, rc)                                           \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLogError("ERROR: "#func":NULL\n");          \
            return rc;                                                 \
       }                                                               \
    }                                                                  \
    doThread();

#define WRAP_CHECK_VOID(func)                                          \
    if (g_fn.func == NULL ) {                                          \
        if ((g_fn.func = dlsym(RTLD_NEXT, #func)) == NULL) {           \
            scopeLogError("ERROR: "#func":NULL\n");          \
            return;                                                    \
       }                                                               \
    }                                                                  \
    doThread();

#define SYMBOL_LOADED(func) ({                                         \
    int retval;                                                        \
    if (g_fn.func == NULL) {                                           \
        g_fn.func = dlsym(RTLD_NEXT, #func);                           \
    }                                                                  \
    retval = (g_fn.func != NULL);                                      \
    retval;                                                            \
})

#endif // __linux__


static void
freeNssEntry(void *data)
{
    if (!data) return;
    nss_list *nssentry = data;

    if (!nssentry) return;
    if (nssentry->ssl_methods) free(nssentry->ssl_methods);
    if (nssentry->ssl_int_methods) free(nssentry->ssl_int_methods);
    free(nssentry);
}

static time_t
fileModTime(const char *path)
{
    int fd;
    struct stat statbuf;

    if (!path) return 0;

    if (!g_fn.open || !g_fn.close) {
        return 0;
    }

    if ((fd = g_fn.open(path, O_RDONLY)) == -1) return 0;
    
    if (fstat(fd, &statbuf) < 0) {
        g_fn.close(fd);
        return 0;
    }

    g_fn.close(fd);
    // STATMODTIME from os.h as timespec names are different between OSs
    return STATMODTIME(statbuf);
}

static void
remoteConfig()
{
    int timeout;
    struct pollfd fds;
    int rc, success, numtries;
    FILE *fs;
    char buf[1024];
    char path[PATH_MAX];
    
    // to be clear; a 1ms timeout
    timeout = 1;
    memset(&fds, 0x0, sizeof(fds));

/*
    Setting fds.events = 0 to neuter ability to process remote
    commands... until this is function is reworked to be TLS-friendly.

    cfg_transport_t ttype = ctlTransportType(g_ctl, CFG_CTL);
    if ((ttype == (cfg_transport_t)-1) || (ttype == CFG_FILE) ||
        (ttype ==  CFG_SYSLOG) || (ttype == CFG_SHM)) {
        fds.events = 0;
    } else {
        fds.events = POLLIN;
    }
*/

    fds.events = 0;
    fds.fd = ctlConnection(g_ctl, CFG_CTL);

    rc = g_fn.poll(&fds, 1, timeout);

    /*
     * Error from poll;
     * doing this separtately in order to count errors. Necessary?
     */
    if (rc < 0) {
        DBG(NULL);
        return;
    }

    /*
     * Timeout or no read data?
     * We can track exceptions where revents != POLLIN. Necessary?
     */
    if ((rc == 0) || (fds.revents == 0) || ((fds.revents & POLLIN) == 0) ||
        ((fds.revents & POLLHUP) != 0) || ((fds.revents & POLLNVAL) != 0)) return;

    snprintf(path, sizeof(path), "/tmp/cfg.%d", g_proc.pid);
    if ((fs = g_fn.fopen(path, "a+")) == NULL) {
        DBG(NULL);
        scopeLogError("ERROR: remoteConfig:fopen");
        return;
    }

    success = rc = errno = numtries = 0;
    do {
        numtries++;
        rc = g_fn.recv(fds.fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (rc <= 0) {
            // Something has happened to our connection
            ctlDisconnect(g_ctl, CFG_CTL);
            break;
        }

        if (g_fn.fwrite(buf, rc, (size_t)1, fs) <= 0) {
            DBG(NULL);
            break;
        } else {
            /*
             * We are done if we get end of msg (EOM) or if we've read more 
             * than expected and never saw an EOM.
             * We are only successful if we've had no errors or disconnects
             * and we receive a viable EOM.
             */
            // EOM
            if (strchr((const char *)buf, '\n') != NULL) {
                success = 1;
                break;
            } else {
                // No EOM after more than enough tries, bail out
                if (numtries > MAXTRIES) {
                    break;
                }
            }
        }
    } while (1);

    if (success == 1) {
        char *cmd;
        struct stat sb;
        request_t *req;

        if (fflush(fs) != 0) DBG(NULL);
        rewind(fs);
        if (lstat(path, &sb) == -1) {
            sb.st_size = DEFAULT_CONFIG_SIZE;
        }

        cmd = calloc(1, sb.st_size);
        if (!cmd) {
            g_fn.fclose(fs);
            unlink(path);
            cmdSendInfoStr(g_ctl, "Error in receive from stream.  Memory error in scope receive.");
            return;
        }
        
        if (g_fn.fread(cmd, sb.st_size, 1, fs) == 0) {
            g_fn.fclose(fs);
            unlink(path);
            free(cmd);
            cmdSendInfoStr(g_ctl, "Error in receive from stream.  Read error in scope.");
            return;
        }
        
        req = cmdParse((const char*)cmd);
        if (req) {
            cJSON* body = NULL;
            switch (req->cmd) {
                case REQ_PARSE_ERR:
                case REQ_MALFORMED:
                case REQ_UNKNOWN:
                case REQ_PARAM_ERR:
                    // Nothing to do here.  Req is not well-formed.
                    break;
                case REQ_SET_CFG:
                    if (req->cfg) {
                        // Apply the config
                        doConfig(req->cfg);
                        g_staticfg = req->cfg;
                    } else {
                        DBG(NULL);
                    }
                    break;
                case REQ_GET_CFG:
                    // construct a response representing our current config
                    body = jsonConfigurationObject(g_staticfg);
                    break;
                case REQ_GET_DIAG:
                    // Not implemented yet.
                    break;
                case REQ_BLOCK_PORT:
                    // Assign new value for port blocking
                    g_cfg.blockconn = req->port;
                    break;
                case REQ_SWITCH:
                    switch (req->action) {
                        case URL_REDIRECT_ON:
                            g_cfg.urls = 1;
                            break;
                        case URL_REDIRECT_OFF:
                             g_cfg.urls = 0;
                            break;
                        default:
                            DBG("%d", req->action);
                    }
                    break;
                case REQ_ADD_PROTOCOL:
                    // define a new protocol
                    addProtocol(req);
                    break;
                case REQ_DEL_PROTOCOL:
                    // remove a protocol
                    delProtocol(req);
                    break;
            default:
                    DBG(NULL);
            }
            
            cmdSendResponse(g_ctl, req, body);
            destroyReq(&req);
        } else {
            cmdSendInfoStr(g_ctl, "Error in receive from stream.  Memory error in scope parsing.");
        }

        free(cmd);
    } else {
        cmdSendInfoStr(g_ctl, "Error in receive from stream.  Scope receive retries exhausted.");
    }

    g_fn.fclose(fs);
    unlink(path);
}

static void
doConfig(config_t *cfg)
{
    // Save the current objects to get cleaned up on the periodic thread
    g_prevmtc = g_mtc;
    g_prevlog = g_log;
    g_prevctl = g_ctl;

    if (cfgLogStream(cfg)) {
        cfgLogStreamDefault(cfg);
    }

    g_thread.interval = cfgMtcPeriod(cfg);
    setReportingInterval(cfgMtcPeriod(cfg));
    if (!g_thread.startTime) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        g_thread.startTime = tv.tv_sec + g_thread.interval;
    }

    setVerbosity(cfgMtcVerbosity(cfg));
    g_cmddir = cfgCmdDir(cfg);
    g_sendprocessstart = cfgSendProcessStartMsg(cfg);

    g_log = initLog(cfg);
    g_mtc = initMtc(cfg);
    g_ctl = initCtl(cfg);

    if (cfgLogStream(cfg)) {
        singleChannelSet(g_ctl, g_mtc);
    }

    // Send a process start message to report our *new* configuration.
    // Only needed if we're connected.  If we're not connected, doConnection()
    // will send the process start message when we ultimately connect.
    if (!ctlNeedsConnection(g_ctl, CFG_CTL)) {
        reportProcessStart(g_ctl, FALSE, CFG_WHICH_MAX);
    }

    // Disconnect the old interfaces that were just replaced
    mtcDisconnect(g_prevmtc);
    logDisconnect(g_prevlog);
    ctlStopAggregating(g_prevctl);
    ctlFlush(g_prevctl);
    ctlDisconnect(g_prevctl, CFG_CTL);
}

// Process dynamic config change if they are available
static int
dynConfig(void)
{
    FILE *fs;
    time_t now;
    char path[PATH_MAX];
    static time_t modtime = 0;

    snprintf(path, sizeof(path), "%s/%s.%d", g_cmddir, DYN_CONFIG_PREFIX, g_proc.pid);

    // Is there a command file for this pid
    if (osIsFilePresent(g_proc.pid, path) == -1) return 0;

    // Have we already processed this file?
    now = fileModTime(path);
    if (now == modtime) {
        // Been there, try to remove the file and we're done
        unlink(path);
        return 0;
    }

    modtime = now;

    // Open the command file
    if ((fs = g_fn.fopen(path, "r")) == NULL) return -1;

    // Modify the static config from the command file
    cfgProcessCommands(g_staticfg, fs);

    // Apply the config
    doConfig(g_staticfg);

    g_fn.fclose(fs);
    unlink(path);
    return 0;
}

static void
threadNow(int sig)
{
    static uint64_t serialize;

    if (!atomicCasU64(&serialize, 0ULL, 1ULL)) return;

    // Create one thread at most
    if (g_thread.once == TRUE) {
        if (!atomicCasU64(&serialize, 1ULL, 0ULL)) DBG(NULL);
        return;
    }

    osTimerStop();

    if (g_fn.pthread_create &&
        (g_fn.pthread_create(&g_thread.periodicTID, NULL, periodic, NULL) != 0)) {
        scopeLogError("ERROR: threadNow:pthread_create");
        if (!atomicCasU64(&serialize, 1ULL, 0ULL)) DBG(NULL);
        return;
    }

    g_thread.once = TRUE;

    // Restore a handler if one exists
    if ((g_replacehandler == TRUE) && (g_thread.act != NULL)) {
        struct sigaction oldact;
        if (g_fn.sigaction) {
            g_fn.sigaction(SIGUSR2, g_thread.act, &oldact);
            g_thread.act = NULL;
        }
    }

    if (!atomicCasU64(&serialize, 1ULL, 0ULL)) DBG(NULL);
}

/*
 * This is not self evident.
 * There are some apps that check to see if only one thread exists when
 * they start and then exit if that is the case. The first place we see
 * this is with Chromium. Several apps use Chromium, including
 * Chrome, Slack and more.
 *
 * There are other processes that don't work if a thread
 * has been created before the application starts. We've
 * seen this in some bash scripts.
 *
 * The resolution is to delay the start of the thread until
 * an app has completed its configuration. In the case of
 * short lived procs, the thread never starts and is
 * not needed.
 *
 * Simple enough. Normally, we'd just start a timer and
 * create the thread when the timer expires. However, it
 * turns out that a timer either creates a thread to
 * handle the expiry or delivers a signal. The use
 * of a thread causes the same problem we're trying
 * to avoid. The use of a signal is problematic
 * because a number of apps install their own handlers.
 * When we install a handler in the library constructor
 * it is replaced when the app starts.
 *
 * This seems to work:
 * If we start a timer to deliver a signal on expiry
 * and also interpose sigaction we can make this work.
 * We use sigaction to install our handler. Then, we
 * interpose sigaction, look for our signal and
 * ensure that our handler will run in the presence
 * of an application installed handler.
 *
 * This creates the situation where the thread is
 * not started until after the app starts and is
 * past it's own init. We no longer need to rely
 * on the thread being created when an interposed
 * function is called. For now, we are leaving
 * the check for the thread in each interposed
 * function as a back up in case the timer has
 * an issue of some sort.
 */
static void
threadInit()
{
    // for debugging... if SCOPE_NO_SIGNAL is defined, then don't create
    // a signal handler, nor a timer to send a signal.
    if (getenv("SCOPE_NO_SIGNAL")) return;

    if (osThreadInit(threadNow, g_thread.interval) == FALSE) {
        scopeLogError("ERROR: threadInit:osThreadInit");
    }
}

static void
doThread()
{
    /*
     * If we try to start the perioidic thread before the constructor
     * is executed and our config is not set, we are able to start the
     * thread too early. Some apps, most notably Chrome, check to
     * ensure that no extra threads are created before it is fully
     * initialized. This check is intended to ensure that we don't
     * start the thread until after we have our config.
     */
    if (!g_ctl) return;

    // Create one thread at most
    if (g_thread.once == TRUE) return;

    /*
     * g_thread.startTime is the start time, set in the constructor.
     * This is put in place to work around one of the Chrome sandbox limits.
     * Shouldn't hurt anything else.
     */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    if (tv.tv_sec >= g_thread.startTime) {
        threadNow(0);
    }
}

static void
stopTimer(void)
{
    // if we are in the constructor, do nothing
    if (!g_ctl) return;

    osTimerStop();
    threadNow(0);
}

// Return process specific CPU usage in microseconds
static long long
doGetProcCPU() {
    struct rusage ruse;
    
    if (getrusage(RUSAGE_SELF, &ruse) != 0) {
        return (long long)-1;
    }

    return
        (((long long)ruse.ru_utime.tv_sec + (long long)ruse.ru_stime.tv_sec) * 1000 * 1000) +
        ((long long)ruse.ru_utime.tv_usec + (long long)ruse.ru_stime.tv_usec);
}

static void
setProcId(proc_id_t *proc)
{
    if (!proc) return;

    proc->pid = getpid();
    proc->ppid = getppid();
    if (gethostname(proc->hostname, sizeof(proc->hostname)) != 0) {
        scopeLogError("ERROR: gethostname");
    }
    osGetProcname(proc->procname, sizeof(proc->procname));

    // free old value of cmd, if an old value exists
    if (proc->cmd) free(proc->cmd);
    proc->cmd = NULL;
    osGetCmdline(proc->pid, &proc->cmd);

    if (proc->hostname && proc->procname && proc->cmd) {
        // limit amount of cmd used in id
        int cmdlen = strlen(proc->cmd);
        char *ptr = (cmdlen < DEFAULT_CMD_SIZE) ? proc->cmd : &proc->cmd[cmdlen-DEFAULT_CMD_SIZE];
        snprintf(proc->id, sizeof(proc->id), "%s-%s-%s", proc->hostname, proc->procname, ptr);
    } else {
        snprintf(proc->id, sizeof(proc->id), "badid");
    }

    proc->uid = getuid();
    proc->gid = getgid();
    if (osGetCgroup(proc->pid, proc->cgroup, MAX_CGROUP) == FALSE) {
        proc->cgroup[0] = '\0';
    }
}

static void
doReset()
{
    setProcId(&g_proc);
    setPidEnv(g_proc.pid);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    g_thread.once = 0;
    g_thread.startTime = tv.tv_sec + g_thread.interval;

    resetState();

    logReconnect(g_log);
    mtcReconnect(g_mtc);
    ctlReconnect(g_ctl, CFG_CTL);
    ctlReconnect(g_ctl, CFG_LS);

    atomicCasU64(&reentrancy_guard, 1ULL, 0ULL);

    reportProcessStart(g_ctl, TRUE, CFG_WHICH_MAX);
    threadInit();
}

static void
reportPeriodicStuff(void)
{
    long mem;
    int nthread, nfds, children;
    long long cpu = 0;
    static long long cpuState = 0;

    // empty the event queues
    doEvent();
    doPayload();

    // We report CPU time for this period.
    cpu = doGetProcCPU();
    if (cpu != -1) {
        doProcMetric(PROC_CPU, cpu - cpuState);
        cpuState = cpu;
    }

    mem = osGetProcMemory(g_proc.pid);
    if (mem != -1) doProcMetric(PROC_MEM, mem);

    nthread = osGetNumThreads(g_proc.pid);
    if (nthread != -1) doProcMetric(PROC_THREAD, nthread);

    nfds = osGetNumFds(g_proc.pid);
    if (nfds != -1) doProcMetric(PROC_FD, nfds);

    children = osGetNumChildProcs(g_proc.pid);
    if (children < 0) {
        children = 0;
    }
    doProcMetric(PROC_CHILD, children);

    // report totals (not by file descriptor/socket descriptor)
    doTotal(TOT_READ);
    doTotal(TOT_WRITE);
    doTotal(TOT_RX);
    doTotal(TOT_TX);
    doTotal(TOT_SEEK);
    doTotal(TOT_STAT);
    doTotal(TOT_OPEN);
    doTotal(TOT_CLOSE);
    doTotal(TOT_DNS);

    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_UDP_CONN);
    doTotal(TOT_OTHER_CONN);

    doTotalDuration(TOT_FS_DURATION);
    doTotalDuration(TOT_NET_DURATION);
    doTotalDuration(TOT_DNS_DURATION);

    // Report errors
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary", NULL);
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary", NULL);
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary", NULL);
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary", NULL);
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary", NULL);
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary", NULL);

    // report net and file by descriptor
    reportAllFds(PERIODIC);

    mtcFlush(g_mtc);
}

void
handleExit(void)
{
    if (g_exitdone == TRUE) return;
    g_exitdone = TRUE;

    if (!atomicCasU64(&reentrancy_guard, 0ULL, 1ULL)) {
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 10000}; // 10 us

        // let the periodic thread finish
        while (!atomicCasU64(&reentrancy_guard, 0ULL, 1ULL)) {
            sigSafeNanosleep(&ts);
        }
    }

    struct timespec ts = {.tv_sec = 1, .tv_nsec = 0}; // 1 s

    char *wait;
    if ((wait = getenv("SCOPE_CONNECT_TIMEOUT_SECS")) != NULL) {
        // wait for a connection to be established 
        // before we emit data
        int wait_time;
        errno = 0;
        wait_time = strtoul(wait, NULL, 10);
        if (!errno && wait_time) {
            for (int i = 0; i < wait_time; i++) {
                if (doConnection() == TRUE) break;
                sigSafeNanosleep(&ts);
            }
        }
    }

    reportPeriodicStuff();

    mtcFlush(g_mtc);
    mtcDisconnect(g_mtc);
    ctlStopAggregating(g_ctl);
    ctlFlush(g_ctl);
    ctlDisconnect(g_ctl, CFG_LS);
    ctlDisconnect(g_ctl, CFG_CTL);
    logFlush(g_log);
    logDisconnect(g_log);
}

static void *
periodic(void *arg)
{
    // Mask all the signals for this thread to avoid issues with go runtime.
    // Go runtime installs their own signal handlers so the go signal handler 
    // may get executed in the context of this thread, which will cause the app to 
    // crash because the go runtime won't be able to find a "g" in the TLS.
    // Since we can’t predict the thread that will be chosen to run the signal handler, 
    // the only reasonable solution seems to be masking the signals for this thread

    sigset_t mask;
    sigfillset(&mask);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    bool perf;
    static time_t summaryTime;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    summaryTime = tv.tv_sec + g_thread.interval;

    perf = checkEnv(PRESERVE_PERF_REPORTING, "true");

    while (1) {
        gettimeofday(&tv, NULL);
        if (tv.tv_sec >= summaryTime) {
            // Process dynamic config changes, if any
            dynConfig();

            // TODO: need to ensure that the previous object is no longer in use
            // Clean up previous objects if they exist.
            //if (g_prevmtc) mtcDestroy(&g_prevmtc);
            //if (g_prevlog) logDestroy(&g_prevlog);

            // Q: What does it mean to connect transports we expect to be
            // "connectionless"?  A: We've observed some processes close all
            // file/socket descriptors during their initialization.
            // If this happens, this the way we manage re-init.
            if (mtcNeedsConnection(g_mtc)) mtcConnect(g_mtc);
            if (logNeedsConnection(g_log)) logConnect(g_log);

            if (atomicCasU64(&reentrancy_guard, 0ULL, 1ULL)) {
                reportPeriodicStuff();
                atomicCasU64(&reentrancy_guard, 1ULL, 0ULL);
            }

            gettimeofday(&tv, NULL);
            summaryTime = tv.tv_sec + g_thread.interval;
        } else if (perf == FALSE) {
            if (atomicCasU64(&reentrancy_guard, 0ULL, 1ULL)) {
                doEvent();
                doPayload();
                atomicCasU64(&reentrancy_guard, 1ULL, 0ULL);
            }
        }
        remoteConfig();
    }

    return NULL;
}

// TODO; should this move to os/linux/os.c?
void *
memcpy(void *dest, const void *src, size_t n)
{
    return memmove(dest, src, n);
}

static int
ssl_read_hook(SSL *ssl, void *buf, int num)
{
    int rc;

    scopeLog(CFG_LOG_TRACE, "ssl_read_hook");
    WRAP_CHECK(SSL_read, -1);
    rc = g_fn.SSL_read(ssl, buf, num);
    if (rc > 0) {
        int fd = -1;
        if (SYMBOL_LOADED(SSL_get_fd)) fd = g_fn.SSL_get_fd(ssl);
        if ((fd == -1) && (g_ssl_fd != -1)) fd = g_ssl_fd;
        doProtocol((uint64_t)ssl, fd, buf, (size_t)rc, TLSRX, BUF);
    }

    return rc;
}

static int
ssl_write_hook(SSL *ssl, void *buf, int num)
{
    int rc;

    scopeLog(CFG_LOG_TRACE, "ssl_write_hook");
    WRAP_CHECK(SSL_write, -1);
    rc = g_fn.SSL_write(ssl, buf, num);
    if (rc > 0) {
        int fd = -1;
        if (SYMBOL_LOADED(SSL_get_fd)) fd = g_fn.SSL_get_fd(ssl);
        if ((fd == -1) && (g_ssl_fd != -1)) fd = g_ssl_fd;
        doProtocol((uint64_t)ssl, fd, (void *)buf, (size_t)rc, TLSTX, BUF);
    }

    return rc;
}

static void *
load_func(const char *module, const char *func)
{
    void *addr;
    
    void *handle = g_fn.dlopen(module, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == NULL) {
        scopeLogError("ERROR: Could not open file %s.\n", module ? module : "(null)");
        return NULL;
    }

    addr = dlsym(handle, func);
    dlclose(handle);

    if (addr == NULL) {
        scopeLogError("ERROR: Could not get function address of %s.\n", func);
        return NULL;
    }

    scopeLogError("%s:%d %s found at %p\n", __FUNCTION__, __LINE__, func, addr);
    return addr;
}

typedef struct
{
    const char *library;    // Input:   e.g. libpthread.so
    const char *symbol;     // Input:   e.g. __write
    void **out_addr;        // Output:  e.g. 0x7fffff2523A23734
} find_sym_t;

static int
findLibSym(struct dl_phdr_info *info, size_t size, void *data)
{
    find_sym_t *find = (find_sym_t *)data;
    *(find->out_addr) = NULL;

    if (strstr(info->dlpi_name, find->library)) {

        void *handle = g_fn.dlopen(info->dlpi_name, RTLD_NOW);
        if (!handle) return 0;
        void *addr = dlsym(handle, find->symbol);
        dlclose(handle);

        // if we don't find addr, keep going
        if (!addr)  return 0;

        // We found symbol in library!  Return the address for it!
        *(find->out_addr) = addr;
        return 1;

    }
    return 0;
}

/*
 * There are 3x SSL_read functions to consider:
 * 1) the one in this file used to interpose SSL_read
 * from a dynamic lib load (ex. curl)
 *
 * 2) the one in an ELF file for the process (main)
 * included from a static link lib (ex. node.js)
 *
 * 3) the one in a dynamic lib (ex. libssl.so)
 *
 * Only perform the hot patch in the case where
 * we find a symbol from the local process (main)
 * that is not our interposed function.
 *
 * We start by getting the address of SSL_read from
 * libscope.so by locating the path to this lib, then
 * get a handle and lookup the symbol.
 */
static ssize_t __write_libc(int, const void *, size_t);
static ssize_t __write_pthread(int, const void *, size_t);
static ssize_t scope_write(int, const void *, size_t);
static int internal_sendmmsg(int, struct mmsghdr *, unsigned int, int);
static ssize_t internal_sendto(int, const void *, size_t, int, const struct sockaddr *, socklen_t);
static ssize_t internal_recvfrom(int, void *, size_t, int, struct sockaddr *, socklen_t *);
static size_t __stdio_write(struct MUSL_IO_FILE *, const unsigned char *, size_t);
static long scope_syscall(long, ...);

static int 
findLibscopePath(struct dl_phdr_info *info, size_t size, void *data)
{
    int len = strlen(info->dlpi_name);
    int libscope_so_len = 11;

    if(len > libscope_so_len && !strcmp(info->dlpi_name + len - libscope_so_len, "libscope.so")) {
        *(char **)data = (char *) info->dlpi_name;
        return 1;
    }
    return 0;
}

/*
 * Iterate all shared objects and GOT hook as necessary.
 * Return FALSE in all cases in order to interate all objects.
 * Ignore a set of objects we know we don't want to hook.
 */
static int
hookSharedObjs(struct dl_phdr_info *info, size_t size, void *data)
{
    if (!info || !data || !info->dlpi_name) return FALSE;

    struct link_map *lm;
    void *addr = NULL;
    Elf64_Sym *sym = NULL;
    Elf64_Rela *rel = NULL;
    char *str = NULL;
    int rsz = 0;
    void *libscopeHandle = data;
    const char *libname = NULL;

    // don't attempt to hook libscope.so, libc*.so, ld-*.so
    // where libc*.so is for example libc.so.6 or libc.musl-x86_64.so.1
    // where ld-*.so is for example ld-linux-x86-64.so.2 or ld-musl-x86_64.so.1
    // if opening the main exec, name is NULL, else use the full lib name
    if (strstr(info->dlpi_name, ".so")) {
        if (strstr(info->dlpi_name, "libc") ||
            strstr(info->dlpi_name, "ld-") ||
            strstr(info->dlpi_name, "libscope")) {
            return FALSE;
        }
        libname = info->dlpi_name;
    }

    void *handle = g_fn.dlopen(libname, RTLD_LAZY);
    if (handle == NULL) return FALSE;

    // Get the link map and ELF sections in advance of something matching
    if ((dlinfo(handle, RTLD_DI_LINKMAP, (void *)&lm) != -1) && (getElfEntries(lm, &rel, &sym, &str, &rsz) != -1)) {
        for (int i=0; inject_hook_list[i].symbol; i++) {
            addr = dlsym(libscopeHandle, inject_hook_list[i].symbol);
            inject_hook_list[i].func = addr;

            if ((dlsym(handle, inject_hook_list[i].symbol)) &&
                (doGotcha(lm, (got_list_t *)&inject_hook_list[i], rel, sym, str, rsz, 1) != -1)) {
                    scopeLog(CFG_LOG_DEBUG, "\tGOT patched %s from shared obj %s",
                             inject_hook_list[i].symbol, info->dlpi_name);
            }
        }
    }

    dlclose(handle);
    return FALSE;
}

/*
 * Called when injected to perform GOT hooking.
 */ 
static bool
hookInject()
{
    char *full_path;

    if (dl_iterate_phdr(findLibscopePath, &full_path)) {
        void *libscopeHandle = g_fn.dlopen(full_path, RTLD_NOW);
        if (libscopeHandle == NULL) {
            return FALSE;
        }

        dl_iterate_phdr(hookSharedObjs, libscopeHandle);
        dlclose(libscopeHandle);
        return TRUE;
    }
    return FALSE;
}

static void
initHook(int attachedFlag)
{
    int rc;
    funchook_t *funchook;
    bool should_we_patch = FALSE;
    char *full_path = NULL;
    elf_buf_t *ebuf = NULL;

    // env vars are not always set as needed, be explicit here
    // this is duplicated if we were started from the scope exec
    if ((osGetExePath(&full_path) != -1) &&
        ((ebuf = getElf(full_path))) &&
        (is_static(ebuf->buf) == FALSE) && (is_go(ebuf->buf) == TRUE)) {
#ifdef __GO__
        initGoHook(ebuf);
        threadNow(0);
        if (arch_prctl(ARCH_GET_FS, (unsigned long)&scope_fs) == -1) {
            scopeLogError("initHook:arch_prctl");
        }

        __asm__ volatile (
            "lea scope_stack(%%rip), %%r11 \n"
            "mov %%rsp, (%%r11)  \n"
            : "=r"(rc)                    //output
            :
            : "%r11"                      //clobbered register
            );

        if (full_path) free(full_path);
        if (ebuf) freeElf(ebuf->buf, ebuf->len);
        return;
#endif  // __GO__
    }

    if (ebuf && ebuf->buf) {

        // This is in support of a libuv specific extension to map an SSL ID to a fd.
        // The symbol uv__read is not public. Therefore, we don't resolve it with dlsym.
        // So, while we have the exec open, we look to see if we can dig it out.
        g_fn.uv__read = getSymbol(ebuf->buf, "uv__read");
        scopeLog(CFG_LOG_TRACE, "%s:%d uv__read at %p", __FUNCTION__, __LINE__, g_fn.uv__read);
    }

    if (full_path) free(full_path);
    if (ebuf) freeElf(ebuf->buf, ebuf->len);

    if (attachedFlag) {
        hookInject();
    }

    if (dl_iterate_phdr(findLibscopePath, &full_path)) {
        void *handle = g_fn.dlopen(full_path, RTLD_NOW);
        if (handle == NULL) {
            return;
        }

        void *addr = dlsym(handle, "SSL_read");
        if (addr != SSL_read) {
            should_we_patch = TRUE;
        }

        dlclose(handle);
    }

    // We're funchooking __write in both libc.so and libpthread.so
    // curl didn't work unless we funchook'd libc.
    // test/linux/unixpeer didn't work unless we funchook'd pthread.
    find_sym_t libc__write = {.library="libc.so",
                              .symbol="__write",
                              .out_addr = (void*)&g_fn.__write_libc};
    dl_iterate_phdr(findLibSym, &libc__write);

    find_sym_t pthread__write = {.library = "libpthread.so",
                                 .symbol = "__write",
                                 .out_addr = (void*)&g_fn.__write_pthread};
    dl_iterate_phdr(findLibSym, &pthread__write);

    // for DNS:
    // On a glibc distro we hook sendmmsg because getaddrinfo calls this
    // directly and we miss DNS requests unless it's hooked.
    //
    // For DNS and console I/O
    // On a musl distro the name server code calls sendto and recvfrom
    // directly. So, we hook these in order to get DNS detail. We check
    // to see which distro is used so as to hook only what is needed.
    // We use the fact that on a musl distro the ld lib env var is set to
    // a dir with a libscope-ver string. If that exists in the env var
    // then we assume musl.
    if (should_we_patch || g_fn.__write_libc || g_fn.__write_pthread ||
        ((g_ismusl == FALSE) && g_fn.sendmmsg) ||
        ((g_ismusl == TRUE) && (g_fn.sendto || g_fn.recvfrom))) {
        funchook = funchook_create();

        if (logLevel(g_log) <= CFG_LOG_TRACE) {
            // TODO: add some mechanism to get the config'd log file path
            funchook_set_debug_file(DEFAULT_LOG_PATH);
        }

        if (should_we_patch) {
            g_fn.SSL_read = (ssl_rdfunc_t)load_func(NULL, SSL_FUNC_READ);
    
            rc = funchook_prepare(funchook, (void**)&g_fn.SSL_read, ssl_read_hook);

            g_fn.SSL_write = (ssl_wrfunc_t)load_func(NULL, SSL_FUNC_WRITE);

            rc = funchook_prepare(funchook, (void**)&g_fn.SSL_write, ssl_write_hook);
        }

        // sendmmsg, sendto, recvfrom for internal libc use in DNS queries
        if ((g_ismusl == FALSE) && g_fn.sendmmsg) {
            rc = funchook_prepare(funchook, (void**)&g_fn.sendmmsg, internal_sendmmsg);
        }

        if (g_fn.syscall) {
            rc = funchook_prepare(funchook, (void**)&g_fn.syscall, scope_syscall);
        }

        if ((g_ismusl == TRUE) && g_fn.sendto) {
            rc = funchook_prepare(funchook, (void**)&g_fn.sendto, internal_sendto);
        }

        if ((g_ismusl == TRUE) && g_fn.recvfrom) {
            rc = funchook_prepare(funchook, (void**)&g_fn.recvfrom, internal_recvfrom);
        }

        // Used for mapping SSL IDs to fds with libuv. Must be funchooked since it's internal to libuv
        if (!g_fn.uv_fileno) g_fn.uv_fileno = load_func(NULL, "uv_fileno");
        if (g_fn.uv__read) rc = funchook_prepare(funchook, (void**)&g_fn.uv__read, uv__read_hook);

        // libmusl
        // Note that both stdout & stderr objects point to the same write function.
        // They are init'd with a static object. After the first write the
        // write pointer is modified. We handle that mod in the interposed
        // function __stdio_write().
        if ((g_ismusl == TRUE) && (!g_fn.__stdout_write || !g_fn.__stderr_write)) {
            // Get the static initializer for the stdout write function pointer
            struct MUSL_IO_FILE *stdout_write = (struct MUSL_IO_FILE *)stdout;

            // Save the write pointer
            g_fn.__stdout_write = (size_t (*)(FILE *, const unsigned char *, size_t))stdout_write->write;

            // Modify the write pointer to use our function
            stdout_write->write = (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write;

            // same for stderr
            struct MUSL_IO_FILE *stderr_write = (struct MUSL_IO_FILE *)stderr;

            // Save the write pointer
            g_fn.__stderr_write = (size_t (*)(FILE *, const unsigned char *, size_t))stderr_write->write;

            // Modify the write pointer to use our function
            stderr_write->write = (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write;
        }

        if (g_ismusl == FALSE) {
            if (g_fn.__write_libc) {
                rc = funchook_prepare(funchook, (void**)&g_fn.__write_libc, __write_libc);
            }

            if (g_fn.__write_pthread) {
                rc = funchook_prepare(funchook, (void**)&g_fn.__write_pthread, __write_pthread);
            }

            // We want to be able to use g_fn.write without
            // accidentally interposing this function.  This resolves
            // https://github.com/criblio/appscope/issues/472
            g_fn.write = scope_write;
        }

        // hook 'em
        rc = funchook_install(funchook, 0);
        if (rc != 0) {
            scopeLogError("ERROR: failed to install SSL_read hook. (%s)\n",
                        funchook_error_message(funchook));
            return;
        }
    }
}

static void
initEnv(int *attachedFlag)
{
    // clear the flag by default
    *attachedFlag = 0;

    if (!g_fn.fopen || !g_fn.fgets || !g_fn.fclose || !g_fn.setenv) {
        // these log statements use debug level so they can be used with constructor debug
        scopeLog(CFG_LOG_DEBUG, "ERROR: missing g_fn's for initEnv()");
        return;
    }

    // build the full path of the .env file
    char path[128];
    int  pathLen = snprintf(path, sizeof(path), "/dev/shm/scope_attach_%d.env", getpid());
    if (pathLen < 0 || pathLen >= sizeof(path)) {
        scopeLog(CFG_LOG_DEBUG, "ERROR: snprintf(scope_attach_PID.env) failed");
        return;
    }

    // open it
    FILE *fd = g_fn.fopen(path, "r");
    if (fd == NULL) {
        scopeLog(CFG_LOG_DEBUG, "ERROR: fopen(scope_attach_PID.env) failed");
        return;
    }

    // the .env file is there so we're attached
    *attachedFlag = 1;

    // read "KEY=VALUE\n" lines and add them to the environment
    char line[8192];
    while (g_fn.fgets(line, sizeof(line), fd)) {
        int len = strlen(line);
        if (line[len-1] == '\n') line[len-1] = '\0';
        char *key = strtok(line, "=");
        if (key) {
            char *val = strtok(NULL, "=");
            if (val) {
                fullSetenv(key, val, 1);
            } else {
                scopeLog(CFG_LOG_DEBUG, "ERROR: strtok(val) failed");
            }
        } else {
            scopeLog(CFG_LOG_DEBUG, "ERROR: strtok(key) failed");
        }
    }

    // done
    g_fn.fclose(fd);
}

__attribute__((constructor)) void
init(void)
{

    // Bootstrapping...  we need to know if we're in musl so we can
    // call the right initFn function...
    {
        char *full_path = NULL;
        elf_buf_t *ebuf = NULL;

        // Needed for getElf()
        g_fn.open = dlsym(RTLD_NEXT, "open");
        if (!g_fn.open) g_fn.open = dlsym(RTLD_DEFAULT, "open");
        g_fn.close = dlsym(RTLD_NEXT, "close");
        if (!g_fn.close) g_fn.close = dlsym(RTLD_DEFAULT, "close");

        g_ismusl =
            ((osGetExePath(&full_path) != -1) &&
            !strstr(full_path, "ldscope") &&
            ((ebuf = getElf(full_path))) &&
            !is_static(ebuf->buf) &&
            !is_go(ebuf->buf) &&
            is_musl(ebuf->buf));

        if (full_path) free(full_path);
        if (ebuf) freeElf(ebuf->buf, ebuf->len);
    }

    // Use dlsym to get addresses for everything in g_fn
    if (g_ismusl) {
        initFn_musl();
    } else {
        initFn();
    }

// TODO: will want to see if this is needed for bash built on ARM...
#ifndef __aarch64__
    // bash can be compiled to use glibc's memory subsystem or it's own
    // internal memory subsystem.  It's own is not threadsafe.  If we
    // find that bash is using it's own memory, replace it with glibc's
    // so our own thread can run safely in parallel.
    if (func_found_in_executable("malloc", "bash")) {
        run_bash_mem_fix();
    }
#endif

    setProcId(&g_proc);
    setPidEnv(g_proc.pid);

    // initEnv() will set this TRUE if it detects `scope_attach_PID.env` in
    // `/dev/shm` with our PID indicating we were injected into a running
    // process.
    int attachedFlag = 0;
    initEnv(&attachedFlag);
    // logging inside constructor start from this line
    g_constructor_debug_enabled = checkEnv("SCOPE_ALLOW_CONSTRUCT_DBG", "true");

    initState();

    g_nsslist = lstCreate(freeNssEntry);

    initTime();

    char *path = cfgPath();
    config_t *cfg = cfgRead(path);
    cfgProcessEnvironment(cfg);

    doConfig(cfg);
    g_staticfg = cfg;
    if (path) free(path);
    if (!g_dbg) dbgInit();
    g_getdelim = 0;

    g_cfg.staticfg = g_staticfg;
    g_cfg.blockconn = DEFAULT_PORTBLOCK;

    reportProcessStart(g_ctl, TRUE, CFG_WHICH_MAX);

    // replaces atexit(handleExit);  Allows events to be reported before
    // the TLS destructors are run.  This mechanism is used regardless
    // of whether TLS is actually configured on any transport.
    transportRegisterForExitNotification(handleExit);

    initHook(attachedFlag);
    
    if (checkEnv("SCOPE_APP_TYPE", "go") &&
        checkEnv("SCOPE_EXEC_TYPE", "static")) {
        threadNow(0);
    } else if (g_ismusl == FALSE) {
        // The check here is meant to be temporary.
        // The behavior of timer_delete() in musl libc
        // is different than that of gnu libc.
        // Therefore, until that is investigated we don't
        // enable a timer/signal.
        threadInit();
    }

    osInitJavaAgent();
}

EXPORTON int
sigaction(int signum, const struct sigaction *act, struct sigaction *oldact)
{
    WRAP_CHECK(sigaction, -1);
    /*
     * If there is a handler being installed, just save it.
     * If no handler, they may just be checking for the current handler.
     */
    if ((signum == SIGUSR2) && (act != NULL)) {
        g_thread.act = act; 
        return 0;
    }

    return g_fn.sigaction(signum, act, oldact);
}

EXPORTON int
open(const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;

    WRAP_CHECK(open, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);

    fd = g_fn.open(pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "open");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "open", pathname);
    }

    return fd;
}

EXPORTON int
openat(int dirfd, const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;

    WRAP_CHECK(openat, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.openat(dirfd, pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "openat");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "openat", pathname);
    }

    return fd;
}

// Note: creat64 is defined to be obsolete
EXPORTON int
creat(const char *pathname, mode_t mode)
{
    int fd;

    WRAP_CHECK(creat, -1);
    fd = g_fn.creat(pathname, mode);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "creat");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "creat", pathname);
    }

    return fd;
}

EXPORTON FILE *
fopen(const char *pathname, const char *mode)
{
    FILE *stream;

    WRAP_CHECK(fopen, NULL);
    stream = g_fn.fopen(pathname, mode);
    if (stream != NULL) {
        // This check for /proc/self/maps is because we want to avoid
        // reporting that our funchook library opens /proc/self/maps
        // with fopen, then calls fgets and fclose on it as well...
        // See https://github.com/criblio/appscope/issues/214 for more info
        if (strcmp(pathname, "/proc/self/maps")) {
            doOpen(fileno(stream), pathname, STREAM, "fopen");
        }
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, -1, (ssize_t)0, "fopen", pathname);
    }

    return stream;
}

EXPORTON FILE *
freopen(const char *pathname, const char *mode, FILE *orig_stream)
{
    FILE *stream;

    WRAP_CHECK(freopen, NULL);
    stream = g_fn.freopen(pathname, mode, orig_stream);
    // freopen just changes the mode if pathname is null
    if (stream != NULL) {
        if (pathname != NULL) {
            doOpen(fileno(stream), pathname, STREAM, "freopen");
            doClose(fileno(orig_stream), "freopen");
        }
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, -1, (ssize_t)0, "freopen", pathname);
    }

    return stream;
}

#ifdef __linux__
EXPORTON int
nanosleep(const struct timespec *req, struct timespec *rem)
{
    stopTimer();
    WRAP_CHECK(nanosleep, -1);
    return g_fn.nanosleep(req, rem);
}

EXPORTON int
select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout)
{
    stopTimer();
    WRAP_CHECK(select, -1);
    return g_fn.select(nfds, readfds, writefds, exceptfds, timeout);
}

EXPORTON int
sigsuspend(const sigset_t *mask)
{
    stopTimer();
    WRAP_CHECK(sigsuspend, -1);
    return g_fn.sigsuspend(mask);
}

EXPORTON int
epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout)
{
    stopTimer();
    WRAP_CHECK(epoll_wait, -1);
    return g_fn.epoll_wait(epfd, events, maxevents, timeout);
}

EXPORTON int
poll(struct pollfd *fds, nfds_t nfds, int timeout)
{
    stopTimer();
    WRAP_CHECK(epoll_wait, -1);
    return g_fn.poll(fds, nfds, timeout);
}

EXPORTON int
pause(void)
{
    stopTimer();
    WRAP_CHECK(pause, -1);
    return g_fn.pause();
}

EXPORTON int
sigwaitinfo(const sigset_t *set, siginfo_t *info)
{
    stopTimer();
    WRAP_CHECK(sigwaitinfo, -1);
    return g_fn.sigwaitinfo(set, info);
}

EXPORTON int
sigtimedwait(const sigset_t *set, siginfo_t *info,
             const struct timespec *timeout)
{
    stopTimer();
    WRAP_CHECK(sigtimedwait, -1);
    return g_fn.sigtimedwait(set, info, timeout);
}

EXPORTON int
epoll_pwait(int epfd, struct epoll_event *events,
            int maxevents, int timeout,
            const sigset_t *sigmask)
{
    stopTimer();
    WRAP_CHECK(epoll_pwait, -1);
    return g_fn.epoll_pwait(epfd, events, maxevents, timeout, sigmask);
}

EXPORTON int
ppoll(struct pollfd *fds, nfds_t nfds, const struct timespec *tmo_p,
      const sigset_t *sigmask)
{
    stopTimer();
    WRAP_CHECK(ppoll, -1);
    return g_fn.ppoll(fds, nfds, tmo_p, sigmask);
}

EXPORTON int
pselect(int nfds, fd_set *readfds, fd_set *writefds,
        fd_set *exceptfds, const struct timespec *timeout,
        const sigset_t *sigmask)
{
    stopTimer();
    WRAP_CHECK(pselect, -1);
    return g_fn.pselect(nfds, readfds, writefds, exceptfds, timeout, sigmask);
}

EXPORTON int
msgsnd(int msqid, const void *msgp, size_t msgsz, int msgflg)
{
    stopTimer();
    WRAP_CHECK(msgsnd, -1);
    return g_fn.msgsnd(msqid, msgp, msgsz, msgflg);
}

EXPORTON ssize_t
msgrcv(int msqid, void *msgp, size_t msgsz, long msgtyp, int msgflg)
{
    stopTimer();
    WRAP_CHECK(msgrcv, -1);
    return g_fn.msgrcv(msqid, msgp, msgsz, msgtyp, msgflg);
}

EXPORTON int
semop(int semid, struct sembuf *sops, size_t nsops)
{
    stopTimer();
    WRAP_CHECK(semop, -1);
    return g_fn.semop(semid, sops, nsops);
}

EXPORTON int
semtimedop(int semid, struct sembuf *sops, size_t nsops,
           const struct timespec *timeout)
{
    stopTimer();
    WRAP_CHECK(semtimedop, -1);
    return g_fn.semtimedop(semid, sops, nsops, timeout);
}

EXPORTON int
clock_nanosleep(clockid_t clockid, int flags,
                const struct timespec *request,
                struct timespec *remain)
{
    stopTimer();
    WRAP_CHECK(clock_nanosleep, -1);
    return g_fn.clock_nanosleep(clockid, flags, request, remain);
}

EXPORTON int
usleep(useconds_t usec)
{
    stopTimer();
    WRAP_CHECK(usleep, -1);
    return g_fn.usleep(usec);
}

EXPORTON int
io_getevents(io_context_t ctx_id, long min_nr, long nr,
             struct io_event *events, struct timespec *timeout)
{
    stopTimer();
    WRAP_CHECK(io_getevents, -1);
    return g_fn.io_getevents(ctx_id, min_nr, nr, events, timeout);
}

EXPORTON int
open64(const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;

    WRAP_CHECK(open64, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.open64(pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "open64");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "open64", pathname);
    }

    return fd;
}

EXPORTON int
openat64(int dirfd, const char *pathname, int flags, ...)
{
    int fd;
    struct FuncArgs fArgs;

    WRAP_CHECK(openat64, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, flags);
    fd = g_fn.openat64(dirfd, pathname, flags, fArgs.arg[0]);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "openat64");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "openat64", pathname);
    }

    return fd;
}

EXPORTON int
__open_2(const char *file, int oflag)
{
    int fd;

    WRAP_CHECK(__open_2, -1);
    fd = g_fn.__open_2(file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__open_2");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "__openat_2", file);
    }

    return fd;
}

EXPORTON int
__open64_2(const char *file, int oflag)
{
    int fd;

    WRAP_CHECK(__open64_2, -1);
    fd = g_fn.__open64_2(file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__open_2");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "__open64_2", file);
    }

    return fd;
}

EXPORTON int
__openat_2(int fd, const char *file, int oflag)
{
    WRAP_CHECK(__openat_2, -1);
    fd = g_fn.__openat_2(fd, file, oflag);
    if (fd != -1) {
        doOpen(fd, file, FD, "__openat_2");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "__openat_2", file);
    }

    return fd;
}

// Note: creat64 is defined to be obsolete
EXPORTON int
creat64(const char *pathname, mode_t mode)
{
    int fd;

    WRAP_CHECK(creat64, -1);
    fd = g_fn.creat64(pathname, mode);
    if (fd != -1) {
        doOpen(fd, pathname, FD, "creat64");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, fd, (ssize_t)0, "creat64", pathname);
    }

    return fd;
}

EXPORTON FILE *
fopen64(const char *pathname, const char *mode)
{
    FILE *stream;

    WRAP_CHECK(fopen64, NULL);
    stream = g_fn.fopen64(pathname, mode);
    if (stream != NULL) {
        doOpen(fileno(stream), pathname, STREAM, "fopen64");
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, -1, (ssize_t)0, "fopen64", pathname);
    }

    return stream;
}

EXPORTON FILE *
freopen64(const char *pathname, const char *mode, FILE *orig_stream)
{
    FILE *stream;

    WRAP_CHECK(freopen64, NULL);
    stream = g_fn.freopen64(pathname, mode, orig_stream);
    // freopen just changes the mode if pathname is null
    if (stream != NULL) {
        if (pathname != NULL) {
            doOpen(fileno(stream), pathname, STREAM, "freopen64");
            doClose(fileno(orig_stream), "freopen64");
        }
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, -1, (ssize_t)0, "freopen64", pathname);
    }

    return stream;
}

EXPORTON ssize_t
pread64(int fd, void *buf, size_t count, off_t offset)
{
    WRAP_CHECK(pread64, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pread64(fd, buf, count, offset);

    doRead(fd, initialTime, (rc != -1), (void *)buf, rc, "pread64", BUF, 0);

    return rc;
}

EXPORTON ssize_t
preadv(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    WRAP_CHECK(preadv, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.preadv(fd, iov, iovcnt, offset);

    doRead(fd, initialTime, (rc != -1), iov, rc, "preadv", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
preadv2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    WRAP_CHECK(preadv2, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.preadv2(fd, iov, iovcnt, offset, flags);

    doRead(fd, initialTime, (rc != -1), iov, rc, "preadv2", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
preadv64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    WRAP_CHECK(preadv64v2, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.preadv64v2(fd, iov, iovcnt, offset, flags);

    doRead(fd, initialTime, (rc != -1), iov, rc, "preadv64v2", IOV, iovcnt);
    
    return rc;
}

EXPORTON ssize_t
__pread_chk(int fd, void *buf, size_t nbytes, off_t offset, size_t buflen)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__pread_chk, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.__pread_chk(fd, buf, nbytes, offset, buflen);

    doRead(fd, initialTime, (rc != -1), (void *)buf, rc, "__pread_chk", BUF, 0);

    return rc;
}

EXPORTOFF ssize_t
__read_chk(int fd, void *buf, size_t nbytes, size_t buflen)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__read_chk, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.__read_chk(fd, buf, nbytes, buflen);

    doRead(fd, initialTime, (rc != -1), (void *)buf, rc, "__read_chk", BUF, 0);

    return rc;
}

EXPORTOFF size_t
__fread_unlocked_chk(void *ptr, size_t ptrlen, size_t size, size_t nmemb, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fread_unlocked_chk, 0);
    uint64_t initialTime = getTime();

    size_t rc = g_fn.__fread_unlocked_chk(ptr, ptrlen, size, nmemb, stream);

    doRead(fileno(stream), initialTime, (rc == nmemb), NULL, rc*size, "__fread_unlocked_chk", NONE, 0);

    return rc;
}

EXPORTON ssize_t
pwrite64(int fd, const void *buf, size_t nbyte, off_t offset)
{
    WRAP_CHECK(pwrite64, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwrite64(fd, buf, nbyte, offset);

    doWrite(fd, initialTime, (rc != -1), buf, rc, "pwrite64", BUF, 0);

    return rc;
}

EXPORTON ssize_t
pwritev(int fd, const struct iovec *iov, int iovcnt, off_t offset)
{
    WRAP_CHECK(pwritev, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwritev(fd, iov, iovcnt, offset);

    doWrite(fd, initialTime, (rc != -1), iov, rc, "pwritev", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
pwritev64(int fd, const struct iovec *iov, int iovcnt, off64_t offset)
{
    WRAP_CHECK(pwritev64, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwritev64(fd, iov, iovcnt, offset);

    doWrite(fd, initialTime, (rc != -1), iov, rc, "pwritev64", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
pwritev2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    WRAP_CHECK(pwritev2, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwritev2(fd, iov, iovcnt, offset, flags);

    doWrite(fd, initialTime, (rc != -1), iov, rc, "pwritev2", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
pwritev64v2(int fd, const struct iovec *iov, int iovcnt, off_t offset, int flags)
{
    WRAP_CHECK(pwritev64v2, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwritev64v2(fd, iov, iovcnt, offset, flags);

    doWrite(fd, initialTime, (rc != -1), iov, rc, "pwritev64v2", IOV, iovcnt);

    return rc;
}

EXPORTON off64_t
lseek64(int fd, off64_t offset, int whence)
{
    WRAP_CHECK(lseek64, -1);

    off64_t rc = g_fn.lseek64(fd, offset, whence);

    doSeek(fd, (rc != -1), "lseek64");

    return rc;
}

EXPORTON int
fseeko64(FILE *stream, off64_t offset, int whence)
{
    WRAP_CHECK(fseeko64, -1);

    int rc = g_fn.fseeko64(stream, offset, whence);

    doSeek(fileno(stream), (rc != -1), "fseeko64");

    return rc;
}

EXPORTON off64_t
ftello64(FILE *stream)
{
    WRAP_CHECK(ftello64, -1);

    off64_t rc = g_fn.ftello64(stream);

    doSeek(fileno(stream), (rc != -1), "ftello64");

    return rc;
}

EXPORTON int
statfs64(const char *path, struct statfs64 *buf)
{
    WRAP_CHECK(statfs64, -1);
    int rc = g_fn.statfs64(path, buf);

    doStatPath(path, rc, "statfs64");

    return rc;
}

EXPORTON int
fstatfs64(int fd, struct statfs64 *buf)
{
    WRAP_CHECK(fstatfs64, -1);
    int rc = g_fn.fstatfs64(fd, buf);

    doStatFd(fd, rc, "fstatfs64");

    return rc;
}

EXPORTON int
fsetpos64(FILE *stream, const fpos64_t *pos)
{
    WRAP_CHECK(fsetpos64, -1);
    int rc = g_fn.fsetpos64(stream, pos);

    doSeek(fileno(stream), (rc == 0), "fsetpos64");

    return rc;
}

EXPORTON int
__xstat(int ver, const char *path, struct stat *stat_buf)
{
    WRAP_CHECK(__xstat, -1);
    int rc = g_fn.__xstat(ver, path, stat_buf);

    doStatPath(path, rc, "__xstat");

    return rc;    
}

EXPORTON int
__xstat64(int ver, const char *path, struct stat64 *stat_buf)
{
    WRAP_CHECK(__xstat64, -1);
    int rc = g_fn.__xstat64(ver, path, stat_buf);

    doStatPath(path, rc, "__xstat64");

    return rc;    
}

EXPORTON int
__lxstat(int ver, const char *path, struct stat *stat_buf)
{
    WRAP_CHECK(__lxstat, -1);
    int rc = g_fn.__lxstat(ver, path, stat_buf);

    doStatPath(path, rc, "__lxstat");

    return rc;
}

EXPORTON int
__lxstat64(int ver, const char *path, struct stat64 *stat_buf)
{
    WRAP_CHECK(__lxstat64, -1);
    int rc = g_fn.__lxstat64(ver, path, stat_buf);

    doStatPath(path, rc, "__lxstat64");

    return rc;
}

EXPORTON int
__fxstat(int ver, int fd, struct stat *stat_buf)
{
    WRAP_CHECK(__fxstat, -1);
    int rc = g_fn.__fxstat(ver, fd, stat_buf);

    doStatFd(fd, rc, "__fxstat");

    return rc;
}

EXPORTON int
__fxstat64(int ver, int fd, struct stat64 * stat_buf)
{
    WRAP_CHECK(__fxstat64, -1);
    int rc = g_fn.__fxstat64(ver, fd, stat_buf);

    doStatFd(fd, rc, "__fxstat64");

    return rc;
}

EXPORTON int
__fxstatat(int ver, int dirfd, const char *path, struct stat *stat_buf, int flags)
{
    WRAP_CHECK(__fxstatat, -1);
    int rc = g_fn.__fxstatat(ver, dirfd, path, stat_buf, flags);

    doStatPath(path, rc, "__fxstatat");

    return rc;
}

EXPORTON int
__fxstatat64(int ver, int dirfd, const char * path, struct stat64 * stat_buf, int flags)
{
    WRAP_CHECK(__fxstatat64, -1);
    int rc = g_fn.__fxstatat64(ver, dirfd, path, stat_buf, flags);

    doStatPath(path, rc, "__fxstatat64");

    return rc;
}

#ifdef __STATX__
EXPORTON int
statx(int dirfd, const char *pathname, int flags,
      unsigned int mask, struct statx *statxbuf)
{
    WRAP_CHECK(statx, -1);
    int rc = g_fn.statx(dirfd, pathname, flags, mask, statxbuf);

    doStatPath(pathname, rc, "statx");

    return rc;
}
#endif // __STATX__

EXPORTON int
statfs(const char *path, struct statfs *buf)
{
    WRAP_CHECK(statfs, -1);
    int rc = g_fn.statfs(path, buf);

    doStatPath(path, rc, "statfs");

    return rc;
}

EXPORTON int
fstatfs(int fd, struct statfs *buf)
{
    WRAP_CHECK(fstatfs, -1);
    int rc = g_fn.fstatfs(fd, buf);

    doStatFd(fd, rc, "fstatfs");

    return rc;
}

EXPORTON int
statvfs(const char *path, struct statvfs *buf)
{
    WRAP_CHECK(statvfs, -1);
    int rc = g_fn.statvfs(path, buf);

    doStatPath(path, rc, "statvfs");

    return rc;
}

EXPORTON int
statvfs64(const char *path, struct statvfs64 *buf)
{
    WRAP_CHECK(statvfs64, -1);
    int rc = g_fn.statvfs64(path, buf);

    doStatPath(path, rc, "statvfs64");

    return rc;
}

EXPORTON int
fstatvfs(int fd, struct statvfs *buf)
{
    WRAP_CHECK(fstatvfs, -1);
    int rc = g_fn.fstatvfs(fd, buf);

    doStatFd(fd, rc, "fstatvfs");

    return rc;
}

EXPORTON int
fstatvfs64(int fd, struct statvfs64 *buf)
{
    WRAP_CHECK(fstatvfs64, -1);
    int rc = g_fn.fstatvfs64(fd, buf);

    doStatFd(fd, rc, "fstatvfs64");

    return rc;
}

EXPORTON int
access(const char *pathname, int mode)
{
    WRAP_CHECK(access, -1);
    int rc = g_fn.access(pathname, mode);

    doStatPath(pathname, rc, "access");

    return rc;
}

EXPORTON int
faccessat(int dirfd, const char *pathname, int mode, int flags)
{
    WRAP_CHECK(faccessat, -1);
    int rc = g_fn.faccessat(dirfd, pathname, mode, flags);

    doStatPath(pathname, rc, "faccessat");

    return rc;
}

EXPORTOFF int
gethostbyname_r(const char *name, struct hostent *ret, char *buf, size_t buflen,
                struct hostent **result, int *h_errnop)
{
    int rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname_r, -1);
    time.initial = getTime();
    rc = g_fn.gethostbyname_r(name, ret, buf, buflen, result, h_errnop);
    time.duration = getDuration(time.initial);

    if ((rc == 0) && (result != NULL)) {
        scopeLog(CFG_LOG_DEBUG, "gethostbyname_r");
        doUpdateState(DNS, -1, time.duration, NULL, name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }  else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "gethostbyname_r", name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }

    return rc;
}

EXPORTOFF int
gethostbyname2_r(const char *name, int af, struct hostent *ret, char *buf,
                 size_t buflen, struct hostent **result, int *h_errnop)
{
    int rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname2_r, -1);
    time.initial = getTime();
    rc = g_fn.gethostbyname2_r(name, af, ret, buf, buflen, result, h_errnop);
    time.duration = getDuration(time.initial);

    if ((rc == 0) && (result != NULL)) {
        scopeLog(CFG_LOG_DEBUG, "gethostbyname2_r");
        doUpdateState(DNS, -1, time.duration, NULL, name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }  else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "gethostbyname2_r", name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }

    return rc;
}

/*
 * We explicitly don't interpose these stat functions on macOS
 * These are not exported symbols in Linux. Therefore, we
 * have them turned off for now.
 * stat, fstat, lstat.
 */
/*
EXPORTOFF int
stat(const char *pathname, struct stat *statbuf)
{
    WRAP_CHECK(stat, -1);
    int rc = g_fn.stat(pathname, statbuf);

    doStatPath(pathname, rc, "stat");

    return rc;
}

EXPORTOFF int
fstat(int fd, struct stat *statbuf)
{
    WRAP_CHECK(fstat, -1);
    int rc = g_fn.fstat(fd, statbuf);

    doStatFd(fd, rc, "fstat");

    return rc;
}

EXPORTOFF int
lstat(const char *pathname, struct stat *statbuf)
{
    WRAP_CHECK(lstat, -1);
    int rc = g_fn.lstat(pathname, statbuf);

    doStatPath(pathname, rc, "lstat");

    return rc;
}
*/
EXPORTON int
fstatat(int fd, const char *path, struct stat *buf, int flag)
{
    WRAP_CHECK(fstatat, -1);
    int rc = g_fn.fstatat(fd, path, buf, flag);

    doStatFd(fd, rc, "fstatat");

    return rc;
}

EXPORTON int
prctl(int option, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(prctl, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, option);

    if (option == PR_SET_SECCOMP) {
        return 0;
    }

    return g_fn.prctl(option, fArgs.arg[0], fArgs.arg[1], fArgs.arg[2], fArgs.arg[3]);
}

EXPORTON int
execve(const char *pathname, char *const argv[], char *const envp[])
{
    int i, nargs, saverr;
    bool isstat = FALSE, isgo = FALSE;
    char **nargv;
    elf_buf_t *ebuf;
    char *scopexec = NULL;

    WRAP_CHECK(execve, -1);

    if (strstr(g_proc.procname, "ldscope") ||
        checkEnv("SCOPE_EXECVE", "false")) {
        return g_fn.execve(pathname, argv, envp);
    }

    if ((ebuf = getElf((char *)pathname))) {
        isstat = is_static(ebuf->buf);
        isgo = is_go(ebuf->buf);
    }

    // not really necessary since we're gonna exec
    if (ebuf) freeElf(ebuf->buf, ebuf->len);

    /*
     * Note: the isgo check is strictly for Go dynamic execs.
     * In this case we use ldscope only to force the use of HTTP 1.1.
     */
    if (getenv("LD_PRELOAD") && (isstat == FALSE) && (isgo == FALSE)) {
        return g_fn.execve(pathname, argv, envp);
    }

    scopexec = getenv("SCOPE_EXEC_PATH");
    if (((scopexec = getpath(scopexec)) == NULL) &&
        ((scopexec = getpath("ldscope")) == NULL)) {

        // can't find the scope executable
        scopeLogWarn("execve: can't find a scope executable for %s", pathname);
        return g_fn.execve(pathname, argv, envp);
    }

    nargs = 0;
    while ((argv[nargs] != NULL)) nargs++;

    size_t plen = sizeof(char *);
    if ((nargs == 0) || (nargv = calloc(1, ((nargs * plen) + (plen * 2)))) == NULL) {
        return g_fn.execve(pathname, argv, envp);
    }

    nargv[0] = scopexec;
    nargv[1] = (char *)pathname;

    for (i = 2; i <= nargs; i++) {
        nargv[i] = argv[i - 1];
    }

    g_fn.execve(nargv[0], nargv, environ);
    saverr = errno;
    if (nargv) free(nargv);
    if (scopexec) free(scopexec);
    errno = saverr;
    return -1;
}

EXPORTON int
__overflow(FILE *stream, int ch)
{
    WRAP_CHECK(__overflow, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.__overflow(stream, ch);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.__overflow(stream, ch);

    doWrite(fileno(stream), initialTime, (rc != EOF), &ch, 1, "__overflow", BUF, 0);

    return rc;
}

static ssize_t
__write_libc(int fd, const void *buf, size_t size)
{
    WRAP_CHECK(__write_libc, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.__write_libc(fd, buf, size);

    doWrite(fd, initialTime, (rc != -1), buf, rc, "__write_libc", BUF, 0);

    return rc;
}

static ssize_t
__write_pthread(int fd, const void *buf, size_t size)
{
    WRAP_CHECK(__write_pthread, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.__write_pthread(fd, buf, size);

    doWrite(fd, initialTime, (rc != -1), buf, rc, "__write_pthread", BUF, 0);

    return rc;
}

static int
isAnAppScopeConnection(int fd)
{
    if (fd == -1) return FALSE;

    if ((fd == ctlConnection(g_ctl, CFG_CTL)) ||
        (fd == ctlConnection(g_ctl, CFG_LS)) ||
        (fd == mtcConnection(g_mtc)) ||
        (fd == logConnection(g_log))) {
        return TRUE;
    }

    return FALSE;
}

/*
 * Note:
 * The syscall function in libc is called from the loader for
 * at least mmap, possibly more. The result is that we can not
 * do any dynamic memory allocation while this executes. Be careful.
 * The DBG() output is ignored until after the constructor runs.
 */
static long
scope_syscall(long number, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(syscall, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, number);

    switch (number) {
    case SYS_close:
    {
        long rc;
        rc = g_fn.syscall(number, fArgs.arg[0]);
        doClose(fArgs.arg[0], "sysclose");
        return rc;
    }
    case SYS_accept4:
    {
        long rc;
        rc = g_fn.syscall(number, fArgs.arg[0], fArgs.arg[1],
                          fArgs.arg[2], fArgs.arg[3]);

        if ((rc != -1) && (doBlockConnection(fArgs.arg[0], NULL) == 1)) {
            if (g_fn.close) g_fn.close(rc);
            errno = ECONNABORTED;
            return -1;
        }

        if (rc != -1) {
            doAccept(rc, (struct sockaddr *)fArgs.arg[1],
                     (socklen_t *)fArgs.arg[2], "accept4");
        } else {
            doUpdateState(NET_ERR_CONN, fArgs.arg[0], (ssize_t)0, "accept4", "nopath");
        }
        return rc;
    }

    /*
     * These messages are in place as they represent
     * functions that use syscall() in libuv, used with node.js.
     * These are functions defined in libuv/src/unix/linux-syscalls.c
     * that we are otherwise interposing. The DBG call allows us to
     * check to see how many of these are called and therefore
     * what we are missing. So far, we only see accept4 used.
     */
    case SYS_sendmmsg:
        //DBG("syscall-sendmsg");
        break;

    case SYS_recvmmsg:
        //DBG("syscall-recvmsg");
        break;

    case SYS_preadv:
        //DBG("syscall-preadv");
        break;

    case SYS_pwritev:
        //DBG("syscall-pwritev");
        break;

    case SYS_dup3:
        //DBG("syscall-dup3");
        break;
#ifdef __STATX__
    case SYS_statx:
        //DBG("syscall-statx");
        break;
#endif // __STATX__
    default:
        // Supplying args is fine, but is a touch more work.
        // On splunk, in a container on my laptop, I saw this statement being
        // hit every 10-15 microseconds over a 15 minute duration.  Wow.
        // DBG("syscall-number: %d", number);
        //DBG(NULL);
        break;
    }

    return g_fn.syscall(number, fArgs.arg[0], fArgs.arg[1], fArgs.arg[2],
                        fArgs.arg[3], fArgs.arg[4], fArgs.arg[5]);
}

static ssize_t
scope_write(int fd, const void* buf, size_t size)
{
    return (ssize_t)syscall(SYS_write, fd, buf, size);
}

static int
scope_open(const char* pathname)
{
    // This implementation is largely based on transportConnectFile().
    int fd = g_fn.open(pathname, O_CREAT|O_WRONLY|O_APPEND|O_CLOEXEC, 0666);
    if (fd == -1) {
        DBG("%s", pathname);
        return fd;
    }

    // Since umask affects open permissions above...
    if (fchmod(fd, 0666) == -1) {
        DBG("%d %s", fd, pathname);
    }
    return fd;
}

#define scope_close(fd) g_fn.close(fd)

EXPORTON size_t
fwrite_unlocked(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
    WRAP_CHECK(fwrite_unlocked, 0);
    if (g_ismusl == FALSE) {
        return g_fn.fwrite_unlocked(ptr, size, nitems, stream);
    }

    uint64_t initialTime = getTime();

    size_t rc = g_fn.fwrite_unlocked(ptr, size, nitems, stream);

    doWrite(fileno(stream), initialTime, (rc == nitems), ptr, rc*size, "fwrite_unlocked", BUF, 0);

    return rc;
}

/*
 * Note: in_fd must be a file
 * out_fd can be a file or a socket
 *
 * Not sure is this is the way we want to do this, but:
 * We emit metrics for the input file that is being sent
 * We optionally emit metrics if the destination uses a socket
 * We do not emit a separate metric if the destination is a file
 */
EXPORTON ssize_t
sendfile(int out_fd, int in_fd, off_t *offset, size_t count)
{
    WRAP_CHECK(sendfile, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.sendfile(out_fd, in_fd, offset, count);

    doSendFile(out_fd, in_fd, initialTime, rc, "sendfile");

    return rc;
}

EXPORTON ssize_t
sendfile64(int out_fd, int in_fd, off64_t *offset, size_t count)
{
    WRAP_CHECK(sendfile, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.sendfile64(out_fd, in_fd, offset, count);

    doSendFile(out_fd, in_fd, initialTime, rc, "sendfile64");

    return rc;
}

EXPORTON int
SSL_read(SSL *ssl, void *buf, int num)
{
    int rc;
    
    //scopeLogError("SSL_read");
    WRAP_CHECK(SSL_read, -1);
    rc = g_fn.SSL_read(ssl, buf, num);

    if (rc > 0) {
        int fd = -1;
        if (SYMBOL_LOADED(SSL_get_fd)) fd = g_fn.SSL_get_fd(ssl);
        if ((fd == -1) && (g_ssl_fd != -1)) fd = g_ssl_fd;
        doProtocol((uint64_t)ssl, fd, buf, (size_t)rc, TLSRX, BUF);
    }
    return rc;
}

EXPORTON int
SSL_write(SSL *ssl, const void *buf, int num)
{
    int rc;
    
    //scopeLogError("SSL_write");
    WRAP_CHECK(SSL_write, -1);

    rc = g_fn.SSL_write(ssl, buf, num);

    if (rc > 0) {
        int fd = -1;
        if (SYMBOL_LOADED(SSL_get_fd)) fd = g_fn.SSL_get_fd(ssl);
        if ((fd == -1) && (g_ssl_fd != -1)) fd = g_ssl_fd;
        doProtocol((uint64_t)ssl, fd, (void *)buf, (size_t)rc, TLSTX, BUF);
    }
    return rc;
}

static int
gnutls_get_fd(gnutls_session_t session)
{
    int fd = -1;
    gnutls_transport_ptr_t fdp;

    if (SYMBOL_LOADED(gnutls_transport_get_ptr) &&
            ((fdp = g_fn.gnutls_transport_get_ptr(session)) != NULL)) {
        // In #279, we found that when the version of gnutls is 3.5.18, the
        // gnutls_transport_get_ptr() return value is the file-descriptor,
        // not a pointer to the file-descriptor. Using gnutls_check_version()
        // got messy as we may have found this behaviour switched on and off.
        // So, we're testing here if the integer value of the pointer is 
        // below the number of file-descriptors.
        if (!g_max_fds) {
            // NB: race-condition where multiple threads get here at the same
            // time and end up setting g_max_fds more than once. No problem.
            struct rlimit fd_limit;
            if (getrlimit(RLIMIT_NOFILE, &fd_limit) != 0) {
                DBG("getrlimit(0 failed");
                return fd;
            }
            g_max_fds = fd_limit.rlim_cur;
        }
        if ((uint64_t)fdp >= g_max_fds) {
            fd = *fdp;
        } else {
            fd = (int)(uint64_t)fdp;
        }
    }

    return fd;
}

EXPORTON ssize_t
gnutls_record_recv(gnutls_session_t session, void *data, size_t data_size)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_recv");
    WRAP_CHECK(gnutls_record_recv, -1);
    rc = g_fn.gnutls_record_recv(session, data, data_size);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, data, rc, TLSRX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_recv_early_data(gnutls_session_t session, void *data, size_t data_size)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_recv_early_data");
    WRAP_CHECK(gnutls_record_recv_early_data, -1);
    rc = g_fn.gnutls_record_recv_early_data(session, data, data_size);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, data, rc, TLSRX, BUF);
    }
    return rc;
}

EXPORTOFF ssize_t
gnutls_record_recv_packet(gnutls_session_t session, gnutls_packet_t *packet)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_recv_packet");
    WRAP_CHECK(gnutls_record_recv_packet, -1);
    rc = g_fn.gnutls_record_recv_packet(session, packet);

    if (rc > 0) {
        //doProtocol((uint64_t)session, -1, data, rc, TLSRX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_recv_seq(gnutls_session_t session, void *data, size_t data_size, unsigned char *seq)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_recv_seq");
    WRAP_CHECK(gnutls_record_recv_seq, -1);
    rc = g_fn.gnutls_record_recv_seq(session, data, data_size, seq);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, data, rc, TLSRX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_send(gnutls_session_t session, const void *data, size_t data_size)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_send");
    WRAP_CHECK(gnutls_record_send, -1);
    rc = g_fn.gnutls_record_send(session, data, data_size);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, (void *)data, rc, TLSTX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_send2(gnutls_session_t session, const void *data, size_t data_size,
                    size_t pad, unsigned flags)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_send2");
    WRAP_CHECK(gnutls_record_send2, -1);
    rc = g_fn.gnutls_record_send2(session, data, data_size, pad, flags);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, (void *)data, rc, TLSTX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_send_early_data(gnutls_session_t session, const void *data, size_t data_size)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_send_early_data");
    WRAP_CHECK(gnutls_record_send_early_data, -1);
    rc = g_fn.gnutls_record_send_early_data(session, data, data_size);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, (void *)data, rc, TLSTX, BUF);
    }
    return rc;
}

EXPORTON ssize_t
gnutls_record_send_range(gnutls_session_t session, const void *data, size_t data_size,
                         const gnutls_range_st *range)
{
    ssize_t rc;

    //scopeLogError("gnutls_record_send_range");
    WRAP_CHECK(gnutls_record_send_range, -1);
    rc = g_fn.gnutls_record_send_range(session, data, data_size, range);

    if (rc > 0) {
        int fd = gnutls_get_fd(session);
        doProtocol((uint64_t)session, fd, (void *)data, rc, TLSTX, BUF);
    }
    return rc;
}

static PRStatus
nss_close(PRFileDesc *fd)
{
    PRStatus rc;
    nss_list *nssentry;

    // Note: NSS docs don't define that PR_GetError should be called on failure
    if (!fd) return PR_FAILURE;

    //scopeLogError("fd:%d nss_close", (uint64_t)fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->close(fd);
    } else {
        rc = PR_FAILURE;
        DBG(NULL);
        scopeLogError("ERROR: nss_close no list entry");
        return rc;
    }

    if (rc == PR_SUCCESS) lstDelete(g_nsslist, (uint64_t)fd->methods);

    return rc;
}

static PRInt32
nss_send(PRFileDesc *fd, const void *buf, PRInt32 amount, PRIntn flags, PRIntervalTime timeout)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    /*
     * Best guess as to an error code.
     * If the call is made when fd is null, it segfaults.
     * Set the OS specific error to 0 such that the app
     * can call PR_GetOSError() as needed.
     */
    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_send", (uint64_t)fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->send(fd, buf, amount, flags, timeout);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_send no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, (void *)buf, (size_t)rc, TLSTX, BUF);
    }

    return rc;
}

static PRInt32
nss_recv(PRFileDesc *fd, void *buf, PRInt32 amount, PRIntn flags, PRIntervalTime timeout)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_recv", (uint64_t)fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->recv(fd, buf, amount, flags, timeout);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_recv no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, buf, (size_t)rc, TLSRX, BUF);
    }

    return rc;
}

static PRInt32
nss_read(PRFileDesc *fd, void *buf, PRInt32 amount)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_read", (uint64_t)fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->read(fd, buf, amount);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_read no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, buf, (size_t)rc, TLSRX, BUF);
    }

    return rc;
}

static PRInt32
nss_write(PRFileDesc *fd, const void *buf, PRInt32 amount)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_write", fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->write(fd, buf, amount);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_write no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, (void *)buf, (size_t)rc, TLSRX, BUF);
    }

    return rc;
}

static PRInt32
nss_writev(PRFileDesc *fd, const PRIOVec *iov, PRInt32 iov_size, PRIntervalTime timeout)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_writev", fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->writev(fd, iov, iov_size, timeout);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_writev no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, (void *)iov, (size_t)iov_size, TLSRX, IOV);
    }

    return rc;
}

static PRInt32
nss_sendto(PRFileDesc *fd, const void *buf, PRInt32 amount, PRIntn flags,
           const PRNetAddr *addr, PRIntervalTime timeout)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_sendto", fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->sendto(fd, (void *)buf, amount, flags, addr, timeout);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_sendto no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, (void *)buf, (size_t)amount, TLSRX, BUF);
    }

    return rc;
}

static PRInt32
nss_recvfrom(PRFileDesc *fd, void *buf, PRInt32 amount, PRIntn flags,
             PRNetAddr *addr, PRIntervalTime timeout)
{
    PRInt32 rc;
    nss_list *nssentry;
    int nfd;

    if (!fd) {
        if (SYMBOL_LOADED(PR_SetError)) {
            g_fn.PR_SetError(PR_BAD_DESCRIPTOR_ERROR, (PRInt32)0);
        }
        return -1;
    }

    //scopeLogError("fd:%d nss_recvfrom", fd->methods);
    if ((nssentry = lstFind(g_nsslist, (uint64_t)fd->methods)) != NULL) {
        rc = nssentry->ssl_methods->recvfrom(fd, buf, amount, flags, addr, timeout);
    } else {
        rc = -1;
        DBG(NULL);
        scopeLogError("ERROR: nss_recvfrom no list entry");
    }

    if (rc > 0) {
        if (SYMBOL_LOADED(PR_FileDesc2NativeHandle)) {
            nfd = g_fn.PR_FileDesc2NativeHandle(fd);
        } else {
            nfd = -1;
        }

        doProtocol((uint64_t)fd, nfd, buf, (size_t)amount, TLSRX, BUF);
    }

    return rc;
}

EXPORTON PRFileDesc *
SSL_ImportFD(PRFileDesc *model, PRFileDesc *currFd)
{
    PRFileDesc *result;

    WRAP_CHECK(SSL_ImportFD, NULL);

    result = g_fn.SSL_ImportFD(model, currFd);
    if (result != NULL) {
        nss_list *nssentry;

        if ((((nssentry = calloc(1, sizeof(nss_list))) != NULL)) &&
            ((nssentry->ssl_methods = calloc(1, sizeof(PRIOMethods))) != NULL) &&
            ((nssentry->ssl_int_methods = calloc(1, sizeof(PRIOMethods))) != NULL)) {

            memmove(nssentry->ssl_methods, result->methods, sizeof(PRIOMethods));
            memmove(nssentry->ssl_int_methods, result->methods, sizeof(PRIOMethods));
            nssentry->id = (uint64_t)nssentry->ssl_int_methods;
            //scopeLogInfo("fd:%d SSL_ImportFD", (uint64_t)nssentry->id);

            // ref contrib/tls/nss/prio.h struct PRIOMethods
            // read ... todo? read, recvfrom, acceptread
            nssentry->ssl_int_methods->recv = nss_recv;
            nssentry->ssl_int_methods->read = nss_read;
            nssentry->ssl_int_methods->recvfrom = nss_recvfrom;

            // write ... todo? write, writev, sendto, sendfile, transmitfile
            nssentry->ssl_int_methods->send = nss_send;
            nssentry->ssl_int_methods->write = nss_write;
            nssentry->ssl_int_methods->writev = nss_writev;
            nssentry->ssl_int_methods->sendto = nss_sendto;

            // close ... todo? shutdown
            nssentry->ssl_int_methods->close = nss_close;

            if (lstInsert(g_nsslist, nssentry->id, nssentry)) {
                // switch to using the wrapped methods
                result->methods = nssentry->ssl_int_methods;
            } else {
                freeNssEntry(nssentry);
            }
        } else {
            freeNssEntry(nssentry);
        }
    }
    return result;
}

EXPORTON void *
dlopen(const char *filename, int flags)
{
    void *handle;
    struct link_map *lm;
    char fbuf[256];

    fbuf[0] = '\0';
    if (flags & RTLD_LAZY) strcat(fbuf, "RTLD_LAZY|");
    if (flags & RTLD_NOW) strcat(fbuf, "RTLD_NOW|");
    if (flags & RTLD_GLOBAL) strcat(fbuf, "RTLD_GLOBAL|");
    if (flags & RTLD_LOCAL) strcat(fbuf, "RTLD_LOCAL|");
    if (flags & RTLD_NODELETE) strcat(fbuf, "RTLD_NODELETE|");
    if (flags & RTLD_NOLOAD) strcat(fbuf, "RTLD_NOLOAD|");
    if (flags & RTLD_DEEPBIND) strcat(fbuf, "RTLD_DEEPBIND|");
    scopeLog(CFG_LOG_DEBUG, "dlopen called for %s with %s", filename, fbuf);

    WRAP_CHECK(dlopen, NULL);

    /*
     * Attempting to hook a number of GOT entries based on a static list.
     * Get the link map and the ELF sections once since they are used for
     * all symbols. Then loop over the list to locate and hook appropriate
     * GOT entries.
     */
    handle = g_fn.dlopen(filename, flags);
    if (handle && (flags & RTLD_DEEPBIND)) {
        Elf64_Sym *sym = NULL;
        Elf64_Rela *rel = NULL;
        char *str = NULL;
        int i, rsz = 0;

        // Get the link map and ELF sections in advance of something matching
        if ((dlinfo(handle, RTLD_DI_LINKMAP, (void *)&lm) != -1) &&
            (getElfEntries(lm, &rel, &sym, &str, &rsz) != -1)) {
            scopeLog(CFG_LOG_DEBUG, "\tlibrary:  %s", lm->l_name);

            // for each symbol in the list try to hook
            for (i=0; hook_list[i].symbol; i++) {
                if ((dlsym(handle, hook_list[i].symbol)) &&
                    (doGotcha(lm, (got_list_t *)&hook_list[i], rel, sym, str, rsz, 0) != -1)) {
                    scopeLog(CFG_LOG_DEBUG, "\tdlopen interposed  %s", hook_list[i].symbol);
                }
            }
        }
    }

    return handle;
}

EXPORTON void
_exit(int status)
{
    handleExit();
    if (g_fn._exit) {
        g_fn._exit(status);
    } else {
        exit(status);
    }
    __builtin_unreachable();
}

#endif // __linux__

EXPORTON int
close(int fd)
{
    WRAP_CHECK(close, -1);

    if (isAnAppScopeConnection(fd)) return 0;

    int rc = g_fn.close(fd);

    doCloseAndReportFailures(fd, (rc != -1), "close");

    return rc;
}

EXPORTON int
fclose(FILE *stream)
{
    WRAP_CHECK(fclose, EOF);
    int fd = fileno(stream);

    if (isAnAppScopeConnection(fd)) return 0;

    int rc = g_fn.fclose(stream);

    doCloseAndReportFailures(fd, (rc != EOF), "fclose");

    return rc;
}

EXPORTON int
fcloseall(void)
{
    WRAP_CHECK(close, EOF);

    int rc = g_fn.fcloseall();
    if (rc != EOF) {
        doCloseAllStreams();
    } else {
        doUpdateState(FS_ERR_OPEN_CLOSE, -1, (ssize_t)0, "fcloseall", "nopath");
    }

    return rc;
}

#ifdef __APPLE__
EXPORTON int
close$NOCANCEL(int fd)
{
    WRAP_CHECK(close$NOCANCEL, -1);

    int rc = g_fn.close$NOCANCEL(fd);

    doCloseAndReportFailures(fd, (rc != -1), "close$NOCANCEL");

    return rc;
}


EXPORTON int
guarded_close_np(int fd, void *guard)
{
    WRAP_CHECK(guarded_close_np, -1);
    int rc = g_fn.guarded_close_np(fd, guard);

    doCloseAndReportFailures(fd, (rc != -1), "guarded_close_np");

    return rc;
}

EXPORTOFF int
close_nocancel(int fd)
{
    WRAP_CHECK(close_nocancel, -1);
    int rc = g_fn.close_nocancel(fd);

    doCloseAndReportFailures(fd, (rc != -1), "close_nocancel");

    return rc;
}

EXPORTON int
accept$NOCANCEL(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept$NOCANCEL, -1);
    sd = g_fn.accept$NOCANCEL(sockfd, addr, addrlen);

    if ((sd != -1) && (doBlockConnection(sockfd, NULL) == 1)) {
        if (g_fn.close) g_fn.close(sd);
        errno = ECONNABORTED;
        return -1;
    }

    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept$NOCANCEL");
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "accept$NOCANCEL", "nopath");
    }

    return sd;
}

EXPORTON ssize_t
__sendto_nocancel(int sockfd, const void *buf, size_t len, int flags,
                  const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;
    WRAP_CHECK(__sendto_nocancel, -1);
    rc = g_fn.__sendto_nocancel(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d __sendto_nocancel", sockfd);
        doSetAddrs(sockfd);

        if (remotePortIsDNS(sockfd)) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc, buf, len, BUF);
    } else {
        setRemoteClose(sockfd, errno);
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "__sendto_nocancel", "nopath");
    }

    return rc;
}

EXPORTOFF int32_t
DNSServiceQueryRecord(void *sdRef, uint32_t flags, uint32_t interfaceIndex,
                      const char *fullname, uint16_t rrtype, uint16_t rrclass,
                      void *callback, void *context)
{
    int32_t rc;
    elapsed_t time = {0};

    WRAP_CHECK(DNSServiceQueryRecord, -1);
    time.initial = getTime();
    rc = g_fn.DNSServiceQueryRecord(sdRef, flags, interfaceIndex, fullname,
                                    rrtype, rrclass, callback, context);
    time.duration = getDuration(time.initial);

    if (rc == 0) {
        scopeLog(CFG_LOG_DEBUG, "DNSServiceQueryRecord");
        doUpdateState(DNS, -1, time.duration, NULL, fullname);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, fullname);
    } else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "DNSServiceQueryRecord", fullname);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, fullname);
    }

    return rc;
}

#endif // __APPLE__

EXPORTON off_t
lseek(int fd, off_t offset, int whence)
{
    WRAP_CHECK(lseek, -1);
    off_t rc = g_fn.lseek(fd, offset, whence);

    doSeek(fd, (rc != -1), "lseek");

    return rc;
}

EXPORTON int
fseek(FILE *stream, long offset, int whence)
{
    WRAP_CHECK(fseek, -1);
    int rc = g_fn.fseek(stream, offset, whence);

    doSeek(fileno(stream), (rc != -1), "fseek");

    return rc;
}

EXPORTON int
fseeko(FILE *stream, off_t offset, int whence)
{
    WRAP_CHECK(fseeko, -1);
    int rc = g_fn.fseeko(stream, offset, whence);

    doSeek(fileno(stream), (rc != -1), "fseeko");

    return rc;
}

EXPORTON long
ftell(FILE *stream)
{
    WRAP_CHECK(ftell, -1);
    long rc = g_fn.ftell(stream);

    doSeek(fileno(stream), (rc != -1), "ftell");

    return rc;
}

EXPORTON off_t
ftello(FILE *stream)
{
    WRAP_CHECK(ftello, -1);
    off_t rc = g_fn.ftello(stream);

    doSeek(fileno(stream), (rc != -1), "ftello"); 

    return rc;
}

EXPORTON void
rewind(FILE *stream)
{
    WRAP_CHECK_VOID(rewind);
    g_fn.rewind(stream);

    doSeek(fileno(stream), TRUE, "rewind");

    return;
}

EXPORTON int
fsetpos(FILE *stream, const fpos_t *pos)
{
    WRAP_CHECK(fsetpos, -1);
    int rc = g_fn.fsetpos(stream, pos);

    doSeek(fileno(stream), (rc == 0), "fsetpos");

    return rc;
}

EXPORTON int
fgetpos(FILE *stream,  fpos_t *pos)
{
    WRAP_CHECK(fgetpos, -1);
    int rc = g_fn.fgetpos(stream, pos);

    doSeek(fileno(stream), (rc == 0), "fgetpos");

    return rc;
}

EXPORTON int
fgetpos64(FILE *stream,  fpos64_t *pos)
{
    WRAP_CHECK(fgetpos64, -1);
    int rc = g_fn.fgetpos64(stream, pos);

    doSeek(fileno(stream), (rc == 0), "fgetpos64");

    return rc;
}

/*
 * This function, at this time, is specific to libmusl.
 * We have hooked the stdout/stderr write pointer.
 * The function pointer assigned by static config is
 * modified by libc during the first call to stdout/stderr.
 * We check for changes and adjust as needed.
 */
static size_t
__stdio_write(struct MUSL_IO_FILE *stream, const unsigned char *buf, size_t len)
{
    uint64_t initialTime = getTime();
    struct iovec iovs[2] = {
		{ .iov_base = stream->wbase, .iov_len = stream->wpos - stream->wbase },
		{ .iov_base = (void *)buf, .iov_len = len }
	};

    struct iovec *iov = iovs;
    int iovcnt = 2;
    int dothis = 0;
    ssize_t rc;

    // Note: not using WRAP_CHECK because these func ptrs are not populated
    // from symbol resolution.
    // stdout
    if (g_fn.__stdout_write && stream && (stream == (struct MUSL_IO_FILE *)stdout)) {
        rc = g_fn.__stdout_write((FILE *)stream, buf, len);
        dothis = 1;

        // has the stdout write pointer changed?
        if (stream->write != (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write) {
            // save the new pointer
            g_fn.__stdout_write = (size_t (*)(FILE *, const unsigned char *, size_t))stream->write;

            // modify the pointer to use this function
            stream->write = (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write;
        }
    }

    // stderr
    if (g_fn.__stderr_write && stream && (stream == (struct MUSL_IO_FILE *)stderr)) {
        rc = g_fn.__stderr_write((FILE *)stream, buf, len);
        dothis = 1;

        // has the stderr write pointer changed?
        if (stream->write != (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write) {
            // save the new pointer
            g_fn.__stderr_write = (size_t (*)(FILE *, const unsigned char *, size_t))stream->write;

            // modify the pointer to use this function
            stream->write = (size_t (*)(FILE *, const unsigned char *, size_t))__stdio_write;
        }
    }

    if (dothis == 1) doWrite(stream->fd, initialTime, (rc != -1),
                             iov, rc, "__stdio_write", IOV, iovcnt);
    return rc;
}

EXPORTON ssize_t
write(int fd, const void *buf, size_t count)
{
    WRAP_CHECK(write, -1);
    if (g_ismusl == FALSE) {
        return __write_libc(fd, buf, count);
    }
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.write(fd, buf, count);

    doWrite(fd, initialTime, (rc != -1), buf, rc, "write", BUF, 0);

    return rc;
}

EXPORTON ssize_t
pwrite(int fd, const void *buf, size_t nbyte, off_t offset)
{
    WRAP_CHECK(pwrite, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pwrite(fd, buf, nbyte, offset);

    doWrite(fd, initialTime, (rc != -1), buf, rc, "pwrite", BUF, 0);

    return rc;
}

EXPORTON ssize_t
writev(int fd, const struct iovec *iov, int iovcnt)
{
    WRAP_CHECK(writev, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.writev(fd, iov, iovcnt);

    doWrite(fd, initialTime, (rc != -1), iov, rc, "writev", IOV, iovcnt);

    return rc;
}

EXPORTON size_t
fwrite(const void * ptr, size_t size, size_t nitems, FILE * stream)
{
    WRAP_CHECK(fwrite, 0);
    if (g_ismusl == FALSE) {
        return g_fn.fwrite(ptr, size, nitems, stream);
    }
    uint64_t initialTime = getTime();

    size_t rc = g_fn.fwrite(ptr, size, nitems, stream);

    doWrite(fileno(stream), initialTime, (rc == nitems), ptr, rc*size, "fwrite", BUF, 0);

    return rc;
}

EXPORTON int
puts(const char *s)
{
    WRAP_CHECK(puts, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.puts(s);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.puts(s);

    doWrite(fileno(stdout), initialTime, (rc != EOF), s, strlen(s), "puts", BUF, 0);

    if (rc != EOF) {
        // puts() "writes the string s and a trailing newline to stdout"
        doWrite(fileno(stdout), initialTime, TRUE, "\n", 1, "puts", BUF, 0);
    }

    return rc;
}

EXPORTON int
putchar(int c)
{
    WRAP_CHECK(putchar, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.putchar(c);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.putchar(c);

    doWrite(fileno(stdout), initialTime, (rc != EOF), &c, 1, "putchar", BUF, 0);

    return rc;
}

EXPORTON int
fputs(const char *s, FILE *stream)
{
    WRAP_CHECK(fputs, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.fputs(s, stream);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.fputs(s, stream);

    doWrite(fileno(stream), initialTime, (rc != EOF), s, strlen(s), "fputs", BUF, 0);

    return rc;
}

EXPORTON int
fputs_unlocked(const char *s, FILE *stream)
{
    WRAP_CHECK(fputs_unlocked, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.fputs_unlocked(s, stream);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.fputs_unlocked(s, stream);

    doWrite(fileno(stream), initialTime, (rc != EOF), s, strlen(s), "fputs_unlocked", BUF, 0);

    return rc;
}

EXPORTON ssize_t
read(int fd, void *buf, size_t count)
{
    WRAP_CHECK(read, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.read(fd, buf, count);

    doRead(fd, initialTime, (rc != -1), (void *)buf, rc, "read", BUF, 0);

    return rc;
}

EXPORTON ssize_t
readv(int fd, const struct iovec *iov, int iovcnt)
{
    WRAP_CHECK(readv, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.readv(fd, iov, iovcnt);

    doRead(fd, initialTime, (rc != -1), iov, rc, "readv", IOV, iovcnt);

    return rc;
}

EXPORTON ssize_t
pread(int fd, void *buf, size_t count, off_t offset)
{
    WRAP_CHECK(pread, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.pread(fd, buf, count, offset);

    doRead(fd, initialTime, (rc != -1), (void *)buf, rc, "pread", BUF, 0);

    return rc;
}

EXPORTON size_t
fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    WRAP_CHECK(fread, 0);
    uint64_t initialTime = getTime();

    size_t rc = g_fn.fread(ptr, size, nmemb, stream);

    doRead(fileno(stream), initialTime, (rc == nmemb), NULL, rc*size, "fread", NONE, 0);

    return rc;
}

EXPORTON size_t
__fread_chk(void *ptr, size_t ptrlen, size_t size, size_t nmemb, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fread_chk, 0);
    uint64_t initialTime = getTime();

    size_t rc = g_fn.__fread_chk(ptr, ptrlen, size, nmemb, stream);

    doRead(fileno(stream), initialTime, (rc == nmemb), NULL, rc*size, "__fread_chk", NONE, 0);

    return rc;
}

EXPORTON size_t
fread_unlocked(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    WRAP_CHECK(fread_unlocked, 0);
    uint64_t initialTime = getTime();

    size_t rc = g_fn.fread_unlocked(ptr, size, nmemb, stream);

    doRead(fileno(stream), initialTime, (rc == nmemb), NULL, rc*size, "fread_unlocked", NONE, 0);

    return rc;
}

EXPORTON char *
fgets(char *s, int n, FILE *stream)
{
    WRAP_CHECK(fgets, NULL);
    uint64_t initialTime = getTime();

    char* rc = g_fn.fgets(s, n, stream);

    doRead(fileno(stream), initialTime, (rc != NULL), NULL, n, "fgets", NONE, 0);

    return rc;
}

EXPORTON char *
__fgets_chk(char *s, size_t size, int strsize, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fgets_chk, NULL);
    uint64_t initialTime = getTime();

    char* rc = g_fn.__fgets_chk(s, size, strsize, stream);

    doRead(fileno(stream), initialTime, (rc != NULL), NULL, size, "__fgets_chk", NONE, 0);

    return rc;
}

EXPORTON char *
fgets_unlocked(char *s, int n, FILE *stream)
{
    WRAP_CHECK(fgets_unlocked, NULL);
    uint64_t initialTime = getTime();

    char* rc = g_fn.fgets_unlocked(s, n, stream);

    doRead(fileno(stream), initialTime, (rc != NULL), NULL, n, "fgets_unlocked", NONE, 0);

    return rc;
}

EXPORTON wchar_t *
__fgetws_chk(wchar_t *ws, size_t size, int strsize, FILE *stream)
{
    // TODO: this function aborts & exits on error, add abort functionality
    WRAP_CHECK(__fgetws_chk, NULL);
    uint64_t initialTime = getTime();

    wchar_t* rc = g_fn.__fgetws_chk(ws, size, strsize, stream);

    doRead(fileno(stream), initialTime, (rc != NULL), NULL, size*sizeof(wchar_t), "__fgetws_chk", NONE, 0);

    return rc;
}

EXPORTON wchar_t *
fgetws(wchar_t *ws, int n, FILE *stream)
{
    WRAP_CHECK(fgetws, NULL);
    uint64_t initialTime = getTime();

    wchar_t* rc = g_fn.fgetws(ws, n, stream);

    doRead(fileno(stream), initialTime, (rc != NULL), NULL, n*sizeof(wchar_t), "fgetws", NONE, 0);

    return rc;
}

EXPORTON wint_t
fgetwc(FILE *stream)
{
    WRAP_CHECK(fgetwc, WEOF);
    uint64_t initialTime = getTime();

    wint_t rc = g_fn.fgetwc(stream);

    doRead(fileno(stream), initialTime, (rc != WEOF), NULL, sizeof(wint_t), "fgetwc", NONE, 0);

    return rc;
}

EXPORTON int
fgetc(FILE *stream)
{
    WRAP_CHECK(fgetc, EOF);
    uint64_t initialTime = getTime();

    int rc = g_fn.fgetc(stream);

    doRead(fileno(stream), initialTime, (rc != EOF), NULL, 1, "fgetc", NONE, 0);

    return rc;
}

EXPORTON int
fputc(int c, FILE *stream)
{
    WRAP_CHECK(fputc, EOF);
    if(g_ismusl == FALSE) {
        return g_fn.fputc(c, stream);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.fputc(c, stream);

    doWrite(fileno(stream), initialTime, (rc != EOF), &c, 1, "fputc", NONE, 0);

    return rc;
}

EXPORTON int
fputc_unlocked(int c, FILE *stream)
{
    WRAP_CHECK(fputc_unlocked, EOF);
    if (g_ismusl == FALSE) {
        return g_fn.fputc_unlocked(c, stream);
    }
    uint64_t initialTime = getTime();

    int rc = g_fn.fputc_unlocked(c, stream);

    doWrite(fileno(stream), initialTime, (rc != EOF), &c, 1, "fputc_unlocked", NONE, 0);

    return rc;
}

EXPORTON wint_t
putwc(wchar_t wc, FILE *stream)
{
    WRAP_CHECK(putwc, WEOF);
    if (g_ismusl == FALSE) {
        return g_fn.putwc(wc, stream);
    }
    uint64_t initialTime = getTime();

    wint_t rc = g_fn.putwc(wc, stream);

    doWrite(fileno(stream), initialTime, (rc != WEOF), &wc, sizeof(wchar_t), "putwc", NONE, 0);

    return rc;
}

EXPORTON wint_t
fputwc(wchar_t wc, FILE *stream)
{
    WRAP_CHECK(fputwc, WEOF);
    if (g_ismusl == FALSE) {
        return g_fn.fputwc(wc, stream);
    }
    uint64_t initialTime = getTime();

    wint_t rc = g_fn.fputwc(wc, stream);

    doWrite(fileno(stream), initialTime, (rc != WEOF), &wc, sizeof(wchar_t), "fputwc", NONE, 0);

    return rc;
}

EXPORTOFF int
fscanf(FILE *stream, const char *format, ...)
{
    struct FuncArgs fArgs;
    LOAD_FUNC_ARGS_VALIST(fArgs, format);
    WRAP_CHECK(fscanf, EOF);
    uint64_t initialTime = getTime();

    int rc = g_fn.fscanf(stream, format,
                     fArgs.arg[0], fArgs.arg[1],
                     fArgs.arg[2], fArgs.arg[3],
                     fArgs.arg[4], fArgs.arg[5]);

    doRead(fileno(stream),initialTime, (rc != EOF), NULL, rc, "fscanf", NONE, 0);

    return rc;
}

EXPORTON ssize_t
getline (char **lineptr, size_t *n, FILE *stream)
{
    WRAP_CHECK(getline, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.getline(lineptr, n, stream);

    size_t bytes = (n) ? *n : 0;
    doRead(fileno(stream), initialTime, (rc != -1), NULL, bytes, "getline", NONE, 0);

    return rc;
}

EXPORTON ssize_t
getdelim (char **lineptr, size_t *n, int delimiter, FILE *stream)
{
    WRAP_CHECK(getdelim, -1);
    uint64_t initialTime = getTime();

    g_getdelim = 1;
    ssize_t rc = g_fn.getdelim(lineptr, n, delimiter, stream);

    size_t bytes = (n) ? *n : 0;
    doRead(fileno(stream), initialTime, (rc != -1), NULL, bytes, "getdelim", NONE, 0);

    return rc;
}

EXPORTON ssize_t
__getdelim (char **lineptr, size_t *n, int delimiter, FILE *stream)
{
    WRAP_CHECK(__getdelim, -1);
    uint64_t initialTime = getTime();

    ssize_t rc = g_fn.__getdelim(lineptr, n, delimiter, stream);
    if (g_getdelim == 1) {
        g_getdelim = 0;
        return rc;
    }

    size_t bytes = (n) ? *n : 0;
    doRead(fileno(stream), initialTime, (rc != -1), NULL, bytes, "__getdelim", NONE, 0);
    return rc;
}

EXPORTON int
fcntl(int fd, int cmd, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    int rc = g_fn.fcntl(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                    fArgs.arg[2], fArgs.arg[3]);
    if (cmd == F_DUPFD) {
        doDup(fd, rc, "fcntl", FALSE);
    }
    
    return rc;
}

EXPORTON int
fcntl64(int fd, int cmd, ...)
{
    struct FuncArgs fArgs;

    WRAP_CHECK(fcntl64, -1);
    LOAD_FUNC_ARGS_VALIST(fArgs, cmd);
    int rc = g_fn.fcntl64(fd, cmd, fArgs.arg[0], fArgs.arg[1],
                      fArgs.arg[2], fArgs.arg[3]);
    if (cmd == F_DUPFD) {
        doDup(fd, rc, "fcntl64", FALSE);
    }

    return rc;
}

EXPORTON int
dup(int fd)
{
    WRAP_CHECK(dup, -1);
    int rc = g_fn.dup(fd);
    doDup(fd, rc, "dup", TRUE);

    return rc;
}

EXPORTON int
dup2(int oldfd, int newfd)
{
    WRAP_CHECK(dup2, -1);
    int rc = g_fn.dup2(oldfd, newfd);

    doDup2(oldfd, newfd, rc, "dup2");

    return rc;
}

EXPORTON int
dup3(int oldfd, int newfd, int flags)
{
    WRAP_CHECK(dup3, -1);
    int rc = g_fn.dup3(oldfd, newfd, flags);
    doDup2(oldfd, newfd, rc, "dup3");

    return rc;
}

EXPORTOFF void
vsyslog(int priority, const char *format, va_list ap)
{
    WRAP_CHECK_VOID(vsyslog);
    scopeLog(CFG_LOG_DEBUG, "vsyslog");
    g_fn.vsyslog(priority, format, ap);
    return;
}

EXPORTON pid_t
fork()
{
    pid_t rc;

    WRAP_CHECK(fork, -1);
    scopeLog(CFG_LOG_DEBUG, "fork");
    rc = g_fn.fork();
    if (rc == 0) {
        // We are the child proc
        doReset();
    }
    
    return rc;
}

EXPORTON int
socket(int socket_family, int socket_type, int protocol)
{
    int sd;

    WRAP_CHECK(socket, -1);
    sd = g_fn.socket(socket_family, socket_type, protocol);
    if (sd != -1) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d socket", sd);
        addSock(sd, socket_type, socket_family);

        if ((socket_family == AF_INET) || (socket_family == AF_INET6)) {

            /*
             * State used in close()
             * We define that a UDP socket represents an open 
             * port when created and is open until the socket is closed
             *
             * a UDP socket is open we say the port is open
             * a UDP socket is closed we say the port is closed
             */
            doUpdateState(OPEN_PORTS, sd, 1, "socket", NULL);
        }
    } else {
        doUpdateState(NET_ERR_CONN, sd, (ssize_t)0, "socket", "nopath");
    }

    return sd;
}

EXPORTON int
shutdown(int sockfd, int how)
{
    int rc;

    WRAP_CHECK(shutdown, -1);
    rc = g_fn.shutdown(sockfd, how);
    if (rc != -1) {
        doClose(sockfd, "shutdown");
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "shutdown", "nopath");
    }

    return rc;
}

EXPORTON int
listen(int sockfd, int backlog)
{
    int rc;
    WRAP_CHECK(listen, -1);
    rc = g_fn.listen(sockfd, backlog);
    if (rc != -1) {
        scopeLog(CFG_LOG_DEBUG, "fd:%d listen", sockfd);

        doUpdateState(OPEN_PORTS, sockfd, 1, "listen", NULL);
        doUpdateState(NET_CONNECTIONS, sockfd, 1, "listen", NULL);
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "listen", "nopath");
    }

    return rc;
}

EXPORTON int
accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)
{
    int sd;

    WRAP_CHECK(accept, -1);
    sd = g_fn.accept(sockfd, addr, addrlen);

    if ((sd != -1) && (doBlockConnection(sockfd, NULL) == 1)) {
        if (g_fn.close) g_fn.close(sd);
        errno = ECONNABORTED;
        return -1;
    }

    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept");
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "accept", "nopath");
    }

    return sd;
}

EXPORTON int
accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags)
{
    int sd;

    WRAP_CHECK(accept4, -1);
    sd = g_fn.accept4(sockfd, addr, addrlen, flags);

    if ((sd != -1) && (doBlockConnection(sockfd, NULL) == 1)) {
        if (g_fn.close) g_fn.close(sd);
        errno = ECONNABORTED;
        return -1;
    }

    if (sd != -1) {
        doAccept(sd, addr, addrlen, "accept4");
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "accept4", "nopath");
    }

    return sd;
}

EXPORTON int
bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;

    WRAP_CHECK(bind, -1);
    rc = g_fn.bind(sockfd, addr, addrlen);
    if (rc != -1) { 
        doSetConnection(sockfd, addr, addrlen, LOCAL);
        scopeLog(CFG_LOG_DEBUG, "fd:%d bind", sockfd);
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "bind", "nopath");
    }

    return rc;

}

EXPORTON int
connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
{
    int rc;
    WRAP_CHECK(connect, -1);
    if (doBlockConnection(sockfd, addr) == 1) {
        errno = ECONNREFUSED;
        return -1;
    }

    rc = g_fn.connect(sockfd, addr, addrlen);
    if (rc != -1) {
        doSetConnection(sockfd, addr, addrlen, REMOTE);
        doUpdateState(NET_CONNECTIONS, sockfd, 1, "connect", NULL);

        scopeLog(CFG_LOG_DEBUG, "fd:%d connect", sockfd);
    } else {
        doUpdateState(NET_ERR_CONN, sockfd, (ssize_t)0, "connect", "nopath");
    }

    return rc;
}

EXPORTON ssize_t
send(int sockfd, const void *buf, size_t len, int flags)
{
    ssize_t rc;
    WRAP_CHECK(send, -1);
    doURL(sockfd, buf, len, NETTX);
    rc = g_fn.send(sockfd, buf, len, flags);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d send", sockfd);
        if (remotePortIsDNS(sockfd)) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc, buf, rc, BUF);
    } else {
        setRemoteClose(sockfd, errno);
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "send", "nopath");
    }

    return rc;
}

static ssize_t
internal_sendto(int sockfd, const void *buf, size_t len, int flags,
                const struct sockaddr *dest_addr, socklen_t addrlen)
{
    ssize_t rc;
    WRAP_CHECK(sendto, -1);
    rc = g_fn.sendto(sockfd, buf, len, flags, dest_addr, addrlen);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d sendto", sockfd);
        doSetConnection(sockfd, dest_addr, addrlen, REMOTE);

        if (remotePortIsDNS(sockfd)) {
            getDNSName(sockfd, (void *)buf, len);
        }

        doSend(sockfd, rc, buf, rc, BUF);
    } else {
        setRemoteClose(sockfd, errno);
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "sendto", "nopath");
    }

    return rc;
}

EXPORTON ssize_t
sendto(int sockfd, const void *buf, size_t len, int flags,
       const struct sockaddr *dest_addr, socklen_t addrlen)
{
    return internal_sendto(sockfd, buf, len, flags, dest_addr, addrlen);
}

EXPORTON ssize_t
sendmsg(int sockfd, const struct msghdr *msg, int flags)
{
    ssize_t rc;
    
    WRAP_CHECK(sendmsg, -1);
    rc = g_fn.sendmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d sendmsg", sockfd);

        // For UDP connections the msg is a remote addr
        if (msg && !sockIsTCP(sockfd)) {
            if (msg->msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msg->msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (remotePortIsDNS(sockfd)) {
            getDNSName(sockfd, msg->msg_iov->iov_base, msg->msg_iov->iov_len);
        }

        doSend(sockfd, rc, msg, rc, MSG);
    } else {
        setRemoteClose(sockfd, errno);
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "sendmsg", "nopath");
    }

    return rc;
}

#ifdef __linux__
static int
internal_sendmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags)
{
    ssize_t rc;

    WRAP_CHECK(sendmmsg, -1);
    rc = g_fn.sendmmsg(sockfd, msgvec, vlen, flags);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d sendmmsg", sockfd);

        // For UDP connections the msg is a remote addr
        if (!sockIsTCP(sockfd)) {
            if (msgvec->msg_hdr.msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msgvec->msg_hdr.msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msgvec->msg_hdr.msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msgvec->msg_hdr.msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (remotePortIsDNS(sockfd)) {
            getDNSName(sockfd, msgvec->msg_hdr.msg_iov->iov_base, msgvec->msg_hdr.msg_iov->iov_len);
        }

        doSend(sockfd, rc, &msgvec->msg_hdr, rc, MSG);

    } else {
        setRemoteClose(sockfd, errno);
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "sendmmsg", "nopath");
    }

    return rc;
}

EXPORTON int
sendmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen, int flags)
{
    return internal_sendmmsg(sockfd, msgvec, vlen, flags);
}
#endif // __linux__

EXPORTON ssize_t
recv(int sockfd, void *buf, size_t len, int flags)
{
    ssize_t rc;

    WRAP_CHECK(recv, -1);
    scopeLog(CFG_LOG_TRACE, "fd:%d recv", sockfd);
    if ((rc = doURL(sockfd, buf, len, NETRX)) == 0) {
        rc = g_fn.recv(sockfd, buf, len, flags);
    }

    if (rc != -1) {
        // it's possible to get DNS over TCP
        if (remotePortIsDNS(sockfd)) {
            getDNSAnswer(sockfd, buf, rc, BUF);
        }

        doRecv(sockfd, rc, buf, rc, BUF);
    } else {
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "recv", "nopath");
    }

    return rc;
}

static ssize_t
internal_recvfrom(int sockfd, void *buf, size_t len, int flags,
         struct sockaddr *src_addr, socklen_t *addrlen)
{
    ssize_t rc;

    WRAP_CHECK(recvfrom, -1);
    rc = g_fn.recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d recvfrom", sockfd);

        if (remotePortIsDNS(sockfd)) {
            getDNSAnswer(sockfd, buf, rc, BUF);
        }

        doRecv(sockfd, rc, buf, rc, BUF);
    } else {
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "recvfrom", "nopath");
    }
    return rc;
}

EXPORTON ssize_t
recvfrom(int sockfd, void *buf, size_t len, int flags,
         struct sockaddr *src_addr, socklen_t *addrlen)
{
    return internal_recvfrom(sockfd, buf, len, flags, src_addr, addrlen);
}

static int
doAccessRights(struct msghdr *msg)
{
    int *recvfd;
    struct cmsghdr *cmptr;
    struct stat sbuf;

    if (!msg) return -1;

    if (((cmptr = CMSG_FIRSTHDR(msg)) != NULL) &&
        (cmptr->cmsg_len >= CMSG_LEN(sizeof(int))) &&
        (cmptr->cmsg_level == SOL_SOCKET) &&
        (cmptr->cmsg_type == SCM_RIGHTS)) {
        // voila; we have a new fd
        int i, numfds;

        numfds = (cmptr->cmsg_len - CMSG_ALIGN(sizeof(struct cmsghdr))) / sizeof(int);
        if (numfds <= 0) return -1;
        recvfd = ((int *) CMSG_DATA(cmptr));

        for (i = 0; i < numfds; i++) {
            // file or socket?
            if (fstat(recvfd[i], &sbuf) != -1) {
                if ((sbuf.st_mode & S_IFMT) == S_IFSOCK) {
                    doAddNewSock(recvfd[i]);
                } else {
                    doOpen(recvfd[i], "Received_File_Descriptor", FD, "recvmsg");
                }
            } else {
                DBG("errno: %d", errno);
                return -1;
            }
        }
    }

    return 0;
}

EXPORTON ssize_t
recvmsg(int sockfd, struct msghdr *msg, int flags)
{
    ssize_t rc;
    
    WRAP_CHECK(recvmsg, -1);
    rc = g_fn.recvmsg(sockfd, msg, flags);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d recvmsg", sockfd);

        // For UDP connections the msg is a remote addr
        if (msg) {
            if (msg->msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msg->msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msg->msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (remotePortIsDNS(sockfd)) {
            getDNSAnswer(sockfd, (char *)msg, rc, MSG);
        }

        doRecv(sockfd, rc, msg, rc, MSG);
        doAccessRights(msg);
    } else {
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "recvmsg", "nopath");
    }
    
    return rc;
}

#ifdef __linux__
EXPORTON int
recvmmsg(int sockfd, struct mmsghdr *msgvec, unsigned int vlen,
         int flags, struct timespec *timeout)
{
    ssize_t rc;

    WRAP_CHECK(recvmmsg, -1);
    rc = g_fn.recvmmsg(sockfd, msgvec, vlen, flags, timeout);
    if (rc != -1) {
        scopeLog(CFG_LOG_TRACE, "fd:%d recvmmsg", sockfd);

        // For UDP connections the msg is a remote addr
        if (msgvec) {
            if (msgvec->msg_hdr.msg_namelen >= sizeof(struct sockaddr_in6)) {
                doSetConnection(sockfd, (const struct sockaddr *)msgvec->msg_hdr.msg_name,
                                sizeof(struct sockaddr_in6), REMOTE);
            } else if (msgvec->msg_hdr.msg_namelen >= sizeof(struct sockaddr_in)) {
                doSetConnection(sockfd, (const struct sockaddr *)msgvec->msg_hdr.msg_name,
                                sizeof(struct sockaddr_in), REMOTE);
            }
        }

        if (remotePortIsDNS(sockfd)) {
            getDNSAnswer(sockfd, (char *)&msgvec->msg_hdr, rc, MSG);
        }

        doRecv(sockfd, rc, &msgvec->msg_hdr, rc, MSG);
        doAccessRights(&msgvec->msg_hdr);
    } else {
        doUpdateState(NET_ERR_RX_TX, sockfd, (ssize_t)0, "recvmmsg", "nopath");
    }

    return rc;
}
#endif //__linux__

EXPORTOFF struct hostent *
gethostbyname(const char *name)
{
    struct hostent *rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname, NULL);
    doUpdateState(DNS, -1, 0, NULL, name);
    time.initial = getTime();
    rc = g_fn.gethostbyname(name);
    time.duration = getDuration(time.initial);

    if (rc != NULL) {
        scopeLog(CFG_LOG_DEBUG, "gethostbyname");
        doUpdateState(DNS, -1, time.duration, NULL, name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    } else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "gethostbyname", name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }

    return rc;
}

EXPORTOFF struct hostent *
gethostbyname2(const char *name, int af)
{
    struct hostent *rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(gethostbyname2, NULL);
    doUpdateState(DNS, -1, 0, NULL, name);
    time.initial = getTime();
    rc = g_fn.gethostbyname2(name, af);
    time.duration = getDuration(time.initial);

    if (rc != NULL) {
        scopeLog(CFG_LOG_DEBUG, "gethostbyname2");
        doUpdateState(DNS, -1, time.duration, NULL, name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    } else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "gethostbyname2", name);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, name);
    }

    return rc;
}

/*
 * we use this to get the DNS request if sendmmsg
 * is not funchooked or if the lib uses a different
 * internal function to send the dns request.
 */
EXPORTOFF int
getaddrinfo(const char *node, const char *service,
            const struct addrinfo *hints,
            struct addrinfo **res)
{
    int rc;
    elapsed_t time = {0};
    
    WRAP_CHECK(getaddrinfo, -1);

    doUpdateState(DNS, -1, 0, NULL, node);
    time.initial = getTime();
    rc = g_fn.getaddrinfo(node, service, hints, res);
    time.duration = getDuration(time.initial);

    if (rc == 0) {
        scopeLog(CFG_LOG_DEBUG, "getaddrinfo");
        doUpdateState(DNS, -1, time.duration, NULL, node);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, node);
    } else {
        doUpdateState(NET_ERR_DNS, -1, (ssize_t)0, "getaddrinfo", node);
        doUpdateState(DNS_DURATION, -1, time.duration, NULL, node);
    }

    return rc;
}

// This was added to avoid having libscope.so depend on GLIBC_2.25.
// libssl.a or libcrypto.a need getentropy which only exists in
// GLIBC_2.25 and newer.
EXPORTON int
getentropy(void *buffer, size_t length)
{
    if (SYMBOL_LOADED(getentropy)) {
        return g_fn.getentropy(buffer, length);
    }

    // Must be something older than GLIBC_2.25...
    // Looks like we're on the hook for this.

#ifndef SYS_getrandom
    errno = ENOSYS;
    return -1;
#else
    if (length > 256) {
        errno = EIO;
        return -1;
    }

    int cancel;
    int ret = 0;
    char *pos = buffer;
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancel);
    while (length) {
        ret = syscall(SYS_getrandom, pos, length, 0);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        pos += ret;
        length -= ret;
        ret = 0;
    }
    pthread_setcancelstate(cancel, 0);
    return ret;
#endif
}

#define LOG_BUF_SIZE 4096

// This overrides a weak definition in src/dbg.c
void
scopeLog(cfg_log_level_t level, const char *format, ...)
{
    char scope_log_var_buf[LOG_BUF_SIZE];
    const char overflow_msg[] = "WARN: scopeLog msg truncated.\n";
    char *local_buf;

    if (!g_log) {
        if (!g_constructor_debug_enabled) return;
        local_buf = scope_log_var_buf + snprintf(scope_log_var_buf, LOG_BUF_SIZE, "Constructor: (pid:%d): ", getpid());
        size_t local_buf_len = sizeof(scope_log_var_buf) + (scope_log_var_buf - local_buf) - 1;

        va_list args;
        va_start(args, format);
        int msg_len = vsnprintf(local_buf, local_buf_len, format, args);
        va_end(args);
        if (msg_len == -1) {
            DBG(NULL);
            return;
        }

        sprintf(local_buf + msg_len, "\n");
        local_buf += msg_len + 1;

        if (DEFAULT_LOG_LEVEL > level) return;

        int fd = scope_open(DEFAULT_LOG_PATH);
        if (fd == -1) {
            DBG(NULL);
            return;
        }

        if (msg_len >= local_buf_len) {
            DBG(NULL);
            scope_write(fd, overflow_msg, sizeof(overflow_msg));
        } else {
            scope_write(fd, scope_log_var_buf, local_buf - scope_log_var_buf);
        }
        scope_close(fd);
        return;
    }

    cfg_log_level_t cfg_level = logLevel(g_log);
    if ((cfg_level == CFG_LOG_NONE) || (cfg_level > level)) return;

    local_buf = scope_log_var_buf + snprintf(scope_log_var_buf, LOG_BUF_SIZE, "Scope: %s(pid:%d): ", g_proc.procname, g_proc.pid);
    size_t local_buf_len = sizeof(scope_log_var_buf) + (scope_log_var_buf - local_buf) - 1;

    va_list args;
    va_start(args, format);
    int msg_len = vsnprintf(local_buf, local_buf_len, format, args);
    va_end(args);
    if (msg_len == -1) {
        DBG(NULL);
        return;
    }
    sprintf(local_buf + msg_len, "\n");
    local_buf += msg_len + 1;

    if (msg_len >= local_buf_len) {
        DBG(NULL);
        logSend(g_log, overflow_msg, level);
    } else {
        logSend(g_log, scope_log_var_buf, level);
    }
}

/*
 * pthread_create was added to support go execs on libmusl.
 * The go execs don't link to crt0/crt1 on libmusl therefore
 * they do not call our lib constructor. We interpose this
 * as a means to call the constructor before the go app runs.
 */
EXPORTON int
pthread_create(pthread_t *thread, const pthread_attr_t *attr,
               void *(*start_routine)(void *), void *arg)
{
    int rc;

    WRAP_CHECK(pthread_create, -1);
    rc = g_fn.pthread_create(thread, attr, start_routine, arg);

    if (!g_ctl) {
        init();
    }

    return rc;
}

/*
 * These functions are interposed to support libmusl.
 * The addition of libssl and libcrypto pull in these
 * glibc internal funcs.
 */
EXPORTON int
__fprintf_chk(FILE *stream, int flag, const char *format, ...)
{
    va_list ap;
    int rc;

    va_start (ap, format);
    rc = vfprintf(stream, format, ap);
    va_end (ap);
    return rc;
}

EXPORTON int
__sprintf_chk(char *str, int flag, size_t strlen, const char *format, ...)
{
    va_list ap;
    int rc;

    va_start(ap, format);
    rc = vsnprintf(str, strlen, format, ap);
    va_end(ap);
    return rc;
}

EXPORTON void *
__memset_chk(void *dest, int cset, size_t len, size_t destlen)
{
    if (g_fn.__memset_chk) {
        return g_fn.__memset_chk(dest, cset, len, destlen);
    }

    return memset(dest, cset, len);
}

EXPORTON void *
__memcpy_chk(void *dest, const void *src, size_t len, size_t destlen)
{
    if (g_fn.__memcpy_chk) {
        return g_fn.__memcpy_chk(dest, src, len, destlen);
    }

    return memcpy(dest, src, len);
}

EXPORTON long int
__fdelt_chk(long int fdelt)
{
    if (g_fn.__fdelt_chk) {
        return g_fn.__fdelt_chk(fdelt);
    }

    if (fdelt < 0 || fdelt >= FD_SETSIZE) {
        DBG(NULL);
        fprintf(stderr, "__fdelt_chk error: buffer overflow detected?\n");
        abort();
    }

    return fdelt / __NFDBITS;
}

/*
 * This is a libuv specific function intended to map an SSL ID to a fd.
 * This libuv function is called in the same call stack as SSL_read.
 * Therefore, we extract the fd here and use it in a subsequent SSL_read/write.
 */
static void
uv__read_hook(void *stream)
{
    if (SYMBOL_LOADED(uv_fileno)) g_fn.uv_fileno(stream, &g_ssl_fd);
    //scopeLog(CFG_LOG_TRACE, "%s: fd %d", __FUNCTION__, g_ssl_fd);
    if (g_fn.uv__read) return g_fn.uv__read(stream);
}

EXPORTWEAK int
__register_atfork(void (*prepare) (void), void (*parent) (void), void (*child) (void), void *__dso_handle)
{
    if (g_fn.__register_atfork) {
        return g_fn.__register_atfork(prepare, parent, child, __dso_handle);
    }

    /*
     * What do we do if we can't resolve a symbol for __register_atfork?
     * glibc returns ENOMEM on error.
     *
     * Note: __register_atfork() is defined to implement the
     * functionality of pthread_atfork(); Therefore, it would seem
     * reasonable to call pthread_atfork() here if the symbol for
     * __register_atfork() is not resolved. However, glibc implements
     * pthread_atfork() by calling __register_atfork() which causes
     * a tight loop here and we would crash.
     */
    return ENOMEM;
}

EXPORTWEAK int
__vfprintf_chk(FILE *fp, int flag, const char *format, va_list ap)
{
    return vfprintf(fp, format, ap);
}

EXPORTWEAK int
__vsnprintf_chk(char *s, size_t maxlen, int flag, size_t slen, const char *format, va_list args)
{
    return vsnprintf(s, slen, format, args);
}

EXPORTWEAK void
__longjmp_chk(jmp_buf env, int val)
{
    longjmp(env, val);
}


static void *
scope_dlsym(void *handle, const char *name, void *who)
{
    return dlsym(handle, name);
}

