#define _GNU_SOURCE
#include <sys/param.h>
#include <time.h>
#include "os.h"
#include "../../src/dbg.h"
#include "../../src/fn.h"
#include "../../src/scopetypes.h"

static int
sendNL(int sd, ino_t node)
{
    if (!g_fn.sendmsg) return -1;

    struct sockaddr_nl nladdr = {
        .nl_family = AF_NETLINK
    };

    struct {
        struct nlmsghdr nlh;
        struct unix_diag_req udr;
    } req = {
        .nlh = {
            .nlmsg_len = sizeof(req),
            .nlmsg_type = SOCK_DIAG_BY_FAMILY,
            .nlmsg_flags = NLM_F_REQUEST
        },
        .udr = {
            .sdiag_family = AF_UNIX,
            .sdiag_protocol = 0,
            .pad = 0,
            .udiag_states = -1,
            .udiag_ino = node,
            .udiag_cookie[0] = -1,
            .udiag_cookie[1] = -1,
            .udiag_show = UDIAG_SHOW_PEER
        }
    }; // reminder cookies must be -1 in order for a single request to work

    struct iovec iov = {
        .iov_base = &req,
        .iov_len = sizeof(req)
    };

    struct msghdr msg = {
        .msg_name = (void *) &nladdr,
        .msg_namelen = sizeof(nladdr),
        .msg_iov = &iov,
        .msg_iovlen = 1
    };

    // should we check for a partial send?
    if (g_fn.sendmsg(sd, &msg, 0) < 0) {
        scopeLog("ERROR:sendNL:sendmsg", sd, LOG_LEVEL);
        return -1;
    }

    return 0;
}

static ino_t
getNL(int sd)
{
    if (!g_fn.recvmsg) return -1;

    int rc;
    char buf[sizeof(struct nlmsghdr) + (sizeof(long) * 4)];
    struct unix_diag_msg *diag;
    struct rtattr *attr;
    struct sockaddr_nl nladdr = {
        .nl_family = AF_NETLINK
    };

    struct iovec iov = {
        .iov_base = buf,
        .iov_len = sizeof(buf)
    };

    struct msghdr msg = {
        .msg_name = (void *) &nladdr,
        .msg_namelen = sizeof(nladdr),
        .msg_iov = &iov,
        .msg_iovlen = 1
    };

    if ((rc = g_fn.recvmsg(sd, &msg, 0)) <= 0) {
        scopeLog("ERROR:getNL:recvmsg", sd, LOG_LEVEL);
        return (ino_t)-1;
    }

    const struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;

    if (!NLMSG_OK(nlhdr, rc)) {
        scopeLog("ERROR:getNL:!NLMSG_OK", sd, LOG_LEVEL);
        return (ino_t)-1;
    }

    for (; NLMSG_OK(nlhdr, rc); nlhdr = NLMSG_NEXT(nlhdr, rc)) {
        if (nlhdr->nlmsg_type == NLMSG_DONE) {
            scopeLog("ERROR:getNL:no message", sd, LOG_LEVEL);
            return (ino_t)-1;
        }

        if (nlhdr->nlmsg_type == NLMSG_ERROR) {
            const struct nlmsgerr *err = NLMSG_DATA(nlhdr);

            if (nlhdr->nlmsg_len < NLMSG_LENGTH(sizeof(*err))) {
                scopeLog("ERROR:getNL:message error", sd, LOG_LEVEL);
            } else {
                char buf[64];
                snprintf(buf, sizeof(buf), "ERROR:getNL:message errno %d", -err->error);
                scopeLog(buf, sd, LOG_LEVEL);
            }

            return (ino_t)-1;
        }

        if (nlhdr->nlmsg_type != SOCK_DIAG_BY_FAMILY) {
            scopeLog("ERROR:getNL:unexpected nlmsg_type", sd, LOG_LEVEL);
            return (ino_t)-1;
        }

        if ((diag = NLMSG_DATA(nlhdr)) != NULL) {
            if (nlhdr->nlmsg_len < NLMSG_LENGTH(sizeof(*diag))) {
                scopeLog("ERROR:getNL:short response", sd, LOG_LEVEL);
                return (ino_t)-1;
            }

            if (diag->udiag_family != AF_UNIX) {
                scopeLog("ERROR:getNL:unexpected family", sd, LOG_LEVEL);
                return (ino_t)-1;
            }

            attr = (struct rtattr *) (diag + 1);
            if (attr->rta_type == UNIX_DIAG_PEER) {
                if (RTA_PAYLOAD(attr) >= sizeof(unsigned int)) {
                    return (ino_t)*(unsigned int *) RTA_DATA(attr);
                }
            }
        }
    }
    return (ino_t)-1;
}

int
osUnixSockPeer(ino_t lnode)
{
    int nsd;
    ino_t rnode;

    if (!g_fn.socket || !g_fn.close) return -1;

    if ((nsd = g_fn.socket(AF_NETLINK, SOCK_RAW, NETLINK_SOCK_DIAG)) == -1) return -1;

    if (sendNL(nsd, lnode) == -1) {
        g_fn.close(nsd);
        return -1;
    }

    rnode = getNL(nsd);
    g_fn.close(nsd);
    return rnode;
}

char *
osGetExePath()
{
    char *path;

    if (!(path = calloc(1, PATH_MAX))) {
        scopeLog("ERROR:calloc in osGetExePath", -1, CFG_LOG_ERROR);
        return NULL;
    }

    if (readlink("/proc/self/exe", path, PATH_MAX - 1) == -1) {
        scopeLog("osGetPath: can't get path to self exe", -1, CFG_LOG_ERROR);
        free(path);
        return NULL;
    }

    return path;
}

int
osGetProcname(char *pname, int len)
{
    strncpy(pname, program_invocation_short_name, len);
    return 0;
}

int
osGetProcMemory(pid_t pid)
{
    int fd;
    long result;
    char *start, *entry, *last;
    const char delim[] = ":";
    char buf[2048];

    if (!g_fn.open || !g_fn.read || !g_fn.close) {
        return -1;
    }

    snprintf(buf, sizeof(buf), "/proc/%d/status", pid);
    if ((fd = g_fn.open(buf, O_RDONLY)) == -1) {
        DBG(NULL);
        return -1;
    }

    if (g_fn.read(fd, buf, sizeof(buf)) == -1) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;
    }

    if ((start = strstr(buf, "VmSize")) == NULL) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;        
    }
    
    entry = strtok_r(start, delim, &last);
    entry = strtok_r(NULL, delim, &last);
    if (entry == NULL) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;        
    }
    
    if ((result = strtol(entry, NULL, 0)) == (long)0) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;
    }
    
    g_fn.close(fd);
    return (int)result;
}

int
osGetNumThreads(pid_t pid)
{
    int fd, i;
    long result;
    char *entry, *last;
    const char delim[] = " ";
    char buf[1024];

    if (!g_fn.open || !g_fn.read || !g_fn.close) {
        return -1;
    }

    // Get the size of the file with stat, malloc buf then free
    snprintf(buf, sizeof(buf), "/proc/%d/stat", pid);
    if ((fd = g_fn.open(buf, O_RDONLY)) == -1) {
        DBG(NULL);
        return -1;
    }

    if (g_fn.read(fd, buf, sizeof(buf)) == -1) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;
    }

    entry = strtok_r(buf, delim, &last);
    for (i = 1; i < 20; i++) {
        entry = strtok_r(NULL, delim, &last);
    }
    g_fn.close(fd);

    if ((result = strtol(entry, NULL, 0)) == (long)0) {
        DBG(NULL);
        return -1;
    }
    return (int)result;
}

int
osGetNumFds(pid_t pid)
{
    int nfile = 0;
    DIR * dirp;
    struct dirent * entry;
    char buf[1024];

    snprintf(buf, sizeof(buf), "/proc/%d/fd", pid);
    if ((dirp = opendir(buf)) == NULL) {
        DBG(NULL);
        return -1;
    }
    
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type != DT_DIR) {
            nfile++;
        }
    }

    closedir(dirp);
    return nfile - 1; // we opened one fd to read /fd :)
}

int
osGetNumChildProcs(pid_t pid)
{
    int nchild = 0;
    DIR * dirp;
    struct dirent * entry;
    char buf[1024];

    snprintf(buf, sizeof(buf), "/proc/%d/task", pid);
    if ((dirp = opendir(buf)) == NULL) {
        DBG(NULL);
        return -1;
    }
    
    while ((entry = readdir(dirp)) != NULL) {
            nchild++;
    }

    closedir(dirp);
    return nchild - 3; // Not including the parent proc itself and ., ..
}

int
osInitTSC(platform_time_t *cfg)
{
    int fd;
    char *entry, *last;
    const char delim[] = ":";
    const char path[] = "/proc/cpuinfo";
    const char freqStr[] = "cpu MHz";
    char *buf;

    if (!g_fn.open || !g_fn.read || !g_fn.close) {
        return -1;
    }

    if ((fd = g_fn.open(path, O_RDONLY)) == -1) {
        DBG(NULL);
        return -1;
    }
    
    /*
     * Anecdotal evidence that there is a max size to proc entrires.
     * In any case this should be big enough.
     */    
    if ((buf = calloc(1, MAX_PROC)) == NULL) {
        DBG(NULL);
        g_fn.close(fd);
        return -1;
    }
    
    if (g_fn.read(fd, buf, MAX_PROC) == -1) {
        DBG(NULL);
        g_fn.close(fd);
        free(buf);
        return -1;
    }

    if (strstr(buf, "rdtscp") == NULL) {
        cfg->tsc_rdtscp = FALSE;
    } else {
        cfg->tsc_rdtscp = TRUE;
    }
    
    if (strstr(buf, "tsc_reliable") == NULL) {
        cfg->tsc_invariant = FALSE;
    } else {
        cfg->tsc_invariant = TRUE;
    }
    
    entry = strtok_r(buf, delim, &last);
    while (1) {
        if ((entry = strtok_r(NULL, delim, &last)) == NULL) {
            cfg->freq = (uint64_t)-1;
            break;
        }

        if (strcasestr((const char *)entry, freqStr) != NULL) {
            // The next token should be what we want
            if ((entry = strtok_r(NULL, delim, &last)) != NULL) {
                if ((cfg->freq = (uint64_t)strtoll(entry, NULL, 0)) == (long long)0) {
                    cfg->freq = (uint64_t)-1;
                }
                break;
            }
        }
    }
    
    g_fn.close(fd);
    free(buf);
    if (cfg->freq == (uint64_t)-1) {
        DBG(NULL);
        return -1;
    }
    return 0;
}

int
osIsFilePresent(pid_t pid, const char *path)
{
    struct stat sb = {0};

    if (!g_fn.__xstat) {
        return -1;
    }

    if (g_fn.__xstat(_STAT_VER, path, &sb) != 0) {
        return -1;
    } else {
        return sb.st_size;
    }
}

int
osGetCmdline(pid_t pid, char **cmd)
{
    int fd = -1;
    int bytesRead = 0;
    char path[64];

    if (!cmd) return 0;
    char* buf = *cmd;
    buf = NULL;

    if (!g_fn.open || !g_fn.read || !g_fn.close) {
        goto out;
    }

    if ((buf = calloc(1, NCARGS)) == NULL) {
        goto out;
    }

    if (snprintf(path, sizeof(path), "/proc/%d/cmdline", pid) < 0) {
        goto out;
    }

    if ((fd = g_fn.open(path, O_RDONLY)) == -1) {
        DBG(NULL);
        goto out;
    }

    if ((bytesRead = g_fn.read(fd, buf, NCARGS)) <= 0) {
        DBG(NULL);
        goto out;
    }

    // Replace all but the last null with spaces
    int i;
    for (i=0; i < (bytesRead - 1); i++) {
        if (buf[i] == '\0') buf[i] = ' ';
    }

out:
    if (!buf || !buf[0]) {
        if (buf) free(buf);
        buf = strdup("none");
    } else {
        // buf is big; try to strdup what we've used and free the rest
        char* tmp = strdup(buf);
        if (tmp) {
            free(buf);
            buf = tmp;
        }
    }
    if (fd != -1) g_fn.close(fd);
    *cmd = buf;
    return (*cmd != NULL);
}

bool
osThreadInit(void(*handler)(int), unsigned interval)
{
    struct sigaction sact;
    struct sigevent sevent = {0};
    timer_t timerid = 0;
    struct itimerspec tspec;

    sigemptyset(&sact.sa_mask);
    sact.sa_handler = handler;
    sact.sa_flags = 0;

    if (!g_fn.sigaction) return FALSE;

    if (g_fn.sigaction(SIGUSR2, &sact, NULL) == -1) {
        DBG("errno %d", errno);
        return FALSE;
    }

    sevent.sigev_notify = SIGEV_SIGNAL;
    sevent.sigev_signo = SIGUSR2;
    if (timer_create(CLOCK_MONOTONIC, &sevent, &timerid) == -1) {
        DBG("errno %d", errno);
        return FALSE;
    }

    tspec.it_interval.tv_sec = 0;
    tspec.it_interval.tv_nsec = 0;
    tspec.it_value.tv_sec = interval;
    tspec.it_value.tv_nsec = 0;
    if (timer_settime(timerid, 0, &tspec, NULL) == -1) {
        DBG("errno %d", errno);
        return FALSE;
    }
    return TRUE;
}

// In linux, this is declared weak so it can be overridden by the strong
// definition in src/javaagent.c.  The scope library will do this.
// This weak definition allows us to not have to define this symbol for
// unit tests or for the scope executable.
void __attribute__((weak))
initJavaAgent() {
     return;
}

void
osInitJavaAgent(void)
{
    initJavaAgent();
}


/*
 * Example from /proc/self/maps:
 * 7f1b23bd4000-7f1b23bd7000 rw-p 001e3000 08:01 402063 /usr/lib/x86_64-linux-gnu/libc-2.29.so
 */
int
osGetPageProt(uint64_t addr)
{
    int prot = -1;
    size_t len = 0;
    char *buf = NULL;
    char log[128];

    if (!g_fn.fopen || !g_fn.getline || !g_fn.fclose) {
        return -1;
    }

    if (addr == 0) {
        return -1;
    }

    FILE *fstream = g_fn.fopen("/proc/self/maps", "r");
    if (fstream == NULL) return -1;

    while (g_fn.getline(&buf, &len, fstream) != -1) {
        char *end = NULL;
        errno = 0;
        uint64_t addr1 = strtoull(buf, &end, 0x10);
        if ((addr1 == 0) || (errno != 0)) {
            if (buf) free(buf);
            g_fn.fclose(fstream);
            return -1;
        }

        uint64_t addr2 = strtoull(end + 1, &end, 0x10);
        if ((addr2 == 0) || (errno != 0)) {
            if (buf) free(buf);
            g_fn.fclose(fstream);
            return -1;
        }

        snprintf(log, sizeof(log), "addr 0x%lux addr1 0x%lux addr2 0x%lux\n", addr, addr1, addr2);
        scopeLog(log, -1, CFG_LOG_TRACE);

        if ((addr >= addr1) && (addr <= addr2)) {
            char *perms = end + 1;
            snprintf(log, sizeof(log), "matched 0x%lx to 0x%lx-0x%lx\n\t%c%c%c",
                     addr, addr1, addr2, perms[0], perms[1], perms[2]);
            scopeLog(log, -1, CFG_LOG_DEBUG);
            prot = 0;
            prot |= perms[0] == 'r' ? PROT_READ : 0;
            prot |= perms[1] == 'w' ? PROT_WRITE : 0;
            prot |= perms[2] == 'x' ? PROT_EXEC : 0;
            if (buf) free(buf);
            break;
        }

        if (buf) {
            free(buf);
            buf = NULL;
        }

        len = 0;
    }

    g_fn.fclose(fstream);
    return prot;
}

static const char scope_help_overview[] =
"OVERVIEW:\n"
"    The Scope library supports extraction of data from within applications.\n"
"    As a general rule applications consist of one or more processes.\n"
"    The Scope library is capable of being loaded into any process as the\n"
"    process starts.\n"
"    The primary means of defining which processes include the Scope library\n"
"    is by exporting the environment variable LD_PRELOAD. The LD_PRELOAD\n"
"    variable is set to point to the path name of the Scope library.\n"
"\n"
"    Scope emits data as metrics and/or events.\n"
"    Scope is fully configurable by means of a configuration file and/or \n"
"    environment variables.\n"
"\n"
"    Metrics are emitted in StatsD format over a configurable link. By default\n"
"    metrics are sent over a UDP socket using localhost and port 8125.\n"
"\n"
"    Events are emitted in JSON format over a configurable link. By default\n"
"    events are sent over a TCP socket using localhost and port 9109.\n"
"\n"
"    Scope logs to a configurable destination along with a configurable\n"
"    verbosity level. The default verbosity setting is level 4 and the\n"
"    default destination is the file /tmp/scope.log.\n"
"\n";

static const char scope_help_configuration[] =
"CONFIGURATION:\n"
"    Configuration File:\n"
"       A YAML config file enables control of all available settings.\n"
"       The config file is optional. Environment variables take precedence\n"
"       over settings in a config file.\n"
"\n"
"    Config File Resolution\n"
"        If the SCOPE_CONF_PATH env variable is defined and points to a\n"
"        file that can be opened, it will use this as the config file. If\n"
"        not, it searches for the config file in this priority order using\n"
"        the first one it finds. If this fails too, it will look for\n"
"        scope.yml in the same directory as LD_PRELOAD.\n"
"\n"
"            $SCOPE_HOME/conf/scope.yml\n"
"            $SCOPE_HOME/scope.yml\n"
"            /etc/scope/scope.yml\n"
"            ~/conf/scope.yml\n"
"            ~/scope.yml\n"
"            ./conf/scope.yml\n"
"            ./scope.yml\n"
"\n"
"    Environment Variables:\n"
"    SCOPE_CONF_PATH\n"
"        Directly specify location and name of config file.\n"
"        Used only at start-time.\n"
"    SCOPE_HOME\n"
"        Specify a directory from which conf/scope.yml or ./scope.yml can\n"
"        be found. Used only at start-time only if SCOPE_CONF_PATH does\n"
"        not exist. For more info see Config File Resolution below.\n"
"    SCOPE_METRIC_ENABLE\n"
"        Single flag to make it possible to disable all metric output.\n"
"        true,false  Default is true\n"
"    SCOPE_METRIC_VERBOSITY\n"
"        0-9 are valid values. Default is 4.\n"
"        For more info see Metric Verbosity below.\n"
"    SCOPE_METRIC_DEST\n"
"        Default is udp://localhost:8125\n"
"        Format is one of:\n"
"            file:///tmp/output.log   (file://stdout, file://stderr are\n"
"                                      special allowed values)\n"
"            udp://server:123         (server is servername or address;\n"
"                                      123 is port number or service name)\n"
"    SCOPE_METRIC_FORMAT\n"
"        metricstatsd\n"
"        Default is metricstatsd\n"
"    SCOPE_STATSD_PREFIX\n"
"        Specify a string to be prepended to every scope metric.\n"
"    SCOPE_STATSD_MAXLEN\n"
"        Default is 512\n"
"    SCOPE_SUMMARY_PERIOD\n"
"        Number of seconds between output summarizations. Default is 10\n"
"    SCOPE_EVENT_ENABLE\n"
"        Single flag to make it possible to disable all event output.\n"
"        true,false  Default is true\n"
"    SCOPE_EVENT_DEST\n"
"        same format as SCOPE_METRIC_DEST above.\n"
"        Default is tcp://localhost:9109\n"
"    SCOPE_EVENT_FORMAT\n"
"        ndjson\n"
"        Default is ndjson\n"
"    SCOPE_EVENT_LOGFILE\n"
"        Create events from logs that match SCOPE_EVENT_LOGFILE_NAME.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_LOGFILE_NAME\n"
"        An extended regular expression that describes log file names.\n"
"        Only used if SCOPE_EVENT_LOGFILE is true. Default is .*log.*\n"
"    SCOPE_EVENT_LOGFILE_VALUE\n"
"        An extended regular expression that describes field values.\n"
"        Only used if SCOPE_EVENT_LOGFILE is true. Default is .*\n"
"    SCOPE_EVENT_CONSOLE\n"
"        Create events from stdout, stderr.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_CONSOLE_NAME\n"
"        An extended regular expression that includes stdout, stderr.\n"
"        Only used if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"    SCOPE_EVENT_CONSOLE_VALUE\n"
"        An extended regular expression that describes field values.\n"
"        Only used if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"    SCOPE_EVENT_METRIC\n"
"        Create events from metrics.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_METRIC_NAME\n"
"        An extended regular expression that describes metric names.\n"
"        Only used if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_METRIC_FIELD\n"
"        An extended regular expression that describes field names.\n"
"        Only used if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_METRIC_VALUE\n"
"        An extended regular expression that describes field values.\n"
"        Only used if SCOPE_EVENT_METRIC is true. Default is .*\n"
"    SCOPE_EVENT_HTTP\n"
"        Create events from HTTP headers.\n"
"        true,false  Default is false.\n"
"    SCOPE_EVENT_HTTP_NAME\n"
"        An extended regular expression that describes http names.\n"
"        Only used if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_EVENT_HTTP_FIELD\n"
"        An extended regular expression that describes field names.\n"
"        Only used if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_EVENT_HTTP_VALUE\n"
"        An extended regular expression that describes field values.\n"
"        Only used if SCOPE_EVENT_HTTP is true. Default is .*\n"
"    SCOPE_LOG_LEVEL\n"
"        debug, info, warning, error, none. Default is error.\n"
"    SCOPE_LOG_DEST\n"
"        same format as SCOPE_METRIC_DEST above.\n"
"        Default is file:///tmp/scope.log\n"
"    SCOPE_TAG_\n"
"        Specify a tag to be applied to every metric. Environment variable\n"
"        expansion is available e.g. SCOPE_TAG_user=$USER\n"
"    SCOPE_CMD_DIR\n"
"        Specifies a directory to look for dynamic configuration files.\n"
"        See Dynamic Configuration below.\n"
"        Default is /tmp\n"
"\n"
"    Dynamic Configuration:\n"
"        Dynamic Configuration allows configuration settings to be\n"
"        changed on the fly after process start-time. At every\n"
"        SCOPE_SUMMARY_PERIOD the library looks in SCOPE_CMD_DIR to\n"
"        see if a file scope.<pid> exists. If it exists, it processes\n"
"        every line, looking for environment variable-style commands.\n"
"        (e.g. SCOPE_CMD_DBG_PATH=/tmp/outfile.txt) It changes the\n"
"        configuration to match the new settings, and deletes the\n"
"        scope.<pid> file when it's complete.\n"
"\n";

static const char scope_help_metrics[] =
"METRICS:\n"
"    Metrics can be enabled or disabled with a single config element.\n"
"    Specific types of metrics and specific field content are managed\n"
"    with a verbosity setting.\n"
"\n"
"    Metric Verbosity\n"
"        Controls two different aspects of metric output - \n"
"        Tag Cardinality and Summarization.\n"
"\n"
"        Tag Cardinality\n"
"            0   No expanded statsd tags\n"
"            1   adds 'data', 'unit'\n"
"            2   adds 'class', 'proto'\n"
"            3   adds 'op'\n"
"            4   adds 'pid', 'host', 'proc', 'http_status'\n"
"            5   adds 'domain', 'file'\n"
"            6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'\n"
"            7   adds 'fd', 'args'\n"
"            8   adds 'duration', 'numops', 'req_per_sec', 'req', 'resp', 'protocol'\n"
"\n"
"        Summarization\n"
"            0-4 has full event summarization\n"
"            5   turns off 'error'\n"
"            6   turns off 'filesystem open/close' and 'dns'\n"
"            7   turns off 'filesystem stat' and 'network connect'\n"
"            8   turns off 'filesystem seek'\n"
"            9   turns off 'filesystem read/write' and 'network send/receive'\n"
"\n"
"    The http.status metric is emitted when the http watch type has been\n"
"    enabled as an event. The http.status metric is not controlled with\n"
"    summarization settings.\n"
"\n";

static const char scope_help_events[] =
"EVENTS:\n"
"    All events can be enabled or disabled with a single config element.\n"
"    Unlike metrics, event content is not managed with verbosity settings.\n"
"    Event content is managed by means of regex filters that manage which\n"
"    specific metric is to be included, what field types should be included\n"
"    and what values in fields should be included.\n"
"\n"
"     Events are organized as 4 watch types: \n"
"     1) File Content. Provide a pathname and all data written to the file\n"
"        will be organized in JSON format and emitted over the event channel.\n"
"     2) Console Output. Select stdin &/or stdout and all data written to\n"
"        these endpoints will be formatted in JSON and emitted over the event\n"
"        channel\n"
"     3) Metrics. Content that is much the same as that extracted for StatsD\n"
"        metrics is formatted in JSON and emitted over the event channel.\n"
"        Metrics emitted as events are not summarized and controlled with\n"
"        verbosity settings. Rather regex filters manage which metric,\n"
"        specific fields in a metric and value patterns within a field.\n"
"     4) HTTP Headers. HTTP headers are extracted, formatted in JSON and\n"
"        emitted over the event channel. There are 3 types of events created\n"
"        for HTTP headers; 1) HTTP request events, 2) HTTP response events,\n"
"        and 3) a metric event corresponding to the request and response\n"
"        sequence. A response event includes the corresponding request,\n"
"        status and duration fields. An HTTP metric event provides fields\n"
"        describing bytes received, requests per second, duration and status.\n"
"\n";

static const char scope_help_protocol[] =
"PROTOCOL DETECTION:\n"
"     Any defined network protocol can be detected by Scope. A protocol\n"
"     definition is provided in a separate YAML config file. Protocol specifics\n"
"     are described in one or more regex definitions. PCRE2 regular\n"
"     expressions are supported. Binary and string protocols are detected.\n"
"     Detection events formatted in JSON are emitted over the event channel.\n"
"\n"
"     The event metric watch type should be enabled in order for protocols to\n"
"     be detected.\n"
"\n"
"     The protocol detection config file should be named scope_protocol.yml.\n"
"     The file should be placed in the directory defined by the SCOPE_HOME \n"
"     environment variable. Alternatively, the protocol definition config file\n"
"     can be found in the same search order as that described for config files.\n"
"\n";


typedef struct {
    const char* cmd;
    const char* text;
} help_list_t;

static const help_list_t help_list[] = {
    {"overview", scope_help_overview},
    {"configuration", scope_help_configuration},
    {"metrics", scope_help_metrics},
    {"events", scope_help_events},
    {"protocol", scope_help_protocol},
    {NULL, NULL}
};

// assumes that we're only building for 64 bit...
char const __invoke_dynamic_linker__[] __attribute__ ((section (".interp"))) = "/lib64/ld-linux-x86-64.so.2";
extern char** _dl_argv;

static int
args_are_all_valid(int argc, char **argv)
{
    int i,j;
    for (i=1; i<argc; i++) {
        if (i==1 && !strcmp(argv[i], "help")) continue;
        if (i==1 && !strcmp(argv[i], "all")) continue;

        int found = FALSE;
        for (j=0; help_list[j].cmd; j++) {
            if (!strcmp(argv[i], help_list[j].cmd)) {
                found = TRUE;
                break;
            }
        }
        if (found) continue;

        printf("%s is not a valid argument.  Printing help instead...\n\n", argv[i]);
        return FALSE;
    }
    return TRUE;
}

static void
print_version(char* path)
{
    printf("Scope Version: " SCOPE_VER "\n");
    printf("\n");
    printf("    Usage: LD_PRELOAD=%s <command name>\n", path);
    printf("    For more info: %s help\n", path);
    printf("\n");
}

static void
print_help(char* path)
{
    int i;
    printf( "Welcome to the help for libscope.so!\n\n"
            "    For this message:\n        %1$s help\n"
            "    To print all help:\n        %1$s all\n"
            "    Or to print help by section:\n", path);
    for (i=0; help_list[i].cmd; i++) {
        printf("        %s %s\n", path, help_list[i].cmd);
    }
    printf("\n");
}


void
__scope_main()
{
    // This depends on _dl_argv being NULL terminated.
    int argc = 0;
    while (_dl_argv[argc]) argc++;

    // Get the full path to the library, or provide default if not possible.
    char path[1024] = {0};
    if (readlink("/proc/self/exe", path, sizeof(path)) == -1) {
        snprintf(path, sizeof(path), "libscope.so");
    }

    if (argc < 2) {
        print_version(path);
    } else if (!args_are_all_valid(argc, _dl_argv)) {
        print_help(path);
    } else {
        int i,j;
        if (!strcmp(_dl_argv[1], "help")) {
            print_help(path);
        }
        // print all help sections
        if (!strcmp(_dl_argv[1], "all")) {
            for (i=0; help_list[i].text; i++) {
                printf("    %s\n", help_list[i].text);
            }
        }
        // print matching help sections
        for (i=1; i<argc; i++) {
            for (j=0; help_list[j].cmd; j++) {
                if (!strcmp(_dl_argv[i], help_list[j].cmd)) {
                    printf("%s", help_list[j].text);
                    break;
                }
            }
        }
    }
    exit(0);
}


