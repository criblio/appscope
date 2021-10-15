#define _GNU_SOURCE
#include <sys/param.h>
#include <time.h>
#include "os.h"
#include "../../src/dbg.h"
#include "../../src/fn.h"
#include "../../src/scopetypes.h"

// want to put this list in an obvious place
//static char thread_delay_list[] = "chrome:nacl_helper";
static timer_t g_timerid = 0;

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
        scopeLog(LOG_LEVEL, "fd:%d ERROR:sendNL:sendmsg", sd);
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
        scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:recvmsg", sd);
        return (ino_t)-1;
    }

    const struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;

    if (!NLMSG_OK(nlhdr, rc)) {
        scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:!NLMSG_OK", sd);
        return (ino_t)-1;
    }

    for (; NLMSG_OK(nlhdr, rc); nlhdr = NLMSG_NEXT(nlhdr, rc)) {
        if (nlhdr->nlmsg_type == NLMSG_DONE) {
            scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:no message", sd);
            return (ino_t)-1;
        }

        if (nlhdr->nlmsg_type == NLMSG_ERROR) {
            const struct nlmsgerr *err = NLMSG_DATA(nlhdr);

            if (nlhdr->nlmsg_len < NLMSG_LENGTH(sizeof(*err))) {
                scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:message error", sd);
            } else {
                scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:message errno %d", sd, -err->error);
            }

            return (ino_t)-1;
        }

        if (nlhdr->nlmsg_type != SOCK_DIAG_BY_FAMILY) {
            scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:unexpected nlmsg_type", sd);
            return (ino_t)-1;
        }

        if ((diag = NLMSG_DATA(nlhdr)) != NULL) {
            if (nlhdr->nlmsg_len < NLMSG_LENGTH(sizeof(*diag))) {
                scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:short response", sd);
                return (ino_t)-1;
            }

            if (diag->udiag_family != AF_UNIX) {
                scopeLog(LOG_LEVEL, "fd:%d ERROR:getNL:unexpected family", sd);
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

static uint64_t
getProcVal(char *srcbuf, const char *tag)
{
    char *entry, *last, *buf;
    uint64_t val = -1;
    const char delim[] = ":";

    if (!srcbuf) return -1;
    buf = strdup(srcbuf);

    entry = strtok_r(buf, delim, &last);
    while (1) {
        if ((entry = strtok_r(NULL, delim, &last)) == NULL) {
            break;
        }

        if (strcasestr((const char *)entry, tag) != NULL) {
            // The next token should be what we want
            if ((entry = strtok_r(NULL, delim, &last)) != NULL) {
                if ((val = (uint64_t)strtoll(entry, NULL, 0)) == (long long)0) {
                    val = (uint64_t)-1;
                }
                break;
            }
        }
    }

    if (buf) free(buf);
    return val;
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

int
osGetExePath(char **path)
{
    if (!path) return -1;
    char *buf = *path;

    if (!(buf = calloc(1, PATH_MAX))) {
        scopeLogError("ERROR:calloc in osGetExePath");
        return -1;
    }

    if (readlink("/proc/self/exe", buf, PATH_MAX - 1) == -1) {
        scopeLogError("osGetPath: can't get path to self exe");
        free(buf);
        return -1;
    }

    *path = buf;
    return 0;
}

int
osGetProcname(char *pname, int len)
{
    if (program_invocation_short_name != NULL) {
        strncpy(pname, program_invocation_short_name, len);
    } else {
        char *ppath = NULL;

        if (osGetExePath(&ppath) != -1) {
            strncpy(pname, basename(ppath), len);
            if (ppath) free(ppath);
        } else {
            return -1;
        }
    }
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
osInitTimer(platform_time_t *cfg)
{
    int fd;
    uint64_t val;
    char *buf;
    const char path[] = "/proc/cpuinfo";

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

    if ((cfg->freq = getProcVal(buf, "cpu MHz")) == -1) {
        cfg->freq = -1;
    }

    if (((val = getProcVal(buf, "CPU architecture")) != -1) &&
        (val >= 8)) {
        cfg->gptimer_avail = TRUE;
    } else {
        cfg->gptimer_avail = FALSE;
    }

#ifdef __aarch64__
    /*
     * This uses the General Purpose Timer definition in an aarch64 instance.
     * The frequency is the lower 32 bits of the CNTFRQ_EL0 register and
     * is defined as HZ. The configured freq is defined in Mhz.
     */
    if (cfg->gptimer_avail == TRUE) {
        uint64_t freq;

        __asm__ volatile (
            "mrs x1, CNTFRQ_EL0 \n"
            "mov %0, x1  \n"
            : "=r" (freq)                // output
            :                            // inputs
            :                            // clobbered register
            );

        freq &= 0x0000000ffffffff;
        freq /= 1000000;
        cfg->freq = freq;
    }
#elif defined(__x86_64__)
    if ((cfg->tsc_invariant == TRUE) && (cfg->freq != -1)) {
        cfg->gptimer_avail = TRUE;
    } else {
        cfg->gptimer_avail = FALSE;
    }
#else
#error No architecture defined
#endif

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
    char *buf = *cmd;
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
osTimerStop(void)
{

    if (g_timerid) {
        timer_delete(g_timerid);
        g_timerid = 0;
        return TRUE;
    }

    return FALSE;
}

bool
osThreadInit(void(*handler)(int), unsigned interval)
{
    struct sigaction sact;
    struct sigevent sevent = {0};
    struct itimerspec tspec;
    sigemptyset(&sact.sa_mask);
    sact.sa_handler = handler;
    sact.sa_flags = SA_RESTART;

    if (!g_fn.sigaction) return FALSE;

    if (g_fn.sigaction(SIGUSR2, &sact, NULL) == -1) {
        DBG("errno %d", errno);
        return FALSE;
    }

    sevent.sigev_notify = SIGEV_SIGNAL;
    sevent.sigev_signo = SIGUSR2;

    if (timer_create(CLOCK_MONOTONIC, &sevent, &g_timerid) == -1) {
        DBG("errno %d", errno);
        return FALSE;
    }

    tspec.it_interval.tv_sec = 0;
    tspec.it_interval.tv_nsec = 0;
    tspec.it_value.tv_sec = interval;
    tspec.it_value.tv_nsec = 0;
    if (timer_settime(g_timerid, 0, &tspec, NULL) == -1) {
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

        scopeLog(CFG_LOG_TRACE, "addr 0x%lux addr1 0x%lux addr2 0x%lux\n", addr, addr1, addr2);

        if ((addr >= addr1) && (addr < addr2)) {
            char *perms = end + 1;
            scopeLog(CFG_LOG_DEBUG, "matched 0x%lx to 0x%lx-0x%lx\n\t%c%c%c", addr, addr1, addr2, perms[0], perms[1], perms[2]);
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

int
osNeedsConnect(int fd)
{
    int rc, timeout;
    struct pollfd fds;

    if (!g_fn.poll) return 0;

    timeout = 0;
    memset(&fds, 0x0, sizeof(fds));
    fds.events = POLLRDHUP | POLLOUT;

    fds.fd = fd;

    rc = g_fn.poll(&fds, 1, timeout);

    if ((rc != 0) && (((fds.revents & POLLRDHUP) != 0) || ((fds.revents & POLLERR) != 0))) return 1;
    return 0;
}

/*
 * Example from /proc/<pid>/cgroup:
   2:freezer:/
   11:pids:/user.slice/user-1000.slice/user@1000.service
   10:memory:/user.slice/user-1000.slice/user@1000.service
   9:cpuset:/
   8:devices:/user.slice
   7:net_cls,net_prio:/
   6:blkio:/user.slice
   5:perf_event:/
   4:hugetlb:/
   3:cpu,cpuacct:/user.slice
   2:rdma:/
   1:name=systemd:/user.slice/user-1000.slice/user@1000.service/gnome-launched-emacs.desktop-21457.scope
   0::/user.slice/user-1000.slice/user@1000.service/gnome-launched-emacs.desktop-21457.scope
 */
bool
osGetCgroup(pid_t pid, char *cgroup, size_t cglen)
{
    size_t len = 0;
    char *buf = NULL;
    char path[PATH_MAX];

    if (!g_fn.fopen || !g_fn.getline || !g_fn.fclose || (cglen <= 0)) {
        return FALSE;
    }

    if (snprintf(path, sizeof(path), "/proc/%d/cgroup", pid) < 0) return FALSE;

    FILE *fstream = g_fn.fopen(path, "r");
    if (fstream == NULL) return FALSE;

    while (g_fn.getline(&buf, &len, fstream) != -1) {
        if (buf && strstr(buf, "0::")) {
            strncpy(cgroup, buf, cglen);
            char *nonl = strchr(cgroup, '\n');
            if (nonl) *nonl = '\0';

            free(buf);
            g_fn.fclose(fstream);
            return TRUE;
        }

        if (buf) free(buf);
        buf = NULL;
        len = 0;
    }

    g_fn.fclose(fstream);
    return FALSE;
}

char *
osGetFileMode(mode_t perm)
{
    char *mode = malloc(MODE_STR);
    if (!mode) return NULL;

    mode[0] = (perm & S_IRUSR) ? 'r' : '-';
    mode[1] = (perm & S_IWUSR) ? 'w' : '-';
    mode[2] = (perm & S_IXUSR) ? 'x' : '-';
    mode[3] = (perm & S_IRGRP) ? 'r' : '-';
    mode[4] = (perm & S_IWGRP) ? 'w' : '-';
    mode[5] = (perm & S_IXGRP) ? 'x' : '-';
    mode[6] = (perm & S_IROTH) ? 'r' : '-';
    mode[7] = (perm & S_IWOTH) ? 'w' : '-';
    mode[8] = (perm & S_IXOTH) ? 'x' : '-';
    mode[9] = '\0';
    return mode;
}

