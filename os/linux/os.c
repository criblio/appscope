#include "os.h"
#include "../../src/dbg.h"
#include "../../src/scopetypes.h"
#include "../../src/wrap.h"

extern struct interposed_funcs_t g_fn;


static int
sendNL(int sd, ino_t node)
{
    if (g_fn.sendmsg == NULL) return -1;

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
    if (g_fn.recvmsg == NULL) return -1;

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
    if ((buf = malloc(MAX_PROC)) == NULL) {
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
    struct stat sb = {};
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
    struct sigevent sevent;
    timer_t timerid;
    struct itimerspec tspec;

    sigemptyset(&sact.sa_mask);
    sact.sa_handler = handler;
    sact.sa_flags = 0;

    if (!g_fn.sigaction || g_fn.sigaction(SIGUSR2, &sact, NULL) == -1) {
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
