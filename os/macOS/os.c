#include "os.h"
#include "../../src/fn.h"
#include "../../src/scopetypes.h"

int
osGetProcname(char *pname, size_t len)
{
    proc_name(getpid(), pname, len);
    return 0;
}

int
osGetProcMemory(pid_t pid)
{
    struct rusage ruse;

    if (getrusage(RUSAGE_SELF, &ruse) != 0) {
        return (long)-1;
    }

    // return kb
    return ruse.ru_maxrss / 1024;
}

int
osGetProcUidGid(pid_t pid, uid_t *uid, gid_t *gid)
{
    return -1;
}

int
osGetNumThreads(pid_t pid)
{
    struct proc_taskinfo task;
    
    if (proc_pidinfo(pid, PROC_PIDTASKINFO, (uint64_t)0,
                     &task, sizeof(struct proc_taskinfo)) == -1) {
        return -1;
    }

    return task.pti_threadnum;
}

int
osGetNumFds(pid_t pid)
{
    int bufferSize = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, 0, 0);
    return bufferSize / PROC_PIDLISTFD_SIZE;
}

int
osGetNumChildProcs(pid_t pid)
{
    int bufferSize = proc_listchildpids(pid, (void *)NULL, 0);
    return bufferSize / PROC_PIDTASKINFO_SIZE;
}

// For consistency, return the TSC freq in Mhz
int
osInitTSC(platform_time_t *cfg)
{
    uint64_t freq;
    size_t size = sizeof(uint64_t);

    if (sysctlbyname("machdep.tsc.frequency", &freq, &size, NULL, 0) != 0) {
        perror("sysctlbyname");
        return -1;
    }

    cfg->freq = freq / 1000000;

    // TODO: Get these values from a CPU register. For now, default.
    // Default to a newer CPU with an invariant TSC
    cfg->tsc_invariant = TRUE;

    // Default to the safer instruction; all CPUs have rdtsc
    cfg->tsc_rdtscp = FALSE;
    return 0;
}

int
osIsFilePresent(const char *path)
{
    struct stat sb = {};
    if (stat(path, &sb) != 0) {
        return -1;
    } else {
        return sb.st_size;
    }
}

/*
 * TBD:
 * This is not functional.
 * Just a place holder.
 */
int
osGetCmdline(pid_t pid, char **cmd)
{
    if (!cmd) return 0;
    char* buf = *cmd;

    // Free old value, if one exists
    if (buf) free(buf);
    buf = NULL;

    // TBD: placeholder for mac development
    buf = strdup("MACPATH");
    *cmd = buf;

    return (*cmd != NULL);
}

/*
 * TBD:
 * Note that this is incomplete.
 * In Linux we create a timer that delivers a
 * signal on expiry. The signal handler starts
 * the periodic thread. Need the equivalent
 * for OSX.
 */
bool
osThreadInit(void(*handler)(int), unsigned interval)
{
    return TRUE;
}

/*
 * In Linux we use Netlink socket capabilities
 * in order to extract the peer inode for
 * a given socket inode. There is no 
 * Netlink in macOS. We will need to find
 * a way to do something similar.
 */
int
osUnixSockPeer(ino_t lnode)
{
    return -1;
}

/*
 * Placeholder for future
 */
void
osInitJavaAgent(void)
{
    return;
}

int
osGetPageProt(uint64_t addr)
{
    return -1;
}

bool
osTimerStop(void)
{
    return TRUE;
}

char *
osGetFileMode(mode_t perm)
{
    return NULL;
}

bool
osGetCgroup(pid_t pid, char *cgroup, size_t cglen)
{
    return FALSE;
}

int
osNeedsConnect(int fd)
{
    return 0;
}

char *
osGetUserName(unsigned uid)
{
    return NULL;
}

char *
osGetGroupName(unsigned gid)
{
    return NULL;
}

long long
osGetProcCPU(void) {
    return -1;
}
