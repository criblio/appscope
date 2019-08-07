#include "os.h"

int osGetProcname(char *pname, size_t len)
{
    proc_name(getpid(), pname, len);
    return 0;
}

int osGetNumThreads(pid_t pid)
{
    struct proc_taskinfo task;
    
    if (proc_pidinfo(pid, PROC_PIDTASKINFO, (uint64_t)0,
                     &task, sizeof(struct proc_taskinfo)) == -1) {
        return -1;
    }

    return task.pti_threadnum;
}

int osGetNumFds(pid_t pid)
{
    int bufferSize = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, 0, 0);
    return bufferSize / PROC_PIDLISTFD_SIZE;
}

int osGetNumChildProcs(pid_t pid)
{
    int bufferSize = proc_listchildpids(pid, (void *)NULL, 0);
    return bufferSize / PROC_PIDTASKINFO_SIZE;
}

// For consistency, return the TSC freq in Mhz
int
osInitTSC(struct config_t *cfg)
{
    uint64_t freq;
    size_t size = sizeof(uint64_t);

    if (sysctlbyname("machdep.tsc.frequency", &freq, &size, NULL, 0) != 0) {
        perror("sysctlbyname");
        return -1;
    }

    printf("%s:%d freq: %lld and %lld\n", __FUNCTION__, __LINE__, freq, freq/1000000);
    cfg->freq = freq / 1000000;

    // TODO: Get these values from a CPU register. For now, default.
    // Default to a newer CPU with an invariant TSC
    cfg->tsc_invariant = TRUE;

    // Default to the safer instruction; all CPUs have rdtsc
    cfg->tsc_rdtscp = FALSE;
    return 0;
}
