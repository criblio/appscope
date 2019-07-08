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
