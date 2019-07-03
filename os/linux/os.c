#include "os.h"

int osGetProcname(char *pname, int len)
{
    strncpy(pname, program_invocation_short_name, len);
    return 0;
}

int osGetNumThreads(pid_t pid)
{
    int fd, i;
    char *entry;
    const char delim[] = " ";
    char buf[1024];

    snprintf(buf, sizeof(buf), "/proc/%d/stat", pid);
    if ((fd = open(buf, O_RDONLY)) == -1) {
        return -1;
    }

     if (read(fd, buf, sizeof(buf)) == -1) {
        return -1;
    }

     entry = strtok(buf, delim);
     for (i = 1; i < 20; i++) {
        entry = strtok(NULL, delim);
    }

     return atoi((const char *)entry);
}
