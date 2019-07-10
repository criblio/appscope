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

    // Get the size of the file with stat, malloc buf then free
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

int osGetNumFds(pid_t pid)
{
    int nfile = 0;
    DIR * dirp;
    struct dirent * entry;
    char buf[1024];

    snprintf(buf, sizeof(buf), "/proc/%d/fd", pid);
    if ((dirp = opendir(buf)) == NULL) {
        return -1;
    }
    
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type != DT_DIR) {
            nfile++;
        }
    }

    closedir(dirp);
    return nfile;
}

int osGetNumChildProcs(pid_t pid)
{
    int nchild = 0;
    DIR * dirp;
    struct dirent * entry;
    char buf[1024];

    snprintf(buf, sizeof(buf), "/proc/%d/task", pid);
    if ((dirp = opendir(buf)) == NULL) {
        return -1;
    }
    
    while ((entry = readdir(dirp)) != NULL) {
            nchild++;
    }

    closedir(dirp);
    return nchild - 3; // Not including the parent proc itself and ., ..
}
