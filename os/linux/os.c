#include "os.h"
#include <linux/binfmts.h>

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
osInitTSC(struct rtconfig_t *cfg)
{
    int fd;
    char *entry, *last;
    const char delim[] = ":";
    const char path[] = "/proc/cpuinfo";
    const char freqStr[] = "cpu MHz";
    char *buf;

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
    if (g_fn.__xstat(_STAT_VER, path, &sb) != 0) {
        return -1;
    } else {
        return sb.st_size;
    }
}

int
osGetCmdline(pid_t pid, char *cmd, size_t cmdlen)
{
    int fd, rc;
    char *buf;
    char path[64];

    //DEBUG
    //strncpy(cmd, "none", cmdlen);
    //return -1;

    if (!cmd || !cmdlen) return -1;

    if (!g_fn.open || !g_fn.read || !g_fn.close) {
        strncpy(cmd, "none", cmdlen);
        return -1;
    }

    if ((buf = calloc(1, NCARGS)) == NULL) {
        strncpy(cmd, "none", cmdlen);
        return -1;
    }

    snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);

    if ((fd = g_fn.open(path, O_RDONLY)) == -1) {
        DBG(NULL);
        free(buf);
        strncpy(cmd, "none", cmdlen);
        return -1;
    }

    if ((rc = g_fn.read(fd, buf, NCARGS)) <= 0) {
        DBG(NULL);
        g_fn.close(fd);
        free(buf);
        strncpy(cmd, "none", cmdlen);
        return -1;
    }

    if (rc > cmdlen) {
        memmove(cmd, &buf[rc - cmdlen], cmdlen - 1);
        strcat(cmd, "\0");
    } else {
        memmove(cmd, buf, rc);
    }

    //strncpy(cmd, "none", cmdlen);
    g_fn.close(fd);
    free(buf);
    return 0;
}
