#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>

#include "nsinfo.h"
#include "scopestdlib.h"

/*
 * Return effective uid of current user inside the namespace for specified pid.
 */
uid_t
nsInfoTranslateUid(pid_t hostPid) {
    uid_t eUid = geteuid();
    char uidPath[PATH_MAX] = {0};
    char buffer[4096] = {0};
    FILE *fd;

    if (snprintf(uidPath, sizeof(uidPath), "/proc/%d/uid_map", hostPid) < 0) {
        perror("snprintf uid_map failed");
        return eUid;
    }
    if ((fd = fopen(uidPath, "r")) == NULL) {
        perror("fopen(/proc/<PID>/uid_map) failed");
        return eUid;
    }

    while (fgets(buffer, sizeof(buffer), fd)) {
        const char delimiters[] = " \t";
        char *last;

        char *entry = strtok_r(buffer, delimiters, &last);
        uid_t uidInsideNs = atoi(entry);

        entry = strtok_r(NULL, delimiters, &last);
        uid_t uidOutsideNs = atoi(entry);
        if ((uidInsideNs == 0) && (uidInsideNs != uidOutsideNs)) {
            eUid = uidOutsideNs;
            break;
        }
    }
    fclose(fd);

    return eUid;
}

/*
 * Return effective gid of current user inside the namespace for specified pid.
 */
gid_t
nsInfoTranslateGid(pid_t hostPid) {
    gid_t gid = getegid();
    char gidPath[PATH_MAX] = {0};
    char buffer[4096] = {0};
    FILE *fd;

    if (snprintf(gidPath, sizeof(gidPath), "/proc/%d/gid_map", hostPid) < 0) {
        perror("snprintf gid_map failed");
        return gid;
    }
    if ((fd = fopen(gidPath, "r")) == NULL) {
        perror("fopen(/proc/<PID>/gid_map) failed");
        return gid;
    }

    while (fgets(buffer, sizeof(buffer), fd)) {
        const char delimiters[] = " \t";
        char *last;

        char *entry = strtok_r(buffer, delimiters, &last);
        gid_t gidInsideNs = atoi(entry);

        entry = strtok_r(NULL, delimiters, &last);
        gid_t gidOutsideNs = atoi(entry);
        if ((gidInsideNs == 0) && (gidInsideNs != gidOutsideNs)) {
            gid = gidOutsideNs;
            break;
        }
    }
    fclose(fd);

    return gid;
}

/*
 * Check if the specified process exists in the same mnt namespace as current process.
 *
 * Returns TRUE if specified process exists in the same mnt namespace as current process, FALSE otherwise.
 */
bool
nsInfoIsPidInSameMntNs(pid_t pid) {
    char path[PATH_MAX] = {0};
    struct stat selfStat = {0};
    struct stat targetStat = {0};

    if (stat("/proc/self/ns/mnt", &selfStat) != 0) {
        perror("stat(/proc/self/ns/mnt) failed");
        return TRUE;
    }

    if (snprintf(path, sizeof(path), "/proc/%d/ns/mnt", pid) < 0) {
        perror("snprintf(/proc/<pid>/ns/mnt) failed");
        return TRUE;
    }

    if (stat(path, &targetStat) != 0) {
        perror("stat(/proc/<pid>/ns/mnt) failed");
        return TRUE;
    }

    /*
    * If two processes are in the same mount namespace, then the device IDs and
    * inode numbers of their /proc/[pid]/ns/mnt symbolic links will be the same.
    */
    return (selfStat.st_dev == targetStat.st_dev) && (selfStat.st_ino == targetStat.st_ino);
}

/*
 * Check if the specified process contains nested PID namespaces. 
 *
 * Returns TRUE if specific process contains nested PID namespaces with
 * PID from the last PID namespace in lastNsPid argument, FALSE otherwise.
 */
bool
nsInfoGetPidNs(pid_t pid, pid_t *lastNsPid) {
    char path[PATH_MAX] = {0};
    char buffer[4096];
    bool status = FALSE;
    int tempNsPid = 0;
    int nsDepth = 0;

    if (snprintf(path, sizeof(path), "/proc/%d/status", pid) < 0) {
        return FALSE;
    }

    FILE *fstream = fopen(path, "r");

    if (fstream == NULL) {
        return FALSE;
    }

    while (fgets(buffer, sizeof(buffer), fstream)) {
        if (strstr(buffer, "NSpid:")) {
            const char delimiters[] = ": \t";
            char *entry, *last;

            entry = strtok_r(buffer, delimiters, &last);
            // Skip NsPid string
            entry = strtok_r(NULL, delimiters, &last);
            // Iterate over NsPids values
            while (entry != NULL) {
                tempNsPid = atoi(entry);
                entry = strtok_r(NULL, delimiters, &last);
                nsDepth++;
            }
            break;
        }
    }

    if (nsDepth > 1) {
        status = TRUE;
        *lastNsPid = tempNsPid;
    }

    fclose(fstream);

    return status;
}
