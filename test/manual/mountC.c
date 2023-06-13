 /*
 * Sample code for adding mounts in existing containers
 * gcc -g -o mc test/manual/mountC.c 
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <limits.h>
#include <string.h>
#include <sys/mount.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>

#define FILTER_FILE "scope_filter"
#define SOCK_FILE "appscope.sock"

// copied from the loader
#define FALSE 0
#define TRUE 1

// copied from the loader
typedef unsigned int bool;
typedef unsigned long libdirfile_t;

// copied from the loader
const char *
libdirGetPath(libdirfile_t objFileType)
{
    return strdup("/tmp/appscope/");
}

// copied from the loader
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

bool
makeDirs(const char *rootdir, pid_t pid, const char *filterdir, const char *file, mode_t mode)
{
    char path[PATH_MAX] = {0};

    if (rootdir) {
        snprintf(path, sizeof(path), "%s/proc/%d/root/%s/",
                 rootdir, pid, filterdir);
    } else {
        snprintf(path, sizeof(path), "/proc/%d/root/%s/", pid, filterdir);
    }

    if ((mkdir(path, mode) == -1) && (errno != EEXIST)) {
        perror("mkdir");
        return FALSE;
    }

    if (file) {
        mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

        strcat(path, "/");
        strcat(path, file);
        if ((creat(path, mode) == -1) && (errno != EEXIST)) {
            perror("creat");
            return FALSE;
        }
    }
    
    return TRUE;
}

char *
getMountPath(pid_t pid)
{
    bool candidate = FALSE;
    pid_t nsPid;
    size_t len;
    char *buf = NULL, *mount = NULL;
    FILE *fstream;
    char path[PATH_MAX];

    // only pids that are in a container
    if (nsInfoGetPidNs(pid, &nsPid) == FALSE) return NULL;

    snprintf(path, sizeof(path), "/proc/%d/mounts", pid);
    fstream = fopen(path, "r");
    if (fstream == NULL) {
        perror("fopen");
        return NULL;
    }

    while (getline(&buf, &len, fstream) != -1) {
        // if a docker overlay mount and not already mounted; appscope
        if ((strstr(buf, "overlay")) &&
            (strstr(buf, "docker"))) {
            char *start, *end;
            if (((start = strstr(buf, "workdir="))) &&
                ((end = strstr(buf, "/work")))) {
                start += strlen("workdir=");
                *end = '\0';
                strcat(start, "/merged");
                mount = strdup(start);
                candidate = TRUE;
            }
        }

        // no longer a candidate as we've already mounted this proc
        if (strstr(buf, "appscope")) candidate = FALSE;

        free(buf);
        buf = NULL;
        len = 0;
    }

    if (buf) free(buf);
    fclose(fstream);

    if (candidate == TRUE) {
        return mount;
    } else {
        free(mount);
    }
    return NULL;
}

bool
doDir(pid_t pid, char *rootdir, char *filterdir, char *file, char *fstype)
{
    char *mountdir;
    libdirfile_t objfile;

    mountdir = (char *)libdirGetPath(objfile);
    if (!mountdir) {
        return FALSE;
    }

    // make the filter file in the merged dir
    if (makeDirs(rootdir, pid, (const char *)mountdir, file, 0666) == FALSE) {
        fprintf(stderr, "Warn: mkdir of %s from %s:%d\n", filterdir, __FUNCTION__, __LINE__);
        free(mountdir);
        return FALSE;
    }

    strcat(mountdir, file);
    strcat(filterdir, mountdir);

    // mount the filter file into the container
    if (mount(mountdir, filterdir, fstype, MS_BIND, NULL) != 0) {
        perror("mount");
        fprintf(stderr, "WARN: mount %s on %s from %s:%d\n", mountdir, filterdir, __FUNCTION__, __LINE__);
        free(mountdir);
        return FALSE;
    }

    free(mountdir);
    return TRUE;
}

bool
mountCDirs(pid_t pid, const char *rootdir, char *target, char *fstype)
{
    pid_t nsPid;
    char *filterdir = NULL;
    size_t targetlen = strlen(target);

    if ((filterdir = malloc(targetlen + 128)) == NULL) return FALSE;

    strcpy(filterdir, target);
    if (doDir(pid, (char *)rootdir, filterdir, FILTER_FILE, fstype) == FALSE) {
        fprintf(stderr, "Can't mount a dir in the container");
        free(filterdir);
        return FALSE;
    }

    strcpy(filterdir, target);
    if (doDir(pid, (char *)rootdir, filterdir, SOCK_FILE, fstype) == FALSE) {
        fprintf(stderr, "Can't mount a dir in the container");
        free(filterdir);
        return FALSE;
    }

    free(filterdir);
    return TRUE;
}

int main(int argc, char **argv)
{
    printf("Adding mounts\n");

    DIR *dirp;
    struct dirent *entry;
    char *mpath = NULL, *rootdir = NULL;

    dirp = opendir("/proc");
    if (dirp == NULL) {
        perror("opendir");
        return FALSE;
    }

    // Iterate all procs
    while ((entry = readdir(dirp)) != NULL) {
        // procs/tasks are a dir
        if (entry->d_type == DT_DIR) {
            pid_t pid = atoi(entry->d_name);
            if (pid > 0) {
                // if pid is in a supported container, get the requisite mount path
                if ((mpath = getMountPath(pid)) != NULL) {
                    printf("%s:%d pid %d mount %s\n", __FUNCTION__, __LINE__, pid, mpath);
                    mountCDirs(pid, rootdir, mpath, NULL);
                    free(mpath);
                }
            }
        }
    }

    return 0;
}
