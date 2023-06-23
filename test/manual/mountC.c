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
#include <getopt.h>

// copied from the loader
#define FALSE 0
#define TRUE 1

// copied from the loader
typedef unsigned int bool;

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

/*
 * The mkdir system call does not support the creation of intermediate dirs.
 * This will walk the path and create all dirs one at a time.
 */
static bool
makeDirs(pid_t pid, const char *rootdir, const char *targetdir, mode_t mode)
{
    char path[PATH_MAX] = {0};

    if (rootdir) {
        snprintf(path, sizeof(path), "%s/proc/%d/root/%s/",
                 rootdir, pid, targetdir);
    } else {
        snprintf(path, sizeof(path), "/proc/%d/root/%s/", pid, targetdir);
    }

    char *slash;
    char *dup_path = strdup(path);
    if (!dup_path) {
        return FALSE;
    }

    char *curr = strstr(dup_path, "root/");
    if (!curr) return FALSE;
    curr += strlen("root/");

    while ((slash = strchr(curr, '/'))) {
        *slash = '\0';
        if (mkdir(dup_path, mode) == -1) {
            if (errno != EEXIST) {
                free(dup_path);
                return FALSE;
            }
        }

        *slash = '/';
        curr = slash + 1;
    }

    if (mkdir(dup_path, mode) == -1) {
        if (errno != EEXIST) {
            free(dup_path);
            return FALSE;
        }
    }

    free(dup_path);

    return TRUE;
}

static bool
doDir(pid_t pid, char *rootdir, char *overlaydir, size_t olen, char *dest, size_t dlen, char *fstype)
{
    char mountdir[olen + dlen + 2];

    if (!overlaydir || !dest) return FALSE;

    strcpy(mountdir, overlaydir);
    strcat(mountdir, dest);

    // make the overlay file in the merged dir
    if (makeDirs(pid, rootdir, (const char *)dest, 0666) == FALSE) {
        fprintf(stderr, "Warn: mkdir of %s from %s:%d\n", overlaydir, __FUNCTION__, __LINE__);
        return FALSE;
    }

    // mount the overlay file into the container
    if (mount(dest, mountdir, fstype, MS_BIND, NULL) != 0) {
        perror("mount");
        fprintf(stderr, "WARN: mount %s on %s from %s:%d\n", dest, overlaydir, __FUNCTION__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

static bool
mountCDirs(pid_t pid, char *target, const char *rootdir,
           const char *rulesdir, const char *sockdir, char *fstype)
{
    if (!target || !rulesdir) return FALSE;
    
    pid_t nsPid;
    char *overlaydir = NULL;
    size_t tlen = strlen(target);
    size_t flen = strlen(rulesdir);

    if ((overlaydir = malloc(tlen + 1)) == NULL) return FALSE;
    strcpy(overlaydir, target);

    if (doDir(pid, (char *)rootdir, overlaydir, tlen, (char *)rulesdir, flen, fstype) == FALSE) {
        fprintf(stderr, "Can't mount %s in the container\n", rulesdir);
        free(overlaydir);
        return FALSE;
    }

    if (sockdir) {
        size_t slen = strlen(sockdir);

        if (doDir(pid, (char *)rootdir, overlaydir, tlen, (char *)sockdir, slen, fstype) == FALSE) {
            fprintf(stderr, "Can't mount %s in the container\n", sockdir);
            free(overlaydir);
            return FALSE;
        }
    }

    free(overlaydir);
    return TRUE;
}

static char *
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

static struct option opts[] = {
	{ "rootdir",    required_argument, 0, 'r' },
	{ "rulesdir",  required_argument, 0, 'f' },
	{ "sockdir",    required_argument, 0, 's' },
	{ 0, 0, 0, 0 }
};

int main(int argc, char **argv)
{
    printf("Adding mounts\n");

    int index;
    DIR *dirp;
    struct dirent *entry;
    char *mpath = NULL, *rootdir = NULL, *rulesdir = NULL, *sockdir = NULL;

    for (;;) {
		index = 0;
        int opt = getopt_long(argc, argv, "+:f:r:s:", opts, &index); // "+:rf:s"
		if (opt == -1) break;

		switch (opt) {
		case 'r':
			rootdir = optarg;
			break;
		case 'f':
			rulesdir = optarg;
			break;
		case 's':
			sockdir = optarg;
			break;
		case ':': // Handle options missing their arg value
			switch (optopt) {
			default:
				fprintf(stderr, "error: missing required value for -%c option\n", optopt);
				exit(EXIT_FAILURE);
			}
			break;
		default:
            fprintf(stderr, "error: need at least a rules dir.\n");
            exit(EXIT_FAILURE);
		}
	}

    if (!rulesdir) {
        fprintf(stderr, "error: a rules dir is required.\n");
        exit(EXIT_FAILURE);
    }

    printf("%s:%d %s %s %s\n", __FUNCTION__, __LINE__, rootdir, rulesdir, sockdir);

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
                    mountCDirs(pid, mpath, rootdir, rulesdir, sockdir, NULL);
                    free(mpath);
                }
            }
        }
    }

    return 0;
}
