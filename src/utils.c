#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "scopestdlib.h"
#include "utils.h"
#include "fn.h"
#include "dbg.h"
#include "runtimecfg.h"
#include "plattime.h"

#define MAC_ADDR_LEN 17
#define ZERO_MACHINE_ID "00000000000000000000000000000000"

// TODO verify if we need to recognize all theste states here
typedef enum {
    INT_MKDIR_STATUS_CREATED         = 0,    // Path was created
    INT_MKDIR_STATUS_EXISTS          = 1,    // Path already points to existing directory
    INT_MKDIR_STATUS_ERR_PERM_ISSUE  = 2,    // Error: Path already points to existing directory but user can not create file there
    INT_MKDIR_STATUS_ERR_NOT_ABS_DIR = 3,    // Error: Path does not points to a directory
    INT_MKDIR_STATUS_ERR_OTHER       = 4,    // Error: Other
} internal_mkdir_status_t;

static int createMachineID(char *string);
static int getMacAddr(char *string);

rtconfig g_cfg = {0};

unsigned int
strToVal(enum_map_t map[], const char *str)
{
    enum_map_t *m;
    for (m=map; m->str; m++) {
        if (!scope_strcmp(str, m->str)) return m->val;
    }
    return -1;
}

const char *
valToStr(enum_map_t map[], unsigned int val)
{
    enum_map_t *m;
    for (m=map; m->str; m++) {
        if (val == m->val) return m->str;
    }
    return NULL;
}

int
checkEnv(char *env, char *val)
{
    char *estr;
    if (((estr = getenv(env)) != NULL) &&
       (scope_strncmp(estr, val, scope_strlen(estr)) == 0)) {
        return TRUE;
    }
    return FALSE;
}


int
fullSetenv(const char *key, const char *val, int overwrite)
{
    int lrc = 0, arc = 0;

    if (!g_fn.setenv || (g_fn.setenv(key, val, overwrite) == -1)) {
        DBG("g_fn.setenv=%p, g_fn.app_setenv=%p key=%s, val=%s",
            g_fn.setenv, g_fn.app_setenv, key, val);
        lrc = -1;
    }

    if (g_fn.app_setenv && (g_fn.app_setenv != g_fn.setenv)) {
        arc = g_fn.app_setenv(key, val, overwrite);
    }

    if ((lrc == -1) || (arc == -1)) return -1;
    return 0;
}

void
setPidEnv(int pid)
{
    char val[32];
    int returnval = scope_snprintf(val, sizeof(val), "%d", pid);
    if (returnval >= sizeof(val) || returnval == -1) {
        DBG("returnval = %d", returnval);
        return;
    }

    if (fullSetenv(SCOPE_PID_ENV, val, 1) == -1) {
        scopeLog(CFG_LOG_DEBUG, "setPidEnv: %s:%s", SCOPE_PID_ENV, val);
    }
}

#ifdef __APPLE__
char *
getpath(const char *cmd)
{
    return NULL;
}

#else

// This tests to see if cmd can be resolved to an executable file.
// If so it will return a malloc'd buffer containing an absolute path,
// otherwise it will return NULL.
char *
getpath(const char *cmd)
{
    char *path_env = NULL;
    char *ret_val = NULL;
    struct stat buf;

    if (!cmd) goto out;

    // an absolute path was specified for cmd.
    if (cmd[0] == '/') {
        //  If we can resolve it, use it.
        if (!scope_stat(cmd, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = scope_strdup(cmd);
        }
        goto out;
    }

    // a relative path was specified for cmd.
    if (scope_strchr(cmd, '/')) {
        char *cur_dir = scope_get_current_dir_name();
        if (!cur_dir) goto out;

        char *path = NULL;
        if (scope_asprintf(&path, "%s/%s", cur_dir, cmd) > 0) {
            // If we can resolve it, use it
            if (!scope_stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
                ret_val = path;
            } else {
                scope_free(path);
            }
        }
        scope_free(cur_dir);
        goto out;
    }

    // try the current dir
    char *path = NULL;
    if (scope_asprintf(&path, "./%s", cmd) > 0) {
        if (!scope_stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = path;
            goto out;
        } else {
            scope_free(path);
        }
    }

    // try to resolve the cmd from PATH env variable
    char *path_env_ptr = getenv("PATH");
    if (!path_env_ptr) goto out;
    path_env = scope_strdup(path_env_ptr); // create a copy for strtok below
    if (!path_env) goto out;

    char *saveptr = NULL;
    char *strtok_path = scope_strtok_r(path_env, ":", &saveptr);
    if (!strtok_path) goto out;

    do {
        char *path = NULL;
        if (scope_asprintf(&path, "%s/%s", strtok_path, cmd) < 0) {
            break;
        }
        if ((scope_stat(path, &buf) == -1) ||    // path doesn't exist
            (!S_ISREG(buf.st_mode)) ||     // path isn't a file
            ((buf.st_mode & 0111) == 0)) { // path isn't executable

            scope_free(path);
            continue;
        }

        // We found the cmd, and it's an executable file
        ret_val = path;
        break;

    } while ((strtok_path = scope_strtok_r(NULL, ":", &saveptr)));

out:
    if (path_env) scope_free(path_env);
    return ret_val;
}
#endif //__APPLE__


int
startsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    return (scope_strncmp(string, substring, scope_strlen(substring)) == 0);
}

int
endsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    int stringlen = scope_strlen(string);
    int sublen = scope_strlen(substring);
    return (sublen <= stringlen) &&
       ((scope_strncmp(&string[stringlen-sublen], substring, sublen)) == 0);
}

int
sigSafeNanosleep(const struct timespec *req)
{
    struct timespec time = *req;
    uint64_t ctime = getTime();
    uint64_t elapsed;
    int rv;

    // If we're interrupted, sleep again for whatever time remains
    do {
        rv = scope_nanosleep(&time, &time);
        elapsed = getDuration(ctime);
    } while (rv && (elapsed < req->tv_nsec));

    return rv;
}

// Generate a UUID v4 string using a random generator
void
setUUID(char *string)
{
   if (string == NULL) {
       scopeLogError("ERROR: setUUIDv4: Null string");
       return;
   }

    unsigned char key[16];
    static bool seeded = FALSE;

    if (!seeded) {
        scope_srand((unsigned int)scope_time(NULL));
        seeded = TRUE;
    }

    for (int i = 0; i < 16; i++) {
        key[i] = (unsigned char)scope_rand() % 255;
    }

    key[6] = 0x40 | (key[6] & 0xf); // Set version to 4
    key[8] = 0x80 | (key[8] & 0x3f); // Set variant to 8

    scope_snprintf(string, UUID_LEN + 1,
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7], 
        key[8], key[9], key[10], key[11],
        key[12], key[13], key[14], key[15]);
}

// Get the Machine ID, or if not available, create one
void
setMachineID(char *string)
{
    if (string == NULL) {
        scopeLogError("ERROR: setMachineID: Null string");
        return;
    }
    scope_strncpy(string, ZERO_MACHINE_ID, MACHINE_ID_LEN + 1);

    char buf[MACHINE_ID_LEN + 1] = {0};
    FILE *fp;

    // Try to get a machine id from /etc
    if ((fp = scope_fopen("/etc/machine-id", "r")) != NULL) {
        if (scope_fgets(buf, sizeof(buf), fp) == NULL) {
            scopeLogInfo("INFO: setMachineID: Could not read Machine ID from file /etc/machine-id");
        }
        scope_fclose(fp);
    }

    if (scope_strlen(buf) != MACHINE_ID_LEN) {
        scopeLogInfo("INFO: setMachineID: Machine ID not found or unexpected length. Creating one.");
        if (createMachineID(buf)) {
            scopeLogError("ERROR: setMachineID: Error creating Machine ID");
            return;
        }
    }

    scope_strncpy(string, buf, MACHINE_ID_LEN + 1);
}

// Create a Machine ID from a mac address
static int
createMachineID(char *string)
{
    if (string == NULL) return 1;

    char mac_addr[MAC_ADDR_LEN];
    if (getMacAddr(mac_addr)) {
        scopeLogError("ERROR: createMachineID: getMacAddr");
        return 1;
    }

    scope_snprintf(string, MACHINE_ID_LEN + 1, 
        "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
        mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3],
        mac_addr[4], mac_addr[5], mac_addr[6], mac_addr[7], 
        mac_addr[8], mac_addr[9], mac_addr[10], mac_addr[11],
        mac_addr[12], mac_addr[13], mac_addr[14], mac_addr[15]);
    return 0;
}

// Get the machine's physical MAC address
static int
getMacAddr(char *string)
{
    DIR *d;
    struct dirent *dir;
    struct stat buf;
    char mac_buf[MAC_ADDR_LEN];
    char dir_path[256];
    char link_path[256];
    char addr_path[256];
    bool found = FALSE;
    
    d = scope_opendir("/sys/class/net/");
    if (!d) return 1;

    // Check if interface eth exists
    // Otherwise find an interface that does not contain "virtual" in the soft link
    while ((dir = scope_readdir(d)) != NULL) {
        scope_sprintf(dir_path, "/sys/class/net/%s", dir->d_name);

        if (scope_strstr(dir->d_name, "eth") != 0) {
            found = TRUE;
            break;
        }
    
        if (scope_lstat(dir_path, &buf) != 0) {
            break;
        }
        if (S_ISLNK(buf.st_mode)) {
            (void)scope_readlink(dir_path, link_path, sizeof(link_path));
            if (scope_strstr(link_path, "virtual") == NULL) {
                found = TRUE;
                break;
            }
        }
    }
    scope_closedir(d);

    if (!found) {
        scopeLogError("Error: getMacAddr: No physical interface found");
        return 1;
    }

    scope_sprintf(addr_path, "%s/address", dir_path);

    FILE *fp;
    if ((fp = scope_fopen(addr_path, "r")) == NULL) {
        scopeLogError("Error: getMacAddr: No address file found");
        return 1;
    }
    if (scope_fgets(mac_buf, sizeof(mac_buf), fp) == NULL) {
        scopeLogError("Error: getMacAddr: No address found in file");
        scope_fclose(fp);
        return 1;
    }
    scope_fclose(fp);

    scope_strncpy(string, mac_buf, MAC_ADDR_LEN + 1);
    return 0;
}

/*
 * Convert unsigned long to a non-null terminated string (signal safe API)
 */
void
sigSafeUtoa(unsigned long val, char *buf, int base, int *len) {
    int i = 0;

    if (val == 0) {
        buf[0] = '0';
        *len = 1;
        return;
    }

    // Process each digit
    while (val != 0) {
        int rem = val % base;
        buf[i++] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';
        val = val/base;
    }
    *len = i;

    // Reverse the output
   for (i = 0; i < *len/2; i++) {  
        char temp = buf[i];  
        buf[i] = buf[*len - i - 1];  
        buf[*len - i - 1] = temp;  
    }  

    return;
}

/*
 * Verify if following absolute path points to directory (signal safe API)
 */
static internal_mkdir_status_t
sigSafeCheckIfDirExists(const char *absDirPath, uid_t uid, gid_t gid) {
    struct stat st = {0};
    if (!scope_stat(absDirPath, &st)) {
        if (S_ISDIR(st.st_mode)) {      
            // Check for file creation abilities in directory
            if (((st.st_uid == uid) && (st.st_mode & S_IWUSR)) ||
                ((st.st_gid == gid) && (st.st_mode & S_IWGRP)) ||
                (st.st_mode & S_IWOTH)) {
                return INT_MKDIR_STATUS_EXISTS;
            }
            return INT_MKDIR_STATUS_ERR_PERM_ISSUE;
        }
        return INT_MKDIR_STATUS_ERR_NOT_ABS_DIR;
    }
    return INT_MKDIR_STATUS_ERR_OTHER;
}

/*
 * Create directory recursive (signal safe API)
 */
bool
sigSafeMkdirRecursive(const char *dirPath) {
    char tempPath[PATH_MAX] = {0};
    int mkdirRes = -1;
    /* Operate only on absolute path */
    if (dirPath == NULL || *dirPath != '/') {
        return FALSE;
    }

    uid_t euid = geteuid();
    gid_t egid = getegid();

    internal_mkdir_status_t res = sigSafeCheckIfDirExists(dirPath, euid, egid);
    /* exit if path exists */
    if (res != INT_MKDIR_STATUS_ERR_OTHER) {
        return TRUE;
    }
    scope_strcpy(tempPath, dirPath);

    /* traverse the full path */
    for (char *p = tempPath + 1; *p; p++) {
        if (*p == '/') {
            /* Temporarily truncate */
            *p = '\0';
            scope_errno = 0;

            struct stat st = {0};
            if (scope_stat(tempPath, &st)) {
                mkdirRes = scope_mkdir(tempPath, 0755);
                if (!mkdirRes) {
                    /* We ensure that we setup correct mode regarding umask settings */
                    if (scope_chmod(tempPath, 0755)) {
                        return FALSE;
                    }
                } else {
                    /* scope_mkdir fails */
                   return FALSE;
                }
            }

            *p = '/';
        }
    }
    struct stat st = {0};
    if (scope_stat(tempPath, &st)) {
        /* if last element was not created in the loop above */
        mkdirRes = scope_mkdir(tempPath, 0755);
        if (mkdirRes) {
            return FALSE;
        }
    }

    /* We ensure that we setup correct mode regarding umask settings */
    return (scope_chmod(tempPath, 0755) == 0) ? TRUE : FALSE;
}

/*
 * Writes to specified file descriptor (signal safe API)
 */
ssize_t 
sigSafeWrite(int fd, const void *buf, size_t count) {
    return scope_write(fd, buf ,count);
}


/*
 * Converts the specific value using base for conversion and
 * drites to specified file descriptor (signal safe API)
 */
ssize_t 
sigSafeWriteNumber(int fd, long val, int base) {
    char buf[32] = {0};
    int msgLen = 0;
    sigSafeUtoa(val, buf, base, &msgLen);
    return scope_write(fd, buf ,msgLen);
}
