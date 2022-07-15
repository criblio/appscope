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
#include "openssl/evp.h"

#define MD5_LEN 32

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
    int rv;

    // If we're interrupted, sleep again for whatever time remains
    do {
        rv = scope_nanosleep(&time, &time);
    } while (rv && (scope_errno == EINTR));

    return rv;
}

// Generate a UUID v4 string using a random generator
void
setUUID(char *string)
{
   if (string == NULL) {
       scopeLogError("ERROR: setUUIDv4");
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

    scope_sprintf(string,
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
        scopeLogError("ERROR: setMachineID");
        return;
    }

    char buf[MACHINE_ID_LEN + 1];
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

    scope_sprintf(string, buf, MACHINE_ID_LEN);
}

// Create a Machine ID - an MD5 hash of the hostname
int
createMachineID(char *string)
{
    if (string == NULL) return 1;

    char hostname[MAX_HOSTNAME];
    if (scope_gethostname(hostname, sizeof(hostname)) != 0) {
        scopeLogError("ERROR: gethostname");
        return 1;
    }

    char md5[MD5_LEN + 1];
    generateMD5(hostname, scope_strlen(hostname), md5);
    scope_sprintf(string, "%s", md5);

    return 0;
}

// Generate an MD5 hash from a string
void
generateMD5(const char *data, int len, char *md5_buf)
{
    unsigned char md5_value[EVP_MAX_MD_SIZE];
    unsigned int md5_len;
    EVP_MD_CTX *md5_ctx = EVP_MD_CTX_new();
    const EVP_MD *md5 = EVP_md5();

    EVP_DigestInit_ex(md5_ctx, md5, NULL);
    EVP_DigestUpdate(md5_ctx, data, len);
    EVP_DigestFinal_ex(md5_ctx, md5_value, &md5_len);
    EVP_MD_CTX_free(md5_ctx);

    for (int i = 0; i < md5_len; i++) {
        scope_snprintf(&(md5_buf[i * 2]), 16 * 2, "%02x", md5_value[i]);
    }
}

