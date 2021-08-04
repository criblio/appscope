#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "fn.h"
#include "dbg.h"
#include "runtimecfg.h"

rtconfig g_cfg = {0};

unsigned int
strToVal(enum_map_t map[], const char *str)
{
    enum_map_t *m;
    for (m=map; m->str; m++) {
        if (!strcmp(str, m->str)) return m->val;
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
       (strncmp(estr, val, strlen(estr)) == 0)) {
        return TRUE;
    }
    return FALSE;
}

/*
 * Handling the case where there can be 2 instances
 * of setenv; 1) libc and 2) application.
 */
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
    int returnval = snprintf(val, sizeof(val), "%d", pid);
    if (returnval >= sizeof(val) || returnval == -1) {
        DBG("returnval = %d", returnval);
        return;
    }

    if (fullSetenv(SCOPE_PID_ENV, val, 1) == -1) {
        char dbmsg[PATH_MAX];
        snprintf(dbmsg, sizeof(dbmsg), "setPidEnv: %s:%s", SCOPE_PID_ENV, val);
        scopeLog(dbmsg, -1, CFG_LOG_DEBUG);
    }
}

#ifdef __MACOS__
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
        if (!stat(cmd, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = strdup(cmd);
        }
        goto out;
    }

    // a relative path was specified for cmd.
    if (strchr(cmd, '/')) {
        char *cur_dir = get_current_dir_name();
        if (!cur_dir) goto out;

        char *path = NULL;
        if (asprintf(&path, "%s/%s", cur_dir, cmd) > 0) {
            // If we can resolve it, use it
            if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
                ret_val = path;
            } else {
                free(path);
            }
        }
        free(cur_dir);
        goto out;
    }

    // try the current dir
    char *path = NULL;
    if (asprintf(&path, "./%s", cmd) > 0) {
        if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = path;
            goto out;
        } else {
            free(path);
        }
    }

    // try to resolve the cmd from PATH env variable
    char *path_env_ptr = getenv("PATH");
    if (!path_env_ptr) goto out;
    path_env = strdup(path_env_ptr); // create a copy for strtok below
    if (!path_env) goto out;

    char *saveptr = NULL;
    char *strtok_path = strtok_r(path_env, ":", &saveptr);
    if (!strtok_path) goto out;

    do {
        char *path = NULL;
        if (asprintf(&path, "%s/%s", strtok_path, cmd) < 0) {
            break;
        }
        if ((stat(path, &buf) == -1) ||    // path doesn't exist
            (!S_ISREG(buf.st_mode)) ||     // path isn't a file
            ((buf.st_mode & 0111) == 0)) { // path isn't executable

            free(path);
            continue;
        }

        // We found the cmd, and it's an executable file
        ret_val = path;
        break;

    } while ((strtok_path = strtok_r(NULL, ":", &saveptr)));

out:
    if (path_env) free(path_env);
    return ret_val;
}
#endif //__MACOS__


int
startsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    return (strncmp(string, substring, strlen(substring)) == 0);
}

int
endsWith(const char *string, const char *substring)
{
    if (!string || !substring) return FALSE;
    int stringlen = strlen(string);
    int sublen = strlen(substring);
    return (sublen <= stringlen) &&
       ((strncmp(&string[stringlen-sublen], substring, sublen)) == 0);
}

int
sigSafeNanosleep(const struct timespec *req)
{
    if (!g_fn.nanosleep) {
        DBG(NULL);
        return -1;
    }

    struct timespec time = *req;
    int rv;

    // If we're interrupted, sleep again for whatever time remains
    do {
        rv = g_fn.nanosleep(&time, &time);
    } while (rv && (errno == EINTR));

    return rv;
}

#ifndef __FUNCHOOK__
int
func_found_in_executable(const char *symbol, const char *exe)
{
    return 0;
}

int
run_bash_mem_fix(void)
{
    return 0;
}

#endif
