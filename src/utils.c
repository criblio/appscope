#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "scopestdlib.h"
#include "utils.h"
#include "fn.h"
#include "dbg.h"
#include "runtimecfg.h"

#define UNW_LOCAL_ONLY
#include "libunwind.h"

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

#define SYMBOL_NAME_LEN (256)
void
scope_backtrace(void)
{
    unw_cursor_t cursor; unw_context_t uc;
    unw_word_t ip;
    int ret = 0;

    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    
    while (unw_step(&cursor) > 0) {
        char symbol[SYMBOL_NAME_LEN];
        unw_word_t offset;
        unw_get_reg(&cursor, UNW_REG_IP, &ip);

        //Obtain symbol name
        ret = unw_get_proc_name(&cursor, symbol, SYMBOL_NAME_LEN, &offset);
        if ( ret!=0 ){
            scope_printf("ip = %lx, ret = %d\n", (long) ip, ret);
        } else {
            scope_printf("ip = %lx, func_name = %s\n", (long) ip, symbol);
        }
    }  
}
