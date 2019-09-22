#include <errno.h>
#include <pwd.h>
#include <regex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cfgutils.h"
#include "dbg.h"
#include "format.h"
#include "scopetypes.h"

char* 
cfgPath(const char* cfgname)
{
    // in priority order:
    //   1) $SCOPE_HOME/conf/scope.cfg
    //   2) $SCOPE_HOME/scope.cfg
    //   3) /etc/scope/scope.cfg
    //   4) ~/conf/scope.cfg
    //   5) ~/scope.cfg
    //   6) ./conf/scope.cfg
    //   7) ./scope.cfg

    char path[1024]; // Somewhat arbitrary choice for MAX_PATH

    const char* homedir = getenv("HOME");
    const char* scope_home = getenv("SCOPE_HOME");
    if (scope_home &&
       (snprintf(path, sizeof(path), "%s/conf/%s", scope_home, cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (scope_home &&
       (snprintf(path, sizeof(path), "%s/%s", scope_home, cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "/etc/scope/%s", cfgname) > 0 ) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (homedir &&
       (snprintf(path, sizeof(path), "%s/conf/%s", homedir, cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if (homedir &&
       (snprintf(path, sizeof(path), "%s/%s", homedir, cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "./conf/%s", cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }
    if ((snprintf(path, sizeof(path), "./%s", cfgname) > 0) &&
        !access(path, R_OK)) {
        return realpath(path, NULL);
    }

    return NULL;
}

static void
processCustomTag(config_t* cfg, const char* e, const char* value)
{
    char name_buf[1024];
    strncpy(name_buf, e, sizeof(name_buf));

    char* name = name_buf + strlen("SCOPE_TAG_");

    // convert the "=" to a null delimiter for the name
    char* end = strchr(name, '=');
    if (end) {
        *end = '\0';
        cfgCustomTagAddFromStr(cfg, name, value);
    }

}

static regex_t*
envRegex()
{
    static regex_t* regex = NULL;

    if (regex) return regex;

    if (!(regex = calloc(1, sizeof(regex_t)))) {
        DBG(NULL);
        return regex;
    }

    if (regcomp(regex, "\\$[a-zA-Z0-9_]+", REG_EXTENDED)) {
        // regcomp failed.
        DBG(NULL);
        free(regex);
        regex = NULL;
    }
    return regex;
}

static char*
doEnvVariableSubstitution(char* value)
{
    if (!value) return NULL;

    regex_t* re = envRegex();
    regmatch_t match = {0};

    int out_size = strlen(value) + 1;
    char* outval = calloc(1, out_size);
    if (!outval) {
        DBG("%s", value);
        return NULL;
    }

    char* outptr = outval;  // "tail" pointer where text can be appended
    char* inptr = value;    // "current" pointer as we scan through value

    while (re && !regexec(re, inptr, 1, &match, 0)) {

        int match_size = match.rm_eo - match.rm_so;

        // if the character before the match is '\', don't do substitution
        char* escape_indicator = &inptr[match.rm_so - 1];
        int escaped = (escape_indicator >= value) && (*escape_indicator == '\\');

        if (escaped) {
            // copy the part before the match, except the escape char '\'
            outptr = stpncpy(outptr, inptr, match.rm_so - 1);
            // copy the matching env variable name
            outptr = stpncpy(outptr, &inptr[match.rm_so], match_size);
            // move to the next part of the input value
            inptr = &inptr[match.rm_eo];
            continue;
        }

        // lookup the part that matched to see if we can substitute it
        char env_name[match_size + 1];
        strncpy(env_name, &inptr[match.rm_so], match_size);
        env_name[match_size] = '\0';
        char* env_value = getenv(&env_name[1]); // offset of 1 skips the $

        // Grow outval buffer any time env_value is bigger than env_name
        int size_growth = (!env_value) ? 0 : strlen(env_value) - match_size;
        if (size_growth > 0) {
            char* new_outval = realloc (outval, out_size + size_growth);
            if (new_outval) {
                out_size += size_growth;
                outptr = new_outval + (outptr - outval);
                outval = new_outval;
            } else {
                DBG("%s", value);
                free(outval);
                return NULL;
            }
        }

        // copy the part before the match
        outptr = stpncpy(outptr, inptr, match.rm_so);
        // either copy in the env value or the variable that wasn't found
        outptr = stpcpy(outptr, (env_value) ? env_value : env_name);
        // move to the next part of the input value
        inptr = &inptr[match.rm_eo];
    }

    // copy whatever is left
    strcpy(outptr, inptr);

    return outval;
}

static void
processCmdDebug(const char* path)
{
    if (!path || !path[0]) return;

    FILE* f;
    if (!(f = fopen(path, "a"))) return;
    dbgDumpAll(f);
    fclose(f);
}


static int
startsWith(const char* string, const char* substring)
{
    return (strncmp(string, substring, strlen(substring)) == 0);
}

//
// An example of this format: SCOPE_STATSD_MAXLEN=1024
//
static void
processEnvStyleInput(config_t* cfg, const char* env_line)
{

    if (!cfg || !env_line) return;

    char* env_ptr, *value;
    if (!(env_ptr = strchr(env_line, '='))) return;
    if (!(value = doEnvVariableSubstitution(&env_ptr[1]))) return;

    if (startsWith(env_line, "SCOPE_OUT_FORMAT")) {
        cfgOutFormatSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_PREFIX")) {
        cfgOutStatsDPrefixSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_STATSD_MAXLEN")) {
        cfgOutStatsDMaxLenSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_OUT_SUM_PERIOD")) {
        cfgOutPeriodSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_CMD_PATH")) {
        cfgOutCmdPathSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_OUT_VERBOSITY")) {
        cfgOutVerbositySetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_LOG_LEVEL")) {
        cfgLogLevelSetFromStr(cfg, value);
    } else if (startsWith(env_line, "SCOPE_OUT_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_OUT, value);
    } else if (startsWith(env_line, "SCOPE_LOG_DEST")) {
        cfgTransportSetFromStr(cfg, CFG_LOG, value);
    } else if (startsWith(env_line, "SCOPE_TAG_")) {
        processCustomTag(cfg, env_line, value);
    } else if (startsWith(env_line, "SCOPE_CMD_DEBUG")) {
        processCmdDebug(value);
    }

    free(value);
}


extern char** environ;

void
cfgProcessEnvironment(config_t* cfg)
{
    if (!cfg) return;
    char* e = NULL;
    int i = 0;
    while ((e = environ[i++])) {
        // Everything we care about starts with a capital 'S'.  Skip 
        // everything else for performance.
        if (e[0] != 'S') continue;

        // Some things should only be processed as commands, not as
        // environment variables.  Skip them here.
        if (startsWith(e, "SCOPE_CMD_DEBUG")) continue;

        // Process everything else.
        processEnvStyleInput(cfg, e);
    }
}

void
cfgProcessCommands(config_t* cfg, FILE* file)
{
    if (!cfg || !file) return;
    char* e = NULL;
    size_t n = 0;

    while (getline(&e, &n, file) != -1) {
        e[strcspn(e, "\r\n")] = '\0'; //overwrite first \r or \n with null
        processEnvStyleInput(cfg, e);
        e[0] = '\0';
    }

    if (e) free(e);
}

static transport_t*
initTransport(config_t* cfg, which_transport_t t)
{
    transport_t* transport = NULL;

    switch (cfgTransportType(cfg, t)) {
        case CFG_SYSLOG:
            transport = transportCreateSyslog();
            break;
        case CFG_FILE:
            transport = transportCreateFile(cfgTransportPath(cfg, t));
            break;
        case CFG_UNIX:
            transport = transportCreateUnix(cfgTransportPath(cfg, t));
            break;
        case CFG_UDP:
            transport = transportCreateUdp(cfgTransportHost(cfg, t), cfgTransportPort(cfg, t));
            break;
        case CFG_SHM:
            transport = transportCreateShm();
            break;
        default:
            DBG("%d", cfgTransportType(cfg, t));
    }
    return transport;
}

static format_t*
initFormat(config_t* cfg)
{
    format_t* fmt = fmtCreate(cfgOutFormat(cfg));
    if (!fmt) return NULL;

    fmtStatsDPrefixSet(fmt, cfgOutStatsDPrefix(cfg));
    fmtStatsDMaxLenSet(fmt, cfgOutStatsDMaxLen(cfg));
    fmtOutVerbositySet(fmt, cfgOutVerbosity(cfg));
    fmtCustomTagsSet(fmt, cfgCustomTags(cfg));
    return fmt;
}

log_t*
initLog(config_t* cfg)
{
    log_t* log = logCreate();
    if (!log) return log;
    transport_t* t = initTransport(cfg, CFG_LOG);
    if (!t) {
        logDestroy(&log);
        return log;
    }
    logTransportSet(log, t);
    logLevelSet(log, cfgLogLevel(cfg));
    return log;
}

out_t*
initOut(config_t* cfg)
{
    out_t* out = outCreate();
    if (!out) return out;

    transport_t* t = initTransport(cfg, CFG_OUT);
    if (!t) {
        outDestroy(&out);
        return out;
    }
    outTransportSet(out, t);

    format_t* f = initFormat(cfg);
    if (!f) {
        outDestroy(&out);
        return out;
    }
    outFormatSet(out, f);

    return out;
}
