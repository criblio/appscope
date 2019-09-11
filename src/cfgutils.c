#include <errno.h>
#include <pwd.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cfgutils.h"
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
processFormatType(config_t* cfg)
{
    const char* value = getenv("SCOPE_OUT_FORMAT");
    if (!cfg || !value) return;

    if (!strcmp(value, "expandedstatsd")) {
        cfgOutFormatSet(cfg, CFG_EXPANDED_STATSD);
    } else if (!strcmp(value, "newlinedelimited")) {
        cfgOutFormatSet(cfg, CFG_NEWLINE_DELIMITED);
    }
}

static void
processStatsDPrefix(config_t* cfg)
{
    const char* value = getenv("SCOPE_STATSD_PREFIX");
    if (!cfg || !value) return;

    cfgOutStatsDPrefixSet(cfg, value);
}


static void
processStatsDMaxLen(config_t* cfg)
{
    const char* value = getenv("SCOPE_STATSD_MAXLEN");
    if (!cfg || !value) return;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutStatsDMaxLenSet(cfg, x);
}

static void
processSummaryPeriod(config_t* cfg)
{
    const char* value = getenv("SCOPE_OUT_SUM_PERIOD");
    if (!cfg || !value) return;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutPeriodSet(cfg, x);
}

static void
processVerbosity(config_t* cfg)
{
    const char* value = getenv("SCOPE_OUT_VERBOSITY");
    if (!cfg || !value) return;

    errno = 0;
    char* endptr = NULL;
    unsigned long x = strtoul(value, &endptr, 10);
    if (errno || *endptr) return;

    cfgOutVerbositySet(cfg, x);
}

extern char** environ;

static void
processTags(config_t* cfg)
{
    char* e = NULL;
    int i = 0;
    while ((e = environ[i++])) {
        // see if e starts with SCOPE_TAG_
        if (e == strstr(e, "SCOPE_TAG_")) {
            char value_cpy[1024];
            strncpy(value_cpy, e, sizeof(value_cpy));

            char* name = value_cpy + strlen("SCOPE_TAG_");

            // convert the "=" to a null delimiter for the name
            // and move value past the null
            char* value = strchr(name, '=');
            if (value) {
                *value = '\0';
                value++;
                cfgCustomTagAdd(cfg, name, value);
            }
        }
    }
}

static void
processTransport(config_t* cfg, which_transport_t t)
{
    char* value = NULL;
    switch (t) {
        case CFG_OUT:
            value = getenv("SCOPE_OUT_DEST");
            break;
        case CFG_LOG:
            value = getenv("SCOPE_LOG_DEST");
            break;
        default:
            return;
    }
    if (!cfg || !value) return;

    // see if value starts with udp:// or file://
    if (value == strstr(value, "udp://")) {

        // copied to avoid directly modifing the process's env variable
        char value_cpy[1024];
        strncpy(value_cpy, value, sizeof(value_cpy));

        char* host = value_cpy + strlen("udp://");

        // convert the ':' to a null delimiter for the host
        // and move port past the null
        char *port = strrchr(host, ':');
        if (!port) return;  // port is *required*
        *port = '\0';
        port++;

        cfgTransportTypeSet(cfg, t, CFG_UDP);
        cfgTransportHostSet(cfg, t, host);
        cfgTransportPortSet(cfg, t, port);

    } else if (value == strstr(value, "file://")) {
        char* path = value + strlen("file://");
        cfgTransportTypeSet(cfg, t, CFG_FILE);
        cfgTransportPathSet(cfg, t, path);
    }
}

static void
processLevel(config_t* cfg)
{
    const char* value = getenv("SCOPE_LOG_LEVEL");
    if (!cfg || !value) return;

    if (!strcmp(value, "debug")) {
        cfgLogLevelSet(cfg, CFG_LOG_DEBUG);
    } else if (!strcmp(value, "info")) {
        cfgLogLevelSet(cfg, CFG_LOG_INFO);
    } else if (!strcmp(value, "warning")) {
        cfgLogLevelSet(cfg, CFG_LOG_WARN);
    } else if (!strcmp(value, "error")) {
        cfgLogLevelSet(cfg, CFG_LOG_ERROR);
    } else if (!strcmp(value, "none")) {
        cfgLogLevelSet(cfg, CFG_LOG_NONE);
    } else if (!strcmp(value, "trace")) {
        cfgLogLevelSet(cfg, CFG_LOG_TRACE);
    }
}

void
cfgProcessEnvironment(config_t* cfg)
{
    if (!cfg) return;
    processFormatType(cfg);
    processStatsDPrefix(cfg);
    processStatsDMaxLen(cfg);
    processSummaryPeriod(cfg);
    processVerbosity(cfg);
    processTags(cfg);
    processTransport(cfg, CFG_OUT);
    processTransport(cfg, CFG_LOG);
    processLevel(cfg);
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
initOut(config_t* cfg, log_t* log)
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

    // out can have a reference to log for debugging
    //if (log) outLogReferenceSet(out, log);

    return out;
}
