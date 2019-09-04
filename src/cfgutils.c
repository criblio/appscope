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
