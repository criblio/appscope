#ifndef __COM_H__
#define __COM_H__

#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/types.h>
#include <ctype.h>
#include <inttypes.h>

#include "ctl.h"
#include "cfgutils.h"
#include "scopetypes.h"

#define TRUE 1
#define FALSE 0
#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128

#ifndef bool
typedef unsigned int bool;
#endif

typedef struct rtconfig_t {
    int numNinfo;
    int numFSInfo;
    bool tsc_invariant;
    bool tsc_rdtscp;
    uint64_t freq;
    const char *cmddir;
    char hostname[MAX_HOSTNAME];
    char procname[MAX_PROCNAME];
    char cmd[DEFAULT_CMD_SIZE];
    pid_t pid;
} rtconfig;

// Post a message to the command buffer
int cmdPostEvtMsg(ctl_t *, cJSON *);
int cmdPostInfoMsg(ctl_t *, cJSON *);

// Send a message directly on the command channel
int cmdSendEvtMsg(ctl_t *, cJSON *);
int cmdSendInfoStr(ctl_t *, const char *);
int cmdSendInfoMsg(ctl_t *, cJSON *);
int cmdSendResponse(ctl_t *, request_t *);

// Process a command received from stream over the command channel
request_t *cmdParse(ctl_t *, char *);

// Create a json object for process start
cJSON *msgStart(rtconfig *, config_t *);

// Create a json object for an event metric
cJSON *msgEvtMetric(evt_t *, event_t *, uint64_t, rtconfig *);

// Create a json object for a log event
cJSON *msgEvtLog(evt_t *, const char *path, const void *, size_t, uint64_t, rtconfig *);

#endif // __COM_H__
