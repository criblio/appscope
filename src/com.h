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
#include "runtimecfg.h"
#include "pcre2.h"

#define ROUND_DOWN(num, unit) ((num) & ~((unit) - 1))
#define ROUND_UP(num, unit) (((num) + (unit) - 1) & ~((unit) - 1))
#define PCRE_STACK_SIZE (32 * 1024)

extern bool g_need_stack_expand;

// Post a message from report to the command buffer
int cmdSendEvent(ctl_t *, event_t *, uint64_t, proc_id_t *);
int cmdSendMetric(mtc_t *, event_t* );
int cmdSendHttp(ctl_t *, event_t *, uint64_t, proc_id_t *);
int cmdPostEvent(ctl_t *, char *);
int cmdSendBin(ctl_t *, char *, size_t);

// Post a message to the command buffer
int cmdPostEvtMsg(ctl_t *, cJSON *);
int cmdPostInfoMsg(ctl_t *, cJSON *);

// Send a message directly on the command channel
int cmdSendEvtMsg(ctl_t *, cJSON *);
int cmdSendInfoStr(ctl_t *, const char *);
int cmdSendInfoMsg(ctl_t *, cJSON *);
int cmdSendResponse(ctl_t *, request_t *, cJSON *);

// Process a command received from stream over the command channel
request_t *cmdParse(const char *);

// Create a json object for process start
cJSON *msgStart(proc_id_t *, config_t *);

// Create a json object describing the current configuration
cJSON *jsonConfigurationObject(config_t *);

// Retreive messages
uint64_t msgEventGet(ctl_t *);

// wrappers
int pcre2_match_wrapper(pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE,
                        uint32_t, pcre2_match_data *, pcre2_match_context *);
int regexec_wrapper(const regex_t *, const char *, size_t, regmatch_t *, int);

#endif // __COM_H__
