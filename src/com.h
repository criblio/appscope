#ifndef __COM_H__
#define __COM_H__

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

#define PCRE_STACK_SIZE (32 * 1024)

extern bool g_need_stack_expand;
extern unsigned g_sendprocessstart;
extern bool g_exitdone;

// Post a message from report to the command buffer
int cmdSendEvent(ctl_t *, event_t *, uint64_t, proc_id_t *);
int cmdSendMetric(mtc_t *, event_t* );
int cmdSendHttp(ctl_t *, event_t *, uint64_t, proc_id_t *);
int cmdPostEvent(ctl_t *, char *);

// Post a message to the command buffer
int cmdPostInfoMsg(ctl_t *, cJSON *);

// Send a message directly on the command channel
int cmdSendInfoStr(ctl_t *, const char *);
int cmdSendResponse(ctl_t *, request_t *, cJSON *);

// Process a command received from stream over the command channel
request_t *cmdParse(const char *);

// Create a json object for process start
cJSON *msgStart(proc_id_t *, config_t *, which_transport_t);
char *msgAddNewLine(char *);
void msgLogConfig(config_t *);
void reportProcessStart(ctl_t *, bool, which_transport_t);
void sendProcessStartMetric();

// Create a json object describing the current configuration
cJSON *jsonConfigurationObject(config_t *);

// Retreive messages
uint64_t msgEventGet(ctl_t *);

// wrappers
int pcre2_match_wrapper(pcre2_code *, PCRE2_SPTR, PCRE2_SIZE, PCRE2_SIZE,
                        uint32_t, pcre2_match_data *, pcre2_match_context *);
int regexec_wrapper(const regex_t *, const char *, size_t, regmatch_t *, int);

bool cmdCbufEmpty(ctl_t *);

// payloads
int cmdSendPayload(ctl_t *, char *, size_t);
int cmdPostPayload(ctl_t *, char *);
uint64_t msgPayloadGet(ctl_t *);

#endif // __COM_H__
