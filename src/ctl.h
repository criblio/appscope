#ifndef __CTL_H__
#define __CTL_H__

#include "cfg.h"
#include "cJSON.h"
#include "transport.h"
#include "evtformat.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include "pcre2.h"

//////////////////
// This structure is intended to capture received request data from an
// instance of logstream.
// Some of this information is wrapped back in responses we will send.
//////////////////
typedef enum {
    REQ_PARSE_ERR,
    REQ_MALFORMED,
    REQ_UNKNOWN,
    REQ_PARAM_ERR,
    REQ_SET_CFG,
    REQ_GET_CFG,
    REQ_GET_DIAG,
    REQ_BLOCK_PORT,
    REQ_SWITCH,
    REQ_ADD_PROTOCOL,
    REQ_DEL_PROTOCOL,
} cmd_t;

typedef enum {
    NO_ACTION,
    URL_REDIRECT_ON,
    URL_REDIRECT_OFF
} switch_action_t;

typedef struct {
    bool binary;
    char *regex;
    pcre2_code *re;
    unsigned int len;
    unsigned int type;
    char *protname;
} protocol_def_t;

typedef struct {
    cmd_t cmd;                 // our interpretation of what we received
    char *cmd_str;             // command string
    long long id;              // unique request id

    config_t *cfg;             // only used for REQ_SET_CFG
    unsigned short port;       // only used for REQ_BLOCK_PORT
    switch_action_t action;    // only used for REQ_SWITCH
    protocol_def_t *protocol;  // define a protocol to detect

    // other params/structure as we define it,
    // presumably these will be received in body field
} request_t;


//////////////////
// This structure is intended to capture data we'll need to construct
// a message to upload, including:
//    unsolicited status                       (UPLD_INFO), or
//    responses to previous logstream requests (UPLD_RESP), or
//    data we've been configured to send       (UPLD_EVT)
//////////////////
typedef enum {
    UPLD_INFO,
    UPLD_RESP,
    UPLD_EVT,
} upload_type_t;

typedef struct {
    upload_type_t type;
    cJSON* body;
    request_t *req;            // NULL unless this is UPLD_RESP
} upload_t;

// Functions to help with send/receive messaging
request_t *   ctlParseRxMsg(const char*);
void          destroyReq(request_t**);
char *        ctlCreateTxMsg(upload_t*);


//////////////////
// Connection and Transport-Oriented Stuff
//////////////////
typedef struct _ctl_t ctl_t;

// Constructors Destructors
ctl_t * ctlCreate();
void    ctlDestroy(ctl_t **);

// Raw Send (without messaging protocol)
void    ctlSendMsg(ctl_t *, char *);
void    ctlSendJson(ctl_t *, cJSON *);

// Messaging protocol send
int     ctlPostMsg(ctl_t *, cJSON *, upload_type_t, request_t *, bool);
int     ctlSendEvent(ctl_t *, event_t *, uint64_t, proc_id_t *);
int     ctlSendHttp(ctl_t *, event_t *, uint64_t, proc_id_t *);
int     ctlSendLog(ctl_t *, const char *, const void *, size_t, uint64_t, proc_id_t *);
void    ctlFlush(ctl_t *);
int     ctlPostEvent(ctl_t *, char *);

// Connection oriented stuff
int              ctlNeedsConnection(ctl_t *);
int              ctlConnection(ctl_t *);
int              ctlConnect(ctl_t *);
int              ctlClose(ctl_t *);
int              ctlReconnect(ctl_t *);
void             ctlTransportSet(ctl_t *, transport_t *);
void             ctlEvtSet(ctl_t *, evt_fmt_t *);
cfg_transport_t  ctlTransportType(ctl_t *);

// Accessor for performance
bool            ctlEvtSourceEnabled(ctl_t *, watch_t);

// Retreive events
uint64_t   ctlGetEvent(ctl_t *);

bool ctlCbufEmpty(ctl_t *);

// Payloads
int ctlPostPayload(ctl_t *, char *);
uint64_t ctlGetPayload(ctl_t *);

#endif // _CTL_H__
