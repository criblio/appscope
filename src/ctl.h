#ifndef __CTL_H__
#define __CTL_H__

#include "cfg.h"
#include "cJSON.h"
#include "transport.h"

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
} cmd_t;

typedef struct {
    cmd_t cmd;                 // our interpretation of what we received
    char* cmd_str;             // command string
    long long id;              // unique request id

    config_t* cfg;             // only used for REQ_SET_CFG

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
    request_t* req;           // NULL unless this is UPLD_RESP
    long long status;         // HTTP-like status
    const char* message;      // optional error message
} upload_t;

// Functions to help with send/receive messaging
request_t*          ctlParseRxMsg(const char*);
void                destroyReq(request_t**);
char*               ctlCreateTxMsg(upload_t*);


//////////////////
// Connection and Transport-Oriented Stuff
//////////////////
typedef struct _ctl_t ctl_t;

// Constructors Destructors
ctl_t *             ctlCreate();
void                ctlDestroy(ctl_t **);

// Raw Send
void                ctlSendMsg(ctl_t *, char *);
void                ctlFlush(ctl_t *);

// Connection oriented stuff
int                 ctlNeedsConnection(ctl_t *);
int                 ctlConnection(ctl_t *);
int                 ctlConnect(ctl_t *);
int                 ctlClose(ctl_t *);
void                ctlTransportSet(ctl_t *, transport_t *);

#endif // _CTL_H__
