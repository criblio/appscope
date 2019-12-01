#include <stdlib.h>
#include "circbuf.h"
#include "ctl.h"
#include "dbg.h"

struct _ctl_t
{
    transport_t* transport;
    cbuf_handle_t evbuf;
};

ctl_t*
ctlCreate()
{
    ctl_t* ctl = calloc(1, sizeof(ctl_t));
    if (!ctl) {
        DBG(NULL);
        return NULL;
    }

    ctl->evbuf = cbufInit(DEFAULT_CBUF_SIZE);

    return ctl;
}

void
ctlDestroy(ctl_t** ctl)
{
    if (!ctl || !*ctl) return;

    ctlFlush(*ctl);
    cbufFree((*ctl)->evbuf);

    transportDestroy(&(*ctl)->transport);

    free(*ctl);
    *ctl = NULL;
}

void
ctlSendMsg(ctl_t* ctl, char * msg)
{
    if (!msg) return;
    if (!ctl) {
        free(msg);
        return;
    }

    if (cbufPut(ctl->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free(msg);
    }
}

static void
sendBufferedMessages(ctl_t* ctl)
{
    if (!ctl) return;

    uint64_t data;
    while (cbufGet(ctl->evbuf, &data) == 0) {
        if (data) {
            char *msg = (char*) data;
            transportSend(ctl->transport, msg);
            free(msg);
        }
    }
}

void
ctlFlush(ctl_t* ctl)
{
    if (!ctl) return;
    sendBufferedMessages(ctl);
    transportFlush(ctl->transport);
}

int
ctlNeedsConnection(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportNeedsConnection(ctl->transport);
}

int
ctlConnection(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportConnection(ctl->transport);
}

int
ctlConnect(ctl_t *ctl)
{
    if (!ctl) return 0;
    return transportConnect(ctl->transport);
}

void
ctlTransportSet(ctl_t* ctl, transport_t* transport)
{
    if (!ctl) return;

    // Don't leak if ctlTransportSet is called repeatedly
    transportDestroy(&ctl->transport);
    ctl->transport = transport;
}


