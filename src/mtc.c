#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "dbg.h"
#include "mtc.h"
#include "circbuf.h"

struct _mtc_t
{
    transport_t* transport;
    mtc_fmt_t* format;
    cbuf_handle_t evbuf;
    int metric_disabled;
};

static void
sendBufferedMessages(mtc_t *mtc)
{
    if (!mtc) return;

    uint64_t data;
    while (cbufGet(mtc->evbuf, &data) == 0) {
        if (data) {
            char *msg = (char*) data;
            transportSend(mtc->transport, msg);
            free(msg);
        }
    }
}

mtc_t *
mtcCreate()
{
    mtc_t *mtc = calloc(1, sizeof(mtc_t));
    if (!mtc) {
        DBG(NULL);
        return NULL;
    }

    // TBD.  This is a quick and dirty way to disable metric output.
    // Added 31-Jan-2020 for a demo.
    mtc->metric_disabled = (getenv("SCOPE_METRIC_DISABLE") != NULL);

    mtc->evbuf = cbufInit(DEFAULT_METRIC_CBUF_SIZE);
    if (!mtc->evbuf) {
        DBG(NULL);
        return NULL;
    }

    return mtc;
}

void
mtcDestroy(mtc_t **mtc)
{
    if (!mtc || !*mtc) return;
    mtc_t *mtcb = *mtc;
    transportDestroy(&mtcb->transport);
    mtcFormatDestroy(&mtcb->format);
    cbufFree((*mtc)->evbuf);
    free(mtcb);
    *mtc = NULL;
}

int
mtcSend(mtc_t *mtc, const char *msg)
{
    if (!mtc || !msg) return -1;

    if (cbufPut(mtc->evbuf, (uint64_t)msg) == -1) {
        // Full; drop and ignore
        DBG(NULL);
        free((char *)msg);
        return -1;
    }

    return 0;
}

int
mtcSendMetric(mtc_t *mtc, event_t *evt)
{
    if (!mtc || !evt) return -1;

    if (mtc->metric_disabled) return 0;

    char *msg = mtcFormatStatsDString(mtc->format, evt, NULL);
    int rv = mtcSend(mtc, msg);
    return rv;
}

void
mtcFlush(mtc_t *mtc)
{
    if (!mtc) return;
    sendBufferedMessages(mtc);
    transportFlush(mtc->transport);
}

int
mtcNeedsConnection(mtc_t *mtc)
{
    if (!mtc) return 0;
    return transportNeedsConnection(mtc->transport);
}

int
mtcConnect(mtc_t *mtc)
{
    if (!mtc) return 0;
    return transportConnect(mtc->transport);
}

int
mtcDisconnect(mtc_t *mtc)
{
    if (!mtc) return 0;
    return transportDisconnect(mtc->transport);
}

void
mtcTransportSet(mtc_t *mtc, transport_t *transport)
{
    if (!mtc) return;

    // Don't leak if mtcTransportSet is called repeatedly
    transportDestroy(&mtc->transport);
    mtc->transport = transport;
}

void
mtcFormatSet(mtc_t *mtc, mtc_fmt_t *format)
{
    if (!mtc) return;

    // Don't leak if mtcFormatSet is called repeatedly
    mtcFormatDestroy(&mtc->format);
    mtc->format = format;
}

