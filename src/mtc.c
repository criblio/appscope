#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "dbg.h"
#include "mtc.h"
#include "circbuf.h"

struct _mtc_t
{
    unsigned enable;
    transport_t* transport;
    mtc_fmt_t* format;
};

mtc_t *
mtcCreate()
{
    mtc_t *mtc = calloc(1, sizeof(mtc_t));
    if (!mtc) {
        DBG(NULL);
        return NULL;
    }
    mtc->enable = DEFAULT_MTC_ENABLE;

    return mtc;
}

void
mtcDestroy(mtc_t **mtc)
{
    if (!mtc || !*mtc) return;
    mtc_t *mtcb = *mtc;
    transportDestroy(&mtcb->transport);
    mtcFormatDestroy(&mtcb->format);
    free(mtcb);
    *mtc = NULL;
}

unsigned
mtcEnabled(mtc_t *mtc)
{
    if (!mtc) return DEFAULT_MTC_ENABLE;
    return mtc->enable;
}

int
mtcSend(mtc_t *mtc, const char *msg)
{
    if (!mtc || !msg) return -1;

    return transportSend(mtc->transport, msg, strlen(msg));
}

int
mtcSendMetric(mtc_t *mtc, event_t *evt)
{
    if (!mtc || !evt) return -1;

    char *msg = mtcFormatEventForOutput(mtc->format, evt, NULL);
    int rv = mtcSend(mtc, msg);
    if (msg) free(msg);
    return rv;
}

void
mtcFlush(mtc_t *mtc)
{
    if (!mtc) return;

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

int
mtcReconnect(mtc_t *mtc)
{
    if (!mtc) return 0;
    return transportReconnect(mtc->transport);
}

void
mtcEnabledSet(mtc_t *mtc, unsigned val)
{
    if (!mtc || val > 1) return;
    mtc->enable = val;
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

