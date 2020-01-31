#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "dbg.h"
#include "out.h"

struct _out_t
{
    transport_t* transport;
    format_t* format;
    int metric_disabled;
};

out_t*
outCreate()
{
    out_t* out = calloc(1, sizeof(out_t));
    if (!out) {
        DBG(NULL);
        return NULL;
    }

    // TBD.  This is a quick and dirty way to disable metric output.
    // Added 31-Jan-2020 for a demo.
    out->metric_disabled = (getenv("SCOPE_METRIC_DISABLE") != NULL);

    return out;
}

void
outDestroy(out_t** out)
{
    if (!out || !*out) return;
    out_t* o = *out;
    transportDestroy(&o->transport);
    fmtDestroy(&o->format);
    free(o);
    *out = NULL;
}

int
outSend(out_t* out, const char* msg)
{
    if (!out || !msg) return -1;

    return transportSend(out->transport, msg);
}

int
outSendEvent(out_t* out, event_t* e)
{
    if (!out || !e) return -1;

    if (out->metric_disabled) return -1;

    char* msg = fmtStatsDString(out->format, e, NULL);
    int rv = outSend(out, msg);
    if (msg) free(msg);
    return rv;
}

void
outFlush(out_t* out)
{
    if (!out) return;
    transportFlush(out->transport);
}

void
outTransportSet(out_t* out, transport_t* transport)
{
    if (!out) return;

    // Don't leak if outTransportSet is called repeatedly
    transportDestroy(&out->transport);
    out->transport = transport;
}

void
outFormatSet(out_t* out, format_t* format)
{
    if (!out) return;

    // Don't leak if outFormatSet is called repeatedly
    fmtDestroy(&out->format);
    out->format = format;
}

