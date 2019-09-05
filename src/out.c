#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "out.h"

struct _out_t
{
    transport_t* transport;
    format_t* format;
    log_t* log_ref;             // for routing out to log
};

out_t*
outCreate()
{
    out_t* out = calloc(1, sizeof(out_t));
    if (!out) return NULL;

    return out;
}

void
outDestroy(out_t** out)
{
    if (!out || !*out) return;
    out_t* o = *out;
    transportDestroy(&o->transport);
    fmtDestroy(&o->format);
    // log_ref isn't owned by us, we just have a ref to it.  Don't destroy it.
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

    char* msg = fmtString(out->format, e);
    int rv = outSend(out, msg);
    if (out->log_ref) logSend(out->log_ref, msg, CFG_LOG_DEBUG);
    if (msg) free(msg);
    return rv;
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

// For debugging.  If a reference to the log is set then we'll route all 
// output events to logs too.
void
outLogReferenceSet(out_t* out, log_t* log)
{
    if (!out) return;
    out->log_ref = log;
}

// Getter funcs
int
outTransportDescriptor(out_t *out)
{
    if (out) return transportDescriptor(out->transport);

    return -1;
}
