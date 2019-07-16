#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "out.h"

struct _out_t
{
    transport_t* transport;
    char* prefix;
};

#define DEFAULT_PREFIX NULL

out_t*
outCreate()
{
    out_t* out = calloc(1, sizeof(out_t));
    if (!out) return NULL;

    out->prefix = (DEFAULT_PREFIX) ? strdup(DEFAULT_PREFIX) : NULL;

    return out;
}

void
outDestroy(out_t** out)
{
    if (!out || !*out) return;
    out_t* o = *out;
    transportDestroy(&o->transport);
    if (o->prefix) free(o->prefix);
    free(o);
    *out = NULL;
}

int
outSend(out_t* out, const char* msg)
{
    if (!out || !msg) return -1;

    return transportSend(out->transport, msg);
}

const char*
outStatsDPrefix(out_t* out)
{
    return (out) ? out->prefix : DEFAULT_PREFIX;
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
outStatsDPrefixSet(out_t* out, const char* prefix)
{
    if (!out) return;

    // Don't leak on repeated sets
    if (out->prefix) free(out->prefix);
    out->prefix = (prefix) ? strdup(prefix) : NULL;
}


