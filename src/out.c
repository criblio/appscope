#include <stddef.h>
#include "out.h"

struct _out_t
{

};

out_t*
outCreate()
{
    return NULL;
}

void
outDestroy(out_t** out)
{
}

int
outSend(out_t* out, char* msg)
{
    return -1;
}

char*
outStatsDPrefix(out_t* out)
{
    return NULL;
}

void
outSetTransport(out_t* out, transport_t* transport)
{
}

void
outStatsDPrefixSet(out_t* out, char* prefix)
{
}
