#define _GNU_SOURCE

#include "dbg.h"
#include "evtutils.h"
#include "report.h"
#include "scopestdlib.h"
#include "state_private.h"


static protocol_info *
evtProtoAllocHttpBase(void)
{
    protocol_info *proto = scope_calloc(1, sizeof(protocol_info));
    http_post *post = scope_calloc(1, sizeof(http_post));
    if (!proto || !post) {
        if (post) scope_free(post);
        if (proto) scope_free(proto);
        return NULL;
    }

    proto->evtype = EVT_PROTO;
    proto->data = (char *)post;

    return proto;
}

protocol_info *
evtProtoAllocHttp1(bool isResponse)
{
    protocol_info * proto = evtProtoAllocHttpBase();
    if (!proto) {
        DBG(NULL);
        return NULL;
    }

    proto->ptype = (isResponse) ? EVT_HRES : EVT_HREQ;

    return proto;
}

protocol_info *
evtProtoAllocHttp2Frame(uint32_t frameLen)
{
    protocol_info *proto = evtProtoAllocHttpBase();
    char *frame = scope_malloc(frameLen);
    if (!proto || !frame) {
        DBG(NULL);
        if (proto) evtProtoFree(proto);
        if (frame) scope_free(frame);
        return NULL;
    }

    proto->ptype = EVT_H2FRAME;
    http_post *post = (http_post *)proto->data;
    post->hdr = frame;

    return proto;
}

protocol_info *
evtProtoAllocDetect(const char * const protocolName)
{
    if (!protocolName) return NULL;

    protocol_info *proto = scope_calloc(1, sizeof(protocol_info));
    char *protname = scope_strdup(protocolName);
    if (!proto || !protname) {
        DBG(NULL);
        if (protname) scope_free(protname);
        if (proto) scope_free(proto);
        return NULL;
    }

    proto->evtype = EVT_PROTO;
    proto->ptype = EVT_DETECT;
    proto->data = protname;

    return proto;
}

bool
evtProtoFree(protocol_info *proto)
{
    if (!proto) return FALSE;

    if ((proto->ptype == EVT_HREQ) || (proto->ptype == EVT_HRES) || (proto->ptype == EVT_H2FRAME)) {
        http_post *post = (http_post *)proto->data;
        if (post) {
            if (post->hdr) scope_free(post->hdr);
            scope_free(post);
        }
    } else if (proto->ptype == EVT_DETECT) {
        // proto->data is a pointer to a strdup'd string
        if (proto->data) scope_free(proto->data);
    }
    scope_free(proto);
    return TRUE;
}

bool
evtFree(evt_type *event)
{
    if (!event) return FALSE;

    switch(event->evtype) {
        case EVT_NET:
        {
            // Alloc'd in postNetState. There are no nested allocations.
            // net_info *net = (net_info *)event;
            scope_free(event);
            break;
        }
        case EVT_FS:
        {
            // Alloc'd in postFSState. There are no nested allocations.
            // fs_info *fs = (fs_info *)event;
            scope_free(event);
            break;
        }
        case EVT_ERR:
        {
            // Alloc'd in postStatErrState. There are no nested allocations.
            // stat_err_info *staterr = (stat_err_info *)event;
            scope_free(event);
            break;
        }
        case EVT_STAT:
        {
            // Alloc'd in postStatErrState. There are no nested allocations.
            // stat_err_info *staterr = (stat_err_info *)event;
            scope_free(event);
            break;
        }
        case EVT_DNS:
        {
            // Alloc'd in postDNSState. There are no nested allocations.
            // net_info *net = (net_info *)event;
            scope_free(event);
            break;
        }
        case EVT_PROTO:
        {
            protocol_info *proto = (protocol_info *)event;
            // Alloc'd in evtProtoAllocHttp1, evtProtoAllocHttp2Frame, or
            // evtProtoAllocDetect
            evtProtoFree(proto);
            break;
        }
        case EVT_SEC:
        {
            // Alloc'd in gotSecurity, funcSecurity, fileSecurity
            // netSecurity, dnsSecurity, envSecurity
            // security_info_t *sec = (security_info_t *)event;
            scope_free(event);
            break;
        }
        default:
            DBG(NULL);
            scope_free(event);
    }
    return TRUE;
}

