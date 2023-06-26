#define _GNU_SOURCE

#include "dbg.h"
#include "evtutils.h"
#include "report.h"
#include "scopestdlib.h"
#include "state_private.h"


static protocol_info *
evtProtoCreateHttpBase(void)
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
evtProtoCreateHttp1(bool isResponse)
{
    protocol_info * proto = evtProtoCreateHttpBase();
    if (!proto) {
        DBG(NULL);
        return NULL;
    }

    proto->ptype = (isResponse) ? EVT_HRES : EVT_HREQ;

    return proto;
}

protocol_info *
evtProtoCreateHttp2Frame(uint32_t frameLen)
{
    protocol_info *proto = evtProtoCreateHttpBase();
    char *frame = scope_malloc(frameLen);
    if (!proto || !frame) {
        DBG(NULL);
        if (proto) evtProtoDelete(proto);
        if (frame) scope_free(frame);
        return NULL;
    }

    proto->ptype = EVT_H2FRAME;
    http_post *post = (http_post *)proto->data;
    post->hdr = frame;

    return proto;
}

protocol_info *
evtProtoCreateDetect(const char * const protocolName)
{
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
evtProtoDelete(protocol_info *proto)
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
evtDelete(evt_type *event)
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
            // Alloc'd in evtProtoCreateHttp1, evtProtoCreateHttp2Frame, or
            // evtProtoCreateDetect
            evtProtoDelete(proto);
            break;
        }
        default:
            DBG(NULL);
            scope_free(event);
    }
    return TRUE;
}

