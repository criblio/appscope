#define _GNU_SOURCE

#include "dbg.h"
#include "evtutils.h"
#include "report.h"
#include "scopestdlib.h"
#include "state_private.h"

protocol_info *
evtProtoCreate(void)
{
    protocol_info *proto = scope_calloc(1, sizeof(struct protocol_info_t));
    http_post *post = scope_calloc(1, sizeof(struct http_post_t));
    if (!proto || !post) {
        if (post) scope_free(post);
        if (proto) scope_free(proto);
        return NULL;
    }

    proto->evtype = EVT_PROTO;
    proto->data = (char *)post;

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
            // Alloc'd in evtProtoCreate (http 1&2) or
            // setProtocol (protocolDetection)
            evtProtoDelete(proto);
            break;
        }
        default:
            DBG(NULL);
            scope_free(event);
    }
    return TRUE;
}

