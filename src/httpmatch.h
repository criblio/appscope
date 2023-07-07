#ifndef __HTTPMATCH_H__
#define __HTTPMATCH_H__

#include "linklist.h"
#include "state.h"
#include "state_private.h"




// A http matcher holds onto http requests until they can be paired with http
// responses.  This pairing allows response events and metrics to include
// important information about the request that does not exist in the response
// itself, like what target url was being requested.

// An event circular buffer decouples the datapath (client threads) from our
// reporting (done on our own periodic thread).  This http matcher is used
// by the reporting side, where any work to output events can be done without
// delaying client threads.

// When all requests and responses on the datapath side can be added to the
// event circular buffer, we know we can delete saved requests whenever we
// pair them with a response.  We'll be able to free memory for requests
// as reponses are received.

// Things can get ugly for http matching when the circular buffer toggles
// between being full and being empty.  In this state, on the reporting side,
// it's possible for us to receive a request but never receive the response
// to pair it with.  We have the "httpReqExpire" functionality to
// handle this - to avoid unbounded growth in saved requests.

typedef struct _store_t httpmatch_t;

typedef void (*freeReq_fn)(http_map *);

httpmatch_t *httpMatchCreate(net_info const * const, list_t const * const, freeReq_fn);
void         httpMatchDestroy(httpmatch_t **);

bool         httpReqSave(httpmatch_t *, http_map *);
http_map *   httpReqGet(httpmatch_t *, uint64_t);
bool         httpReqDelete(httpmatch_t *, uint64_t);
bool         httpReqExpire(httpmatch_t *, uint64_t, bool);


// Similar thing is going on for http2, but instead of storing requests,
// we're storing channels which contain an stateful hpack decoder which
// is common to all of the streams flowing over that channel.  If we miss
// a channel close, there is potentially a lot of state (hpack decoder
// and set of streams) that can be leaked if we don't have a mechanism
// to expire channels that are no longer used.
//

// saved state for an HTTP/2 channel
struct http2Channel;
typedef struct http2Channel http2Channel_t;

typedef struct _store_t channelstore_t;

typedef void (*freeChannel_fn)(http2Channel_t *);

channelstore_t *channelStoreCreate(net_info const * const, list_t const * const, freeChannel_fn);
void            channelStoreDestroy(channelstore_t **);

bool            channelSave(channelstore_t *, http2Channel_t *, uint64_t, int);
http2Channel_t *channelGet(channelstore_t *, uint64_t);
bool            channelDelete(channelstore_t *, uint64_t);
bool            channelExpire(channelstore_t *, uint64_t, bool);



#endif // __HTTPMATCH_H__
