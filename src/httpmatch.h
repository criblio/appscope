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

typedef struct _httpmatch_t httpmatch_t;

typedef void (*freeReq_fn)(http_map *);

httpmatch_t *httpMatchCreate(net_info const * const, list_t const * const, freeReq_fn);
void         httpMatchDestroy(httpmatch_t **);


bool         httpReqSave(httpmatch_t *, http_map *);
http_map *   httpReqGet(httpmatch_t *, uint64_t);
void         httpReqDelete(httpmatch_t *, uint64_t);
bool         httpReqExpire(httpmatch_t *, uint64_t);





#endif // __HTTPMATCH_H__
