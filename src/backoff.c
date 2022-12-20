#define _GNU_SOURCE
#include "backoff.h"
#include "dbg.h"
#include "scopestdlib.h"


// Implementation based on this recommendation:
// https://cloud.google.com/iot/docs/how-tos/exponential-backoff


struct _backoff_t {
    unsigned int backoff_seconds;  // 1s, 2s, 4s, 8s, ...
    unsigned int backoff_ms;       // current backoff in ms (includes jitter)
    unsigned int ms_count;         // number of ms we've waited so far
};

const unsigned int max_backoff_seconds = 64 * 4; // 4min 16s


backoff_t *
backoffCreate(void)
{
    backoff_t *backoff = scope_calloc(1, sizeof(backoff_t));
    if (!backoff) {
        DBG(NULL);
        return NULL;
    }

    backoff->backoff_seconds = 1;
    backoff->backoff_ms = 1;
    backoff->ms_count = 0;

    scope_srand(scope_getpid());

    return backoff;
}


void
backoffDestroy(backoff_t **backoff)
{
    if (!backoff || !*backoff) return;
    backoff_t *backoff_p = *backoff;

    scope_free(backoff_p);
    *backoff = NULL;
}


int
backoffAlgoAllowsConnect(backoff_t *backoff)
{
    // If there isn't a backoff algo, always allow the connection.
    if (!backoff) return 1;

    int allowConnect = (++backoff->ms_count % backoff->backoff_ms == 0);

    if (allowConnect) {
        // for next retry period
        int jitter_ms = scope_rand() % 1000;
        backoff->backoff_ms = (backoff->backoff_seconds * 1000) + jitter_ms;

        if (backoff->backoff_seconds < max_backoff_seconds) {
            backoff->backoff_seconds = backoff->backoff_seconds * 2;
        }
        backoff->ms_count = 0;
    }

    return allowConnect;
}

