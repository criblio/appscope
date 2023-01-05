#define _GNU_SOURCE
#include "backoff.h"
#include "dbg.h"
#include "scopestdlib.h"


// Implementation based on this recommendation:
// https://cloud.google.com/iot/docs/how-tos/exponential-backoff


struct _backoff_t {
    unsigned int backoff_base;     // 1000ms, 2000ms, 4000ms ...
    unsigned int backoff_limit;    // current backoff in ms (includes jitter)
    unsigned int ms_count;         // number of ms we've waited so far
};

// 4min 16s
#define BACKOFF_BASE_MAX ( 64 * 4 * 1000)


backoff_t *
backoffCreate(void)
{
    backoff_t *backoff = scope_calloc(1, sizeof(backoff_t));
    if (!backoff) {
        DBG(NULL);
        return NULL;
    }

    backoffReset(backoff);

    return backoff;
}

void
backoffReset(backoff_t *backoff)
{
    if (!backoff) return;

    backoff->backoff_base = 1000;
    backoff->backoff_limit = 1;
    backoff->ms_count = 0;
}

void
backoffDestroy(backoff_t **backoff)
{
    if (!backoff || !*backoff) return;
    backoff_t *backoff_p = *backoff;

    scope_free(backoff_p);
    *backoff = NULL;
}


bool
backoffAlgoAllowsConnect(backoff_t *backoff)
{
    // If there isn't a backoff algo, always allow the connection.
    if (!backoff) return TRUE;

    if (++backoff->ms_count >= backoff->backoff_limit) {

        // A connetion attempt is allowed.  Init for the next retry period.
        int jitter_ms = scope_rand() % 1000;
        backoff->backoff_limit = (backoff->backoff_base) + jitter_ms;

        if (backoff->backoff_base < BACKOFF_BASE_MAX) {
            backoff->backoff_base = backoff->backoff_base * 2;
        }
        backoff->ms_count = 0;
        return TRUE;
    }

    // It's not time to allow a connection attempt
    return FALSE;
}

