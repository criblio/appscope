#define _GNU_SOURCE
#include "backoff.h"
#include "scopestdlib.h"

struct _backoff_t {
    unsigned retry_counter; // current number of retries
    unsigned retry_max;     // maximum number of retries
    unsigned max_delay_ms;  // maximum delay [ms]
    unsigned current_delay_ms; // current max delay [ms]
    // unsigned previous_time_ms; // used only for decorrelated jitter
    // unsigned base;             // used only for decorrelated jitter
};

static struct _backoff_t backoff;

int
backoffInit(unsigned retry_max, unsigned start_timeout_ms, unsigned max_delay_ms, unsigned seed)
{
    if (start_timeout_ms > max_delay_ms) {
        return BACKOFF_ERROR;
    }
    backoff.retry_counter = 0;
    backoff.retry_max = retry_max;

    backoff.max_delay_ms = max_delay_ms;
    backoff.current_delay_ms = start_timeout_ms;
    // backoff.previous_time_ms = start_timeout_ms;
    // backoff.base = start_timeout_ms;

    scope_srandom(seed);
    return BACKOFF_OK;
}

int
backoffGetTime(unsigned *backoff_time_ms)
{
    // Handle the total number of counter
    if (backoff.retry_counter == backoff.retry_max) {
        return BACKOFF_RETRY_LIMIT;
    }
    backoff.retry_counter++;

    // Full Jitter: calculate random value between <0 and Current delay>
    // sleep = random between (0 and min(max_delay, base * 2 ** attempt))
    *backoff_time_ms = (unsigned) (scope_random() % (backoff.current_delay_ms + 1U));

    // Alternative: Decorrelated Jitter:
    // sleep = min(max_delay, random between (base, 3 * previous_sleep))
    // *backoff_time_ms = (unsigned) ((scope_random() % (backoff.previous_time_ms * 3 - backoff.base + 1)) + backoff.base);

    // Double the value of current delay or set it to max
    backoff.current_delay_ms <<= 1U;
    if (backoff.current_delay_ms > backoff.max_delay_ms) {
        backoff.current_delay_ms = backoff.max_delay_ms;
    }

    return BACKOFF_OK;
}
