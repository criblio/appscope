#define _GNU_SOURCE
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdlib.h>
#include "dbg.h"
#include "atomic.h"
#include "circbuf.h"

cbuf_handle_t
cbufInit(size_t size)
{
    cbuf_handle_t cbuf = calloc(1, sizeof(struct circbuf_t));
    if (!cbuf) {
        DBG("Circbuf:calloc");
        return NULL;
    }
    
    uint64_t *buffer = calloc(size + 1, sizeof(uint64_t));
    if (!buffer) {
        free(cbuf);
        DBG("Circbuf:calloc");
        return NULL;
    }

    cbufReset(cbuf);
    cbuf->maxlen = size + 1;
    cbuf->buffer = buffer;
    return cbuf;
}

void
cbufFree(cbuf_handle_t cbuf)
{
    if (!cbuf) return;
    if (cbuf->buffer) free(cbuf->buffer);
    free(cbuf);
    return;
}

void
cbufReset(cbuf_handle_t cbuf)
{
    if (!cbuf) return;

    cbuf->head = 0;
    cbuf->tail = 0;
    return;
}

int
cbufPut(cbuf_handle_t cbuf, uint64_t data)
{
    int head, head_next, attempts, success;

    if (!cbuf) return -1;
    attempts = success = 0;

    do {
        head = cbuf->head;
        head_next = (head + 1) % cbuf->maxlen;
        if (head_next == cbuf->tail) {
            // Note: we commented this out as it caused a
            // double free error when running with 100,000
            // Go routines. We should determine why.
            DBG("maxlen: %"PRIu64, cbuf->maxlen); // Full
            break;
        }
        success = atomicCas32(&cbuf->head, head, head_next);
    } while (!success && (attempts++ < cbuf->maxlen));

    if (success) {
        if (cbuf->buffer[head_next] != 0) {
            // We expect that the entry is not used; has been read
            DBG(NULL);
            return -1;
        }
        cbuf->buffer[head_next] = data;
        return 0;
    }

    return -1;
}

int
cbufGet(cbuf_handle_t cbuf, uint64_t *data)
{
    int tail, tail_next, attempts, success;
    if (!cbuf || !data) return -1;

    attempts = success = 0;

    do {
        tail = cbuf->tail;
        tail_next = (tail + 1) % cbuf->maxlen;
        if (tail == cbuf->head) break; // Empty
        success = atomicCas32(&cbuf->tail, tail, tail_next);
    } while (!success && (attempts++ < cbuf->maxlen));

    if (success) {
        *data = cbuf->buffer[tail_next];
        if (*data == 0) {
            // We expect data before we read
            // Should we bail out here?
            DBG(NULL);
        }

        // Setting data to 0 to indicate to a put that we're empty
        cbuf->buffer[tail_next] = 0ULL;
        return 0;
    }

    return -1;
}

size_t
cbufCapacity(cbuf_handle_t cbuf)
{
    if (!cbuf) return -1;
    return cbuf->maxlen - 1;
}

int
cbufEmpty(cbuf_handle_t cbuf)
{
    if (cbuf->tail == cbuf->head) return TRUE;
    return FALSE;
}
