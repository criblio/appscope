#ifndef __CIRCBUF_H__
#define __CIRCBUF_H__
#include <stdint.h>
#include <unistd.h>

/*
 * Note:
 * There are 2 things to be aware of with this implementation.
 *
 * 1) There are normally a few utility functions that would be expected
 * with a circular buffer capability. For example, is the buffer empty,
 * or full, how many entries are in the buffer. We have made a deliberate
 * decision not to add these utility functions as they represent a reference
 * at the time the function is called, but the results can change from the
 * time a utility function is called and the time a put or get may be called.
 * If the utility were to be used as a means to decide if a get or put should
 * be done, then the results could be misleading. The thought is, just call
 * put/get directly and examine the return code. If utility would be used in
 * a way to determine if a put/get should be called a larger atomic activity
 * would be needed. We don't want to impose that at this time.
 *
 * 2) There is a common question with a circular buffer; what is the behavior
 * of a put when the buffer is full. Should new data overwrite existing data
 * or should the put return an error when the buffer is full? The ultimate
 * answer relates to how to respond to back pressure when data can't be
 * consumed as fast as it is being applied. Should we keep the first set of
 * data or should we keep the latest data in a back pressure situation?
 * We have chosen to not support overwrite at this point as it complicates the
 * mult-threaded aspects and we are not sure if it is needed at this point.
 */

typedef struct circbuf_t {
    uint64_t *buffer;
    int head;
    int tail;
    int maxlen;
} cbuf_t;

typedef cbuf_t * cbuf_handle_t ;

// Given number of entries in a circbuf, return a circular buffer handle
cbuf_handle_t cbufInit(size_t size);

// Free the cbuf itself, not the buffers
void cbufFree(cbuf_handle_t cbuf);

// Reset to empty, head == tail
void cbufReset(cbuf_handle_t cbuf);

// Add to the cbuf, if there is room
// 0 on success, -1 if buffer is full
int cbufPut(cbuf_handle_t cbuf, uint64_t data);

// Get an entry fromn the cbuf
// 0 on success, -1 if the buffer is empty
int cbufGet(cbuf_handle_t cbuf, uint64_t *data);

// Returns max capacity of the cbuf
size_t cbufCapacity(cbuf_handle_t cbuf);

// True if the circbuf is empty, else False
int cbufEmpty(cbuf_handle_t cbuf);

#endif // __CIRCBUF_H__
