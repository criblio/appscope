#ifndef __BACKOFF_H__
#define __BACKOFF_H__

// This was written to keep state of connections so we can wait
// and appropriate amount of time between attempting connections
// and retrying them later.

// Everything here expects that backoffAlgoAllowsConnect()
// is called at a 1ms frequency.


typedef struct _backoff_t backoff_t;

// Constructors Destructors
backoff_t *   backoffCreate(void);
void          backoffDestroy(backoff_t **);

// Accessor
void          backoffReset(backoff_t *);
int           backoffAlgoAllowsConnect(backoff_t *);

#endif // __BACKOFF_H__
