#ifndef __BACKOFF_H__
#define __BACKOFF_H__

#define BACKOFF_ERROR      -1
#define BACKOFF_OK          0
#define BACKOFF_RETRY_LIMIT 1

int backoffInit(unsigned, unsigned, unsigned, unsigned);
int backoffGetTime(unsigned *);

#endif // __BACKOFF_H__
