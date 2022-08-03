#define _GNU_SOURCE
#include <errno.h>
#include <unistd.h>
#include "pthread_impl.h"

int errno_val[1024 * 64];

int *__errno_location(void)
{
	return &errno_val[gettid()];
}

weak_alias(__errno_location, ___errno_location);
