#include "pthread_impl.h"

int __pthread_setcancelstate(int new, int *old)
{
	if (new > 2U) return EINVAL;
	return 0;
}

weak_alias(__pthread_setcancelstate, pthread_setcancelstate);
