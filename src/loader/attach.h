#ifndef __ATTACH_H__
#define __ATTACH_H__

#include <stdbool.h>

int load_and_attach(pid_t, char *);
int attach(pid_t);
int detach(pid_t);


#endif // __ATTACH_H__s
