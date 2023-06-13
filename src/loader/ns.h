#ifndef __NS_H__
#define __NS_H__

#include <stdbool.h>
#include <unistd.h>

#include "scopetypes.h"

int nsForkAndExec(pid_t, pid_t, bool);
service_status_t nsService(pid_t, const char *);
service_status_t nsUnservice(pid_t);
int nsAttach(pid_t, const char *);
int nsDetach(pid_t, const char *);
int nsInstall(const char *, pid_t, libdirfile_t);
int nsPreload(const char *, pid_t);
int nsMount(const char *, pid_t, const char *);
int nsFilter(const char *, pid_t, void *, size_t);

#endif // __NS_H__
