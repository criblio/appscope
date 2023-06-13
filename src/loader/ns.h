#ifndef __NS_H__
#define __NS_H__

#include <stdbool.h>
#include <unistd.h>

#include "scopetypes.h"

int nsForkAndExec(pid_t, pid_t, bool);
bool setNamespaceRootDir(const char *, pid_t, const char *);

// TODO migrate these into loader.c commands
int nsAttach(pid_t, const char *);
int nsInstall(const char *, pid_t, libdirfile_t);

#endif // __NS_H__
