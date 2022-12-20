#ifndef __LOADER_H__
#define __LOADER_H__
#include <sys/types.h>

extern int g_log_level;

int cmdService(char *, pid_t);
int cmdUnservice(pid_t);
int cmdConfigure(char *, pid_t);
int cmdUnconfigure(pid_t);
int cmdRun(bool, bool, pid_t, pid_t, int, char **, char **);

#endif // __LOADER_H__
