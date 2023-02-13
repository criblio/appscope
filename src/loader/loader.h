#ifndef __LOADER_H__
#define __LOADER_H__

#include <unistd.h>

extern int g_log_level;

int cmdService(char *, pid_t);
int cmdUnservice(pid_t);
int cmdConfigure(char *, pid_t);
int cmdUnconfigure(pid_t);
int cmdGetFile(char *, pid_t);
int cmdAttach(bool, bool, pid_t, pid_t);
int cmdRun(bool, bool, pid_t, pid_t, int, char **);

#endif // __LOADER_H__s
