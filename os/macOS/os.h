#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <sys/sysctl.h>
#include <libproc.h>
#include <sys/resource.h>

#include "../../src/wrap.h"

#define STATMODTIME(sb) sb.st_mtimespec.tv_sec

#ifndef off64_t
typedef uint64_t off64_t;
#endif
#ifndef fpos64_t
typedef uint64_t fpos64_t;
#endif

#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif

extern struct rtconfig_t g_cfg;

extern int osGetProcname(char *, size_t);
extern int osGetNumThreads(pid_t);
extern int osGetNumFds(pid_t);
extern int osGetNumChildProcs(pid_t);
extern int osInitTSC(struct rtconfig_t *);
extern int osGetProcMemory(pid_t);
extern int osIsFilePresent(pid_t, const char *);
extern int osGetCmdline(pid_t, char **);
extern bool osThreadInit(void(*handler)(int), unsigned);
extern int osUnixSockPeer(ino_t);
