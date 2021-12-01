#include "../../src/plattime.h"
#include <dlfcn.h>
#include <errno.h>
#include <libproc.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef CMSG_ALIGN
#define CMSG_ALIGN(n) __DARWIN_ALIGN32(n)
#endif

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

extern int osGetProcname(char *, size_t);
extern int osGetNumThreads(pid_t);
extern int osGetNumFds(pid_t);
extern int osGetNumChildProcs(pid_t);
extern int osInitTSC(platform_time_t *);
extern int osGetProcMemory(pid_t);
extern int osIsFilePresent(pid_t, const char *);
extern int osGetCmdline(pid_t, char **);
extern bool osThreadInit(void (*handler)(int), unsigned);
extern int osUnixSockPeer(ino_t);
extern void osInitJavaAgent(void);
extern int osGetPageProt(uint64_t);
extern bool osTimerStop(void);
extern bool osGetCgroup(pid_t, char *, size_t);
extern char *osGetFileMode(mode_t);
extern int osNeedsConnect(int);
