#ifndef __OS_H__
#define __OS_H__

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sock_diag.h>
#include <linux/unix_diag.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/vfs.h>
#include <unistd.h>

#include "../../contrib/funchook/include/funchook.h"
#include "../../src/plattime.h"

// Anecdotal evidence that a proc entry should be max 4096 bytes
#define MAX_PROC 4096

// Experimental results for the size of a maps file
#define MAX_MAPS 150000

#define STATMODTIME(sb) sb.st_mtime

#if DEBUG > 0 // defined in wrap.h
#define LOG_LEVEL CFG_LOG_ERROR
#else
#define LOG_LEVEL CFG_LOG_DEBUG
#endif

extern char *program_invocation_short_name;

extern int osGetProcname(char *, int);
extern int osGetNumThreads(pid_t);
extern int osGetNumFds(pid_t);
extern int osGetNumChildProcs(pid_t);
extern int osInitTimer(platform_time_t *);
extern int osGetProcMemory(pid_t);
extern int osIsFilePresent(pid_t, const char *);
extern int osGetCmdline(pid_t, char **);
extern bool osThreadInit(void (*handler)(int), unsigned);
extern int osUnixSockPeer(ino_t);
extern void osInitJavaAgent(void);
extern int osGetPageProt(unsigned long);
extern int osGetExePath(char **);
extern bool osTimerStop(void);
extern bool osGetCgroup(pid_t, char *, size_t);
extern char *osGetFileMode(mode_t);
extern int osNeedsConnect(int);

#endif //__OS_H__
