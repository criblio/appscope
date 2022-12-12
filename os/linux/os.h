#ifndef __OS_H__
#define __OS_H__

#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>
#include <dirent.h>
#include <sys/vfs.h>
#include <sys/prctl.h>
#include <sys/epoll.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sock_diag.h>
#include <linux/unix_diag.h>
#include <link.h>
#include <sys/mman.h>

#include "../../src/plattime.h"
#include "../../contrib/funchook/include/funchook.h"

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
extern int osGetProcUidGid(pid_t, uid_t *, gid_t *);
extern int osGetNumThreads(pid_t);
extern int osGetNumFds(pid_t);
extern int osGetNumChildProcs(pid_t);
extern int osInitTimer(platform_time_t *);
extern int osGetProcMemory(pid_t);
extern int osIsFilePresent(const char *);
extern int osGetCmdline(pid_t, char **);
extern bool osThreadInit(void(*handler)(int), unsigned);
extern int osUnixSockPeer(ino_t);
extern void osInitJavaAgent(void);
extern int osGetPageProt(unsigned long);
extern int osGetExePath(pid_t, char **);
extern bool osTimerStop(void);
extern bool osGetCgroup(pid_t, char *, size_t);
extern char *osGetFileMode(mode_t);
extern int osNeedsConnect(int);
extern char *osGetUserName(unsigned);
extern char *osGetGroupName(unsigned);
extern long long osGetProcCPU(void);
extern uint64_t osFindLibrary(const char *, pid_t, bool);
extern int osFindFd(pid_t, const char *);
extern void osCreateSM(proc_id_t *, unsigned long);

#endif  //__OS_H__
