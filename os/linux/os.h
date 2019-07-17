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

#include "../../src/wrap.h"

extern char *program_invocation_short_name;
extern struct interposed_funcs_t g_fn;
extern int osGetProcname(char *, int);
extern int osGetNumThreads(pid_t);
extern int osGetNumFds(pid_t);
extern int osGetNumChildProcs(pid_t);
