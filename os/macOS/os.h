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

extern int osGetProcname(char *, size_t);
