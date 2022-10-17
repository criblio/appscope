#ifndef __SETUP_H__
#define __SETUP_H__

#include "scopetypes.h"

#define SCOPE_FILTER_USR_PATH ("/usr/lib/appscope/scope_filter")
#define SCOPE_FILTER_TMP_PATH ("/tmp/appscope/scope_filter")

service_status_t setupService(const char *);
int setupConfigure(void *, size_t, bool);
char *setupLoadFileIntoMem(size_t *, const char *);

#endif // __SETUP_H__
