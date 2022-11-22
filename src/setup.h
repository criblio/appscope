#ifndef __SETUP_H__
#define __SETUP_H__

#include "scopetypes.h"

#define SCOPE_FILTER_USR_PATH ("/usr/lib/appscope/scope_filter")
#define SCOPE_FILTER_TMP_PATH ("/tmp/appscope/scope_filter")

bool isCfgFileConfigured(const char *);
int removeScopeCfgFile(const char *);
service_status_t setupService(const char *, uid_t, gid_t);
service_status_t setupUnservice(uid_t, gid_t);
int setupConfigure(void *, size_t, uid_t, gid_t);
int setupUnconfigure(uid_t, gid_t);
char *setupLoadFileIntoMem(size_t *, const char *);

#endif // __SETUP_H__
