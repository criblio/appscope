#ifndef __SETUP_H__
#define __SETUP_H__

#include <unistd.h>

#include "scopetypes.h"

char *setupLoadFileIntoMem(size_t *, const char *);
bool isCfgFileConfigured(const char *);
int removeScopeCfgFile(const char *);
service_status_t setupService(const char *, uid_t, gid_t);
service_status_t setupUnservice(void);
bool setupMount(pid_t, const char *, uid_t, gid_t);
bool setupRules(void *, size_t, uid_t, gid_t);
bool setupPreload(const char *, uid_t, gid_t);

#endif // __SETUP_H__
