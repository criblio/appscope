#ifndef __SETUP_H__
#define __SETUP_H__

#include <unistd.h>

#include "scopetypes.h"

char *setupLoadFileIntoMem(size_t *, const char *);
bool isCfgFileConfigured(const char *);
int removeScopeCfgFile(const char *);
service_status_t setupService(const char *, uid_t, gid_t);
service_status_t setupUnservice(void);
int setupMount(const char *, uid_t, gid_t);
int setupFilter(void *, size_t, uid_t, gid_t);
int setupPreload(const char *, uid_t, gid_t);

#endif // __SETUP_H__
