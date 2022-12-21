#ifndef __SETUP_H__
#define __SETUP_H__

#include "../scopetypes.h"

bool isCfgFileConfigured(const char *);
int removeScopeCfgFile(const char *);
service_status_t setupService(const char *, uid_t, gid_t);
service_status_t setupUnservice(void);
int setupConfigure(void *, size_t, uid_t, gid_t);
int setupUnconfigure(void);
char *setupLoadFileIntoMem(size_t *, const char *);

#endif // __SETUP_H__
