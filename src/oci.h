#ifndef __OCI_H__
#define __OCI_H__

#include "scopestdlib.h"
#include "scopetypes.h"

/*
 *  API to manage OCI configuration file
 */
void *ociReadCfgIntoMem(const char*);
char *ociModifyCfg(const void *, const char *, const char*);
bool ociWriteConfig(const char *, const char *);

#endif // __OCI_H__
