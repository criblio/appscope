#ifndef __INJECT_H__
#define __INJECT_H__

#define SHM_NAME_INJECT         "libscopea"
#define LIBSCOPE_INJECTED_PATH  "/dev/shm/" SHM_NAME_INJECT

void injectScope(int, char*);

#endif