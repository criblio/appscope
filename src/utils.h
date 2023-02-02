#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>
#include "scopetypes.h"

typedef struct {
    const char *str;
    unsigned val;
} enum_map_t;

unsigned int strToVal(enum_map_t[], const char *);
const char* valToStr(enum_map_t[], unsigned int);

int checkEnv(char *, char *);
int fullSetenv(const char *, const char *, int);
void setPidEnv(int);
char *getpath(const char *);

int startsWith(const char *, const char *);
int endsWith(const char *, const char *);

void setUUID(char *);
void setMachineID(char *);

// Signal Safe API
int sigSafeNanosleep(const struct timespec *);
void sigSafeUtoa(unsigned long, char *, int, int *);
bool sigSafeMkdirRecursive(const char *);
ssize_t sigSafeWrite(int , const void *, size_t);
ssize_t sigSafeWriteNumber(int , long, int);

#endif // __UTILS_H__
