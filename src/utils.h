#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>

typedef struct {
    const char *str;
    unsigned val;
} enum_map_t;

unsigned int strToVal(enum_map_t[], const char *);
const char *valToStr(enum_map_t[], unsigned int);

int checkEnv(char *, char *);
void setPidEnv(int);
char *getpath(const char *);

int startsWith(const char *string, const char *substring);
int endsWith(const char *string, const char *substring);

int sigSafeNanosleep(const struct timespec *req);

#endif // __UTILS_H__
