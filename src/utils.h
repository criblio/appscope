#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>

typedef struct {
    const char *str;
    unsigned val;
} enum_map_t;

unsigned int strToVal(enum_map_t[], const char*);
const char* valToStr(enum_map_t[], unsigned int);

int checkEnv(char *, char *);
int fullSetenv(const char *, const char *, int);
void setPidEnv(int);
char *getpath(const char *);

int startsWith(const char *string, const char *substring);
int endsWith(const char *string, const char *substring);

int sigSafeNanosleep(const struct timespec *req);

void setUUID(char *string);
void setMachineID(char *string);
int createMachineID(char *string);
void generateMD5(const char *data, int len, char *md5_buf);

#endif // __UTILS_H__
