#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>

#define DEVMODE 0
#define __NR_memfd_create 319
#define _MFD_CLOEXEC 0x0001U


typedef struct {
    const char *str;
    unsigned val;
} enum_map_t;

typedef struct libscope_info_t {
    char *path;
    char *shm_name;
    int fd;
    int use_memfd;
} libscope_info;


unsigned int strToVal(enum_map_t[], const char*);
const char* valToStr(enum_map_t[], unsigned int);

int checkEnv(char *, char *);
void setPidEnv(int);
char *getpath(const char *);

int startsWith(const char *string, const char *substring);
int endsWith(const char *string, const char *substring);

int sigSafeNanosleep(const struct timespec *req);
int extract_bin(char *, libscope_info *, unsigned char *, unsigned char *);
void release_bin(libscope_info *);
int setup_loader(char *, char *);
void setEnvVariable(char *, char *);

#endif // __UTILS_H__
