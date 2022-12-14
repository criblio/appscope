#ifndef __LOADERUTILS_H__
#define __LOADERUTILS_H__

#ifdef __linux__

#include <elf.h>
#include <link.h>

typedef struct {
    char *cmd;
    char *buf;
    int len;
    unsigned char *text_addr;
    uint64_t       text_len;
} elf_buf_t;

void freeElf(char *, size_t);
elf_buf_t * getElf(char *);
bool is_static(char *);
bool is_go(char *);
void setPidEnv(int);
char *getpath(const char *);

#endif // __linux__

#endif // __LOADERUTILS_H__
