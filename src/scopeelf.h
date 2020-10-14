#ifndef __SCOPEELF_H__
#define __SCOPEELF_H__

#ifdef __LINUX__

#include <elf.h>
#include <link.h>

typedef struct {
    const char *symbol;
    void *func;
    void *gfn;
} got_list_t;

typedef struct {
    char *buf;
    int len;
    unsigned char *text_addr;
    uint64_t       text_len;
} elf_buf_t;

void freeElf(char *, size_t);
elf_buf_t * getElf(char *);
int doGotcha(struct link_map *, got_list_t *, Elf64_Rela *, Elf64_Sym *, char *, int);
int getElfEntries(struct link_map *, Elf64_Rela **, Elf64_Sym **, char **, int *rsz);
void * getSymbol(const char *, char *);
int checkEnv(char *, char *);

#endif // __LINUX__

#endif // __SCOPEELF_H__
