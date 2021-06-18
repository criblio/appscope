
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/user.h>
#include <unistd.h>
#include <string.h>
#include <link.h>
#include <errno.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dlfcn.h>
#include <stddef.h>
#include "dbg.h"
#include "inject.h"

#define __RTLD_DLOPEN	0x80000000

typedef struct {
    char *path;
    uint64_t addr;
} libdl_info_t;

static uint64_t 
findLibrary(const char *library, pid_t pid) 
{
    char filename[PATH_MAX];
    char buffer[9076];
    FILE *fd;
    uint64_t addr = 0;

    snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
    if ((fd = fopen(filename, "r")) == NULL) {
        printf("Failed to open maps file for process %d\n", pid);
        return 0;
    }

    while(fgets(buffer, sizeof(buffer), fd)) {
        if (strstr(buffer, library)) {
            addr = strtoull(buffer, NULL, 16);
            break;
        }
    }

    fclose(fd);
    return addr;
}

static uint64_t 
freeSpaceAddr(pid_t pid) 
{
    FILE *fd;
    char filename[PATH_MAX];
    char line[850];
    uint64_t addr;
    char str[20];
    char perms[5];

    sprintf(filename, "/proc/%d/maps", pid);
    if ((fd = fopen(filename, "r")) == NULL) {
        printf("Failed to open maps file for process %d\n", pid);
        return 0;
    }

    while(fgets(line, 850, fd) != NULL) {
        sscanf(line, "%lx-%*x %s %*s %s %*d", &addr, perms, str);
        if (strstr(perms, "x") != NULL) break;
    }

    fclose(fd);
    return addr;
}

static void 
ptraceRead(int pid, uint64_t addr, void *data, int len)
{
    int numRead = 0;
    int i = 0;
    long word = 0;
    long *ptr = (long *) data;

    while (numRead < len) {
        word = ptrace(PTRACE_PEEKTEXT, pid, addr + numRead, NULL);
        if(word == -1) {
            perror("ptrace(PTRACE_PEEKTEXT) failed");
            exit(EXIT_FAILURE);
        }
        numRead += sizeof(word);
        ptr[i++] = word;
    }
}

static void 
ptraceWrite(int pid, uint64_t addr, void *data, int len) 
{
    long word = 0;
    int i = 0;

    for(i=0; i < len; i += sizeof(word), word=0) {
        memcpy(&word, data + i, sizeof(word));
        if (ptrace(PTRACE_POKETEXT, pid, addr + i, word) == -1) {
            perror("ptrace(PTRACE_POKETEXT) failed");
            exit(EXIT_FAILURE);
        }
    }
}

static void 
ptraceAttach(pid_t target) {
    int waitpidstatus;

    if(ptrace(PTRACE_ATTACH, target, NULL, NULL) == -1) {
        perror("ptrace(PTRACE_ATTACH) failed");
        exit(EXIT_FAILURE);
    }

    if(waitpid(target, &waitpidstatus, WUNTRACED) != target) {
        perror("waitpid failed");
        exit(EXIT_FAILURE);
    }
}

static void 
call_dlopen(void) 
{
    asm(
        "andq $0xfffffffffffffff0, %rsp \n" //align stack to 16-byte boundary
        "callq *%rax \n"
        "int $3 \n"
    );
}

static void call_dlopen_end() {}

static void 
inject(pid_t pid, uint64_t dlopenAddr, char *path) 
{
    struct user_regs_struct oldregs, regs;
    unsigned char *oldcode;
    int status;
    uint64_t freeAddr, codeAddr;
    int libpathLen;
    ptrdiff_t oldcodeSize;

    ptraceAttach(pid);

    // save registers
    ptrace(PTRACE_GETREGS, pid, NULL, &oldregs);
    memcpy(&regs, &oldregs, sizeof(struct user_regs_struct));

    // find free space
    freeAddr = freeSpaceAddr(pid);
    
    // back up the code
    libpathLen = strlen(path) + 1;
    oldcodeSize = (call_dlopen_end - call_dlopen) + libpathLen;
    oldcode = (unsigned char *)malloc(oldcodeSize);
    ptraceRead(pid, freeAddr, oldcode, oldcodeSize);

    // write the path to the library 
    ptraceWrite(pid, freeAddr, path, libpathLen);

    // inject the code right after the library path
    codeAddr = freeAddr + libpathLen + 1;
    ptraceWrite(pid, codeAddr, &call_dlopen, call_dlopen_end - call_dlopen); 

    // set RIP to point to the injected code
    regs.rip = codeAddr;
    regs.rax = dlopenAddr;               // address of dlopen
    regs.rdi = freeAddr;                 // dlopen's first arg - path to the library
    regs.rsi = RTLD_NOW | __RTLD_DLOPEN; // dlopen's second arg - flags

    ptrace(PTRACE_SETREGS, pid, NULL, &regs);

    // continue execution and wait until the target process is stopped
    ptrace(PTRACE_CONT, pid, NULL, NULL);
    waitpid(pid, &status, WUNTRACED);

    // if process has been stopped by SIGSTOP send SIGCONT signal along with PTRACE_CONT call
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
        ptrace(PTRACE_CONT, pid, SIGCONT, NULL);
        waitpid(pid, &status, WUNTRACED);
    }

    // make sure the target process was stoppend by SIGTRAP triggered by int 0x3
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {

        // check if the library has been successfully injected
        ptrace(PTRACE_GETREGS, pid, NULL, &regs);
        if (regs.rax != 0x0) {
            printf("Appscope library injected at %p\n", (void*)regs.rax);
        } else {
            fprintf(stderr, "Scope library could not be injected\n");
        }

        //restore the app's state
        ptraceWrite(pid, freeAddr, oldcode, oldcodeSize);
        ptrace(PTRACE_SETREGS, pid, NULL, &oldregs);
        ptrace(PTRACE_DETACH, pid, NULL, NULL);

    } else {
        fprintf(stderr, "Error: Process stopped for unknown reason\n");
        exit(EXIT_FAILURE);
    }
}

static int 
findLib(struct dl_phdr_info *info, size_t size, void *data)
{
    if (strstr(info->dlpi_name, "libc.so") != NULL) {
        char libpath[PATH_MAX];
        if (realpath(info->dlpi_name, libpath)) {
            ((libdl_info_t *)data)->path = libpath;
            ((libdl_info_t *)data)->addr = info->dlpi_addr;
            return 1;
        }
    }
    return 0;
}

void 
injectScope(int pid, char* path) 
{
    uint64_t remoteLib, localLib;
    void *dlopenAddr = NULL;
    libdl_info_t info;
   
    if (!dl_iterate_phdr(findLib, &info)) {
        fprintf(stderr, "Failed to find libc\n");
        exit(EXIT_FAILURE);
    }
 
    localLib = info.addr;
    dlopenAddr = dlsym(RTLD_DEFAULT, "__libc_dlopen_mode");
    if (dlopenAddr == NULL) {
        fprintf(stderr, "Failed to find __libc_dlopen_mode function\n");
        exit(EXIT_FAILURE);
    }

    // find the base address of libc in the target process
    remoteLib = findLibrary(info.path, pid);
    if (!remoteLib) {
        fprintf(stderr, "Failed to find libc in target process\n");
        exit(EXIT_FAILURE);
    }

    // calculate the address of dlopen in the target process 
    dlopenAddr = remoteLib + (dlopenAddr - localLib);

    // inject libscope.so into the target process
    inject(pid, (uint64_t) dlopenAddr, path);
}

