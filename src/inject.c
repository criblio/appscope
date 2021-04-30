
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
#include <stdlib.h>
#include <sys/mman.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <dlfcn.h>
#include "dbg.h"
#include "inject.h"
#include "fn.h"
#include "scopeelf.h"

#define __RTLD_DLOPEN	0x80000000

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
injectme(void) 
{
    asm(
        "andq $0xfffffffffffffff0, %rsp \n" //align stack to 16-byte boundary
        "mov %rax, %r9 \n"
        "xor %rax, %rax \n"
        "callq *%r9 \n"
        "int $3 \n"
    );
}

static void 
inject(pid_t pid, uint64_t dlopenAddr, char *path) 
{
    struct user_regs_struct oldregs, regs;
    unsigned char *oldcode;
    uint64_t freeaddr;
    int status;

    ptraceAttach(pid);

    // save registers
    ptrace(PTRACE_GETREGS, pid, NULL, &oldregs);
    memcpy(&regs, &oldregs, sizeof(struct user_regs_struct));

    int oldcodeSize = 256;
    oldcode = (unsigned char *)malloc(oldcodeSize);

    // find free space
    freeaddr = freeSpaceAddr(pid);
    printf("free space = 0x%lx\n", freeaddr);

    //back up the code
    ptraceRead(pid, freeaddr, oldcode, oldcodeSize);

    // Write our new stub
    ptraceWrite(pid, (uint64_t)freeaddr, path, strlen(path)+1);
    //ptraceWrite(pid, (uint64_t)freeaddr, "/tmp/libscope.so\x0", 32);
    ptraceWrite(pid, (uint64_t)freeaddr + 64, (&injectme) + 4, 64); //leave 64bytes lib path, also +4 to skip prologue

    // Update RIP to point to our code
    regs.rip = freeaddr + 64;
    regs.rax = dlopenAddr;
    regs.rdi = freeaddr;  //dlopen's first arg
    regs.rsi = RTLD_NOW | __RTLD_DLOPEN; //dlopen's second arg
    //regs.rsi = RTLD_LAZY; 

    ptrace(PTRACE_SETREGS, pid, NULL, &regs);

    // Continue execution
    ptrace(PTRACE_CONT, pid, NULL, NULL);
    waitpid(pid, &status, WUNTRACED);

    // Ensure that we are returned because of our int 0x3 trap
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {

        // Get process registers, indicating if the injection suceeded
        ptrace(PTRACE_GETREGS, pid, NULL, &regs);
        if (regs.rax != 0x0) {
            printf("Scope library injected at address %p\n", (void*)regs.rax);
        } else {
            fprintf(stderr, "Scope library could not be injected\n");
        }

        //restore the app's state
        ptraceWrite(pid, (uint64_t)freeaddr, oldcode, oldcodeSize);
        ptrace(PTRACE_SETREGS, pid, NULL, &oldregs);
        ptrace(PTRACE_DETACH, pid, NULL, NULL);

    } else {
        fprintf(stderr, "Fatal Error: Process stopped for unknown reason\n");
        exit(EXIT_FAILURE);
    }
}


typedef struct {
    char *path;
    uint64_t addr;
} libdl_info_t;

static int 
findLib(struct dl_phdr_info *info, size_t size, void *data)
{
    //if (strstr(info->dlpi_name, "libdl.so") != NULL) {
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

int 
injectScope(int pid, char* path) 
{
    uint64_t remoteLib, localLib;
    void *dlopenAddr = NULL;
    libdl_info_t info;
   
    if (!dl_iterate_phdr(findLib, &info)) {
        fprintf(stderr, "Failed to find libc\n");
        return -1;
    }
 
    localLib = info.addr;
    dlopenAddr = dlsym(RTLD_DEFAULT, "__libc_dlopen_mode");
    //dlopenAddr = dlsym(RTLD_DEFAULT, "dlopen");
    if (dlopenAddr == NULL) {
        fprintf(stderr, "Failed to find __libc_dlopen_mode function\n");
        return -1;
    }

    printf("libc found %s: 0x%lx, %p\n", info.path, info.addr, dlopenAddr);

    // Find the base address of libc in the target process
    remoteLib = findLibrary(info.path, pid);
    if (!remoteLib) {
        fprintf(stderr, "Failed to find libc in target process\n");
        return -1;
    }

    printf("remote libc = %p\n", (void*)remoteLib);
    
    // Due to ASLR, we need to calculate the address in the target process 
    dlopenAddr = remoteLib + (dlopenAddr - localLib);
    
    // Inject libscope.so into the target process
    inject(pid, (uint64_t) dlopenAddr, path);
    
    return 0;
}

