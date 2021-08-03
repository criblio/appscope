
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

/*
 * It turns out that PTRACE_GETREGS & PTRACE_SETREGS are arch specific.
 * From the man page: PTRACE_GETREGS and PTRACE_GETFPREGS are not present on all architectures.
 * We will need to use PTRACE_GETREGSET and PTRACE_SETREGSET as these are defined to read
 * registers in an architecture-dependent way.
 * TODO: the code needs to be updated to handle this when we apply ARM64 specifics.
 */
#ifdef __aarch64__
#define PTRACE_GETREGS PTRACE_GETREGSET
#define PTRACE_SETREGS PTRACE_SETREGSET
#endif

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
        perror("fopen(/proc/PID/maps) failed");
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
        perror("fopen(/proc/PID/maps) failed");
        return 0;
    }

    while(fgets(line, 850, fd) != NULL) {
        sscanf(line, "%lx-%*x %s %*s %s %*d", &addr, perms, str);
        if (strstr(perms, "x") != NULL) break;
    }

    fclose(fd);
    return addr;
}

static int 
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
            return EXIT_FAILURE;
        }
        numRead += sizeof(word);
        ptr[i++] = word;
    }

    return EXIT_SUCCESS;
}

static int 
ptraceWrite(int pid, uint64_t addr, void *data, int len) 
{
    long word = 0;
    int i = 0;

    for(i=0; i < len; i += sizeof(word), word=0) {
        memcpy(&word, data + i, sizeof(word));
        if (ptrace(PTRACE_POKETEXT, pid, addr + i, word) == -1) {
            perror("ptrace(PTRACE_POKETEXT) failed");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

static int
ptraceAttach(pid_t target) {
    int waitpidstatus;

    if(ptrace(PTRACE_ATTACH, target, NULL, NULL) == -1) {
        perror("ptrace(PTRACE_ATTACH) failed");
        return EXIT_FAILURE;
    }

    if(waitpid(target, &waitpidstatus, WUNTRACED) != target) {
        perror("waitpid failed");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static void 
call_dlopen(void) 
{
#ifdef __x86_64__
    asm(
        "andq $0xfffffffffffffff0, %rsp \n" //align stack to 16-byte boundary
        "callq *%rax \n"
        "int $3 \n"
    );
#endif
}

static void call_dlopen_end() {}

static int 
inject(pid_t pid, uint64_t dlopenAddr, char *path, int glibc)
{
    struct user_regs_struct oldregs, regs;
    unsigned char *oldcode;
    int status;
    uint64_t freeAddr, codeAddr;
    int libpathLen;
    ptrdiff_t oldcodeSize;

    if (ptraceAttach(pid)) {
        return EXIT_FAILURE;
    }

    // save registers
    ptrace(PTRACE_GETREGS, pid, NULL, &oldregs);
    memcpy(&regs, &oldregs, sizeof(struct user_regs_struct));

    // find free space
    freeAddr = freeSpaceAddr(pid);
    if (!freeAddr) {
        return EXIT_FAILURE;
    }
    
    // back up the code
    libpathLen = strlen(path) + 1;
    oldcodeSize = (call_dlopen_end - call_dlopen) + libpathLen;
    oldcode = (unsigned char *)malloc(oldcodeSize);
    if (ptraceRead(pid, freeAddr, oldcode, oldcodeSize)) {
        return EXIT_FAILURE;
    }

    // write the path to the library 
    if (ptraceWrite(pid, freeAddr, path, libpathLen)) {
        return EXIT_FAILURE;
    }

    // inject the code right after the library path
    codeAddr = freeAddr + libpathLen + 1;
    if (ptraceWrite(pid, codeAddr, &call_dlopen, call_dlopen_end - call_dlopen)) {
        return EXIT_FAILURE;
    }
#ifdef __x86_64__
    // set RIP to point to the injected code
    regs.rip = codeAddr;
    regs.rax = dlopenAddr;               // address of dlopen
    regs.rdi = freeAddr;                 // dlopen's first arg - path to the library

    if (glibc == TRUE) {
         // GNU ld.so uses a custom flag
        regs.rsi = RTLD_NOW | __RTLD_DLOPEN;
    } else {
        regs.rsi = RTLD_NOW;
    }
#endif
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
#ifdef __x86_64__
        if (regs.rax != 0x0) {
            //printf("Appscope library injected at %p\n", (void*)regs.rax);
        } else {
            fprintf(stderr, "error: dlopen() failed, library could not be injected\n");
        }
#endif
        //restore the app's state
        ptraceWrite(pid, freeAddr, oldcode, oldcodeSize);
        ptrace(PTRACE_SETREGS, pid, NULL, &oldregs);
        ptrace(PTRACE_DETACH, pid, NULL, NULL);

    } else {
        fprintf(stderr, "error: target process stopped\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int 
findLib(struct dl_phdr_info *info, size_t size, void *data)
{
    if (strstr(info->dlpi_name, "libc.so") != NULL ||
        strstr(info->dlpi_name, "ld-musl") != NULL) {
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
    int glibc = TRUE;
   
    if (!dl_iterate_phdr(findLib, &info)) {
        fprintf(stderr, "error: failed to find local libc\n");
        return EXIT_FAILURE;
    }
 
    localLib = info.addr;
    dlopenAddr = dlsym(RTLD_DEFAULT, "__libc_dlopen_mode");
    if (dlopenAddr == NULL) {
        dlopenAddr = dlsym(RTLD_DEFAULT, "dlopen");
        glibc = FALSE;
    }

    if (dlopenAddr == NULL) {
        fprintf(stderr, "error: failed to find dlopen()\n");
        return EXIT_FAILURE;
    }

    // find the base address of libc in the target process
    remoteLib = findLibrary(info.path, pid);
    if (!remoteLib) {
        fprintf(stderr, "error: failed to find libc in target process\n");
        return EXIT_FAILURE;
    }

    // calculate the address of dlopen in the target process 
    dlopenAddr = remoteLib + (dlopenAddr - localLib);

    // inject libscope.so into the target process
    return inject(pid, (uint64_t) dlopenAddr, path, glibc);
}

