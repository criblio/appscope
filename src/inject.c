
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <unistd.h>
#include <string.h>
#include <link.h>
#include <errno.h>
#include <stdint.h>
#include <linux/limits.h>
#include <dlfcn.h>
#include <stddef.h>
#include <inttypes.h>
#include "scopestdlib.h"
#include "dbg.h"
#include "inject.h"
#include "os.h"

typedef enum {
    REM_CMD_DLOPEN,
    REM_CMD_REATTACH,
    REM_CMD_MAX
} remote_cmd_t;

#define __RTLD_DLOPEN	0x80000000

// Size of injected code segment
#define INJECTED_CODE_SIZE_LEN (512)
// Offset between injected path code and injected code segment
#define PATH_CODE_OFFSET (16)
// Maximum size of injected path
#define SCOPE_PATH_SIZE (256)

typedef struct {
    char *path;
    uint64_t addr;
} libdl_info_t;

#ifdef __x86_64__
    #define IP_REG regs.rip
    #define FUNC_REG regs.rax
    #define FIRST_ARG_REG regs.rdi
    #define SECOND_ARG_REG regs.rsi
    #define RET_REG regs.rax
    #define DBG_TRAP "int $3 \n"
#elif defined(__aarch64__)
    #define IP_REG regs.pc
    #define FUNC_REG regs.regs[2]
    #define FIRST_ARG_REG regs.regs[0]
    #define SECOND_ARG_REG regs.regs[1]
    #define RET_REG regs.regs[0]
    #define DBG_TRAP "brk #0 \n"
#endif

static int
freeSpaceAddr(pid_t pid, uint64_t *addr_begin, uint64_t *free_size)
{
    FILE *fd;
    char filename[PATH_MAX];
    char line[850];
    uint64_t addr_end;
    char str[20];
    char perms[5];

    scope_sprintf(filename, "/proc/%d/maps", pid);
    if ((fd = scope_fopen(filename, "r")) == NULL) {
        scope_perror("fopen(/proc/PID/maps) failed");
        return EXIT_FAILURE;
    }

    while(scope_fgets(line, 850, fd) != NULL) {
        scope_sscanf(line, "%lx-%lx %s %*s %s %*d", addr_begin, &addr_end, perms, str);
        if ((scope_strstr(perms, "x") != NULL) &&
            (scope_strstr(perms, "w") == NULL)) {
            break;
        }
    }
    *free_size = addr_end - *addr_begin;

    scope_fclose(fd);
    return EXIT_SUCCESS;
}

static int
ptraceRead(int pid, uint64_t addr, void *data, int len)
{
    int numRead = 0;
    int i = 0;
    long word = 0;
    long *ptr = (long *) data;

    while (numRead < len) {
        word = scope_ptrace(PTRACE_PEEKTEXT, pid, (void*)(addr + numRead), NULL);
        if(word == -1) {
            scope_perror("ptrace(PTRACE_PEEKTEXT) failed");
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
        scope_memcpy(&word, data + i, sizeof(word));
        if (scope_ptrace(PTRACE_POKETEXT, pid, (void*)(addr + i), (void*)word) == -1) {
            scope_perror("ptrace(PTRACE_POKETEXT) failed");
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

static int
ptraceAttach(pid_t target) {
    int waitpidstatus;

    if(scope_ptrace(PTRACE_ATTACH, target, NULL, NULL) == -1) {
        scope_perror("ptrace(PTRACE_ATTACH) failed");
        return EXIT_FAILURE;
    }

    if(scope_waitpid(target, &waitpidstatus, WUNTRACED) != target) {
        scope_perror("waitpid failed");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static void
call_remfunc(void)
{
#ifdef __x86_64__
    asm(
        "andq $0xfffffffffffffff0, %rsp \n" //align stack to 16-byte boundary
        "callq *%rax \n"
        DBG_TRAP
    );
#elif defined(__aarch64__)
    __asm__ volatile(
        "blr x2 \n"
        DBG_TRAP
    );
#endif
}

static void call_remfunc_end() {}

static int
inject(pid_t pid, remote_cmd_t cmd, uint64_t remAddr, char *path, int glibc)
{
    struct iovec my_iovec;
    struct user_regs_struct oldregs, regs;
    my_iovec.iov_len = sizeof(regs);
    unsigned char *oldcode = NULL;
    int status;
    uint64_t freeAddr, codeAddr, freeAddrSize;
    char libpath[SCOPE_PATH_SIZE] = {0};
    int libpathLen;
    int ret = EXIT_FAILURE;

    if ((cmd == REM_CMD_DLOPEN) && !path) return ret;

    if (cmd == REM_CMD_DLOPEN) {
        libpathLen = scope_strlen(path) + 1;
        if (libpathLen > SCOPE_PATH_SIZE) {
            scope_fprintf(scope_stderr, "library path %s is longer than %d, library could not be injected\n", path, SCOPE_PATH_SIZE);
            goto exit;
        }
        scope_strncpy(libpath, path, libpathLen);
    }

    if (ptraceAttach(pid)) {
        goto exit;
    }

    // save registers
    my_iovec.iov_base = &oldregs;
    if (scope_ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, &my_iovec) == -1) {
        scope_fprintf(scope_stderr, "error: ptrace get register(), library could not be injected\n");
        goto detach;
    }
    scope_memcpy(&regs, &oldregs, sizeof(struct user_regs_struct));

    // find free space in text section
    if (freeSpaceAddr(pid, &freeAddr, &freeAddrSize)) {
        goto detach;
    }

    // sanity check for size condition
    if (freeAddrSize < INJECTED_CODE_SIZE_LEN) {
        scope_fprintf(scope_stderr, "Insufficient space in  0x%" PRIx64 " to inject, library could not be injected\n", freeAddr);
        goto detach;
    }
    
    // back up the code
    oldcode = (unsigned char *)scope_malloc(INJECTED_CODE_SIZE_LEN);
    if (ptraceRead(pid, freeAddr, oldcode, INJECTED_CODE_SIZE_LEN)) {
        goto detach;
    }

    // write the path to the library
    if (cmd == REM_CMD_DLOPEN) {
        if (ptraceWrite(pid, freeAddr, &libpath, SCOPE_PATH_SIZE)) {
            goto restore_app;
        }

        // inject the code after offset the library path
        codeAddr = freeAddr + SCOPE_PATH_SIZE + PATH_CODE_OFFSET;
        if (ptraceWrite(pid, codeAddr, &call_remfunc, call_remfunc_end - call_remfunc)) {
            goto restore_app;
        }
    } else if (cmd == REM_CMD_REATTACH) {
        codeAddr = freeAddr;
        if (ptraceWrite(pid, codeAddr, &call_remfunc, call_remfunc_end - call_remfunc)) {
            goto restore_app;
        }
    } else {
        scope_fprintf(scope_stderr, "error: %s: unrecognized remote command\n", __FUNCTION__);
        goto detach;
    }

    // set instruction pointer to point to the injected code
    IP_REG = codeAddr;
    FUNC_REG = remAddr;                      // address of remote function to call

    if (cmd == REM_CMD_DLOPEN) {
        FIRST_ARG_REG = freeAddr;            // dlopen's first arg - path to the library
        SECOND_ARG_REG = RTLD_NOW;
        // GNU ld.so uses a custom flag
        if (glibc == TRUE) {
            SECOND_ARG_REG |= __RTLD_DLOPEN;
        }
    }

    my_iovec.iov_base = &regs;
    if (scope_ptrace(PTRACE_SETREGSET, pid, (void *)NT_PRSTATUS, &my_iovec) == -1) {
        scope_fprintf(scope_stderr, "error: ptrace set register(), library could not be injected\n");
        goto restore_app;
    }

    // continue execution and wait until the target process is stopped
    if (scope_ptrace(PTRACE_CONT, pid, NULL, NULL) == -1) {
        scope_fprintf(scope_stderr, "error: ptrace continue(), library could not be injected\n");
        goto restore_app;
    }
    scope_waitpid(pid, &status, WUNTRACED);

    // if process has been stopped by SIGSTOP send SIGCONT signal along with PTRACE_CONT call
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
        if (scope_ptrace(PTRACE_CONT, pid, (void*)SIGCONT, NULL) == -1) {
            scope_fprintf(scope_stderr, "error: ptrace continue(), library could not be injected\n");
            goto restore_app;
        }
        scope_waitpid(pid, &status, WUNTRACED);
    }

    // make sure the target process was stoppend by SIGTRAP triggered by DBG_TRAP
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {

        my_iovec.iov_base = &regs;
        // check if the library has been successfully injected
        if (scope_ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, &my_iovec) == -1) {
            scope_fprintf(scope_stderr, "error: ptrace get register(), library could not be injected\n");
            goto restore_app;
        }

        uint64_t success = (cmd == REM_CMD_REATTACH) ? TRUE : 0;
        if (RET_REG == success) {
            ret = EXIT_SUCCESS;
            //printf("Appscope library injected at %p\n", (void*)RET_REG);
        } else {
            scope_fprintf(scope_stderr, "error: %s: remote function failed\n", __FUNCTION__);
        }

    } else {
        scope_fprintf(scope_stderr, "error: target process stopped with signal %d\n", WSTOPSIG(status));
    }

restore_app:
    //restore the app's state
    ptraceWrite(pid, freeAddr, oldcode, INJECTED_CODE_SIZE_LEN);
    my_iovec.iov_base = &oldregs;
    if (scope_ptrace(PTRACE_SETREGSET, pid, (void *)NT_PRSTATUS, &my_iovec) == -1) {
        scope_fprintf(scope_stderr, "error: ptrace set register(), error during restore the application state\n");
    }
detach:
    scope_ptrace(PTRACE_DETACH, pid, NULL, NULL);

exit:
    scope_free(oldcode);
    return ret;
}

static int 
findLib(struct dl_phdr_info *info, size_t size, void *data)
{
    if (scope_strstr(info->dlpi_name, "libc.so") != NULL ||
        scope_strstr(info->dlpi_name, "ld-musl") != NULL) {
        char libpath[PATH_MAX];
        if (scope_realpath(info->dlpi_name, libpath)) {
            ((libdl_info_t *)data)->path = libpath;
            ((libdl_info_t *)data)->addr = info->dlpi_addr;
            return 1;
        }
    }
    return 0;
}

int 
injectScope(int pid, char *path)
{
    uint64_t remoteLib, localLib;
    void *dlopenAddr = NULL;
    libdl_info_t info;
    int glibc = TRUE;

    if (!dl_iterate_phdr(findLib, &info)) {
        scope_fprintf(scope_stderr, "error: failed to find local libc\n");
        return EXIT_FAILURE;
    }

    localLib = info.addr;
    dlopenAddr = dlsym(RTLD_DEFAULT, "__libc_dlopen_mode");
    if (dlopenAddr == NULL) {
        dlopenAddr = dlsym(RTLD_DEFAULT, "dlopen");
        glibc = FALSE;
    }

    if (dlopenAddr == NULL) {
        scope_fprintf(scope_stderr, "error: failed to find dlopen()\n");
        return EXIT_FAILURE;
    }

    // find the base address of libc in the target process
    remoteLib = osFindLibrary(info.path, pid);
    if ((remoteLib == 0) || (remoteLib == -1)) {
        scope_fprintf(scope_stderr, "error: failed to find libc in target process\n");
        return EXIT_FAILURE;
    }

    // calculate the address of dlopen in the target process
    dlopenAddr = remoteLib + (dlopenAddr - localLib);

    // inject libscope.so into the target process
    return inject(pid, REM_CMD_DLOPEN, (uint64_t) dlopenAddr, path, glibc);
}

int
injectReattach(int pid, uint64_t remaddr)
{
    // execute the attach command in the target process
    return inject(pid, REM_CMD_REATTACH, remaddr, NULL, 0);
}
