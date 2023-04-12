#define _GNU_SOURCE

#include <errno.h>
#include <dlfcn.h>
#include <link.h>
#include <linux/limits.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <unistd.h>

#include "inject.h"
#include "loaderutils.h"
#include "scopetypes.h"

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
    #define CALL_FUNC "callq *%rax \n"
    #define FIRST_ARG_REG regs.rdi
    #define SECOND_ARG_REG regs.rsi
    #define RET_REG regs.rax
    #define DBG_TRAP "int $3 \n"
#elif defined(__aarch64__)
    #define IP_REG regs.pc
    #define FUNC_REG regs.regs[2]
    #define CALL_FUNC "blr x2 \n"
    #define FIRST_ARG_REG regs.regs[0]
    #define SECOND_ARG_REG regs.regs[1]
    #define RET_REG regs.regs[0]
    #define DBG_TRAP "brk #0 \n"
#elif defined(__riscv) && __riscv_xlen == 64
    #include <asm/ptrace.h>
    #define IP_REG regs.pc
    #define FUNC_REG regs.t1
    #define CALL_FUNC "jalr t1 \n"
    #define FIRST_ARG_REG regs.a0
    #define SECOND_ARG_REG regs.a1
    #define RET_REG regs.a0
    #define DBG_TRAP "ebreak \n"
#else
   #error Bad arch defined
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

    sprintf(filename, "/proc/%d/maps", pid);
    if ((fd = fopen(filename, "r")) == NULL) {
        perror("fopen(/proc/PID/maps) failed");
        return EXIT_FAILURE;
    }

    while(fgets(line, 850, fd) != NULL) {
        sscanf(line, "%lx-%lx %s %*s %s %*d", addr_begin, &addr_end, perms, str);
        if ((strstr(perms, "x") != NULL) &&
            (strstr(perms, "w") == NULL)) {
            break;
        }
    }
    *free_size = addr_end - *addr_begin;

    fclose(fd);
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
        word = ptrace(PTRACE_PEEKTEXT, pid, (void*)(addr + numRead), NULL);
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
        if (ptrace(PTRACE_POKETEXT, pid, (void*)(addr + i), (void*)word) == -1) {
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
        //perror("ptrace(PTRACE_ATTACH) failed");
        return EXIT_FAILURE;
    }

    if(waitpid(target, &waitpidstatus, WUNTRACED) != target) {
        perror("waitpid failed");
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
        CALL_FUNC
        DBG_TRAP
    );
#elif defined(__aarch64__)
    __asm__ volatile(
        CALL_FUNC
        DBG_TRAP
    );
#elif defined(__riscv) && __riscv_xlen == 64
    __asm__ volatile(
        CALL_FUNC
        DBG_TRAP
    );
#else
   #error Bad arch defined
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

    if (cmd == REM_CMD_DLOPEN) {
        if (!path) return ret;

        libpathLen = strlen(path) + 1;
        if (libpathLen > SCOPE_PATH_SIZE) {
            fprintf(stderr, "library path %s is longer than %d, library could not be injected\n", path, SCOPE_PATH_SIZE);
            goto exit;
        }
        strncpy(libpath, path, libpathLen);
    }

    if (ptraceAttach(pid)) {
        goto exit;
    }

    // save registers
    my_iovec.iov_base = &oldregs;
    if (ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, &my_iovec) == -1) {
        fprintf(stderr, "error: ptrace get register(), library could not be injected\n");
        goto detach;
    }
    memcpy(&regs, &oldregs, sizeof(struct user_regs_struct));

    // find free space in text section
    if (freeSpaceAddr(pid, &freeAddr, &freeAddrSize)) {
        goto detach;
    }

    // sanity check for size condition
    if (freeAddrSize < INJECTED_CODE_SIZE_LEN) {
        fprintf(stderr, "Insufficient space in  0x%" PRIx64 " to inject, library could not be injected\n", freeAddr);
        goto detach;
    }
    
    // back up the code
    oldcode = (unsigned char *)malloc(INJECTED_CODE_SIZE_LEN);
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
        fprintf(stderr, "error: %s: unrecognized remote command: %d\n", __FUNCTION__, cmd);
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
    if (ptrace(PTRACE_SETREGSET, pid, (void *)NT_PRSTATUS, &my_iovec) == -1) {
        fprintf(stderr, "error: ptrace set register(), library could not be injected\n");
        goto restore_app;
    }

    // continue execution and wait until the target process is stopped
    if (ptrace(PTRACE_CONT, pid, NULL, NULL) == -1) {
        fprintf(stderr, "error: ptrace continue(), library could not be injected\n");
        goto restore_app;
    }
    waitpid(pid, &status, WUNTRACED);

    // if process has been stopped by SIGSTOP send SIGCONT signal along with PTRACE_CONT call
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP) {
        if (ptrace(PTRACE_CONT, pid, (void*)SIGCONT, NULL) == -1) {
            fprintf(stderr, "error: ptrace continue(), library could not be injected\n");
            goto restore_app;
        }
        waitpid(pid, &status, WUNTRACED);
    }

    // make sure the target process was stoppend by SIGTRAP triggered by DBG_TRAP
    if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {

        my_iovec.iov_base = &regs;
        // check if the library has been successfully injected
        if (ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, &my_iovec) == -1) {
            fprintf(stderr, "error: ptrace get register(), library could not be injected\n");
            goto restore_app;
        }

        if (RET_REG != 0) {
            ret = EXIT_SUCCESS;
            //printf("Appscope library injected at %p\n", (void*)RET_REG);
        } else {
            fprintf(stderr, "error: %s: remote function failed\n", __FUNCTION__);
        }

    } else {
        fprintf(stderr, "error: target process stopped with signal %d\n", WSTOPSIG(status));
    }

restore_app:
    //restore the app's state
    ptraceWrite(pid, freeAddr, oldcode, INJECTED_CODE_SIZE_LEN);
    my_iovec.iov_base = &oldregs;
    if (ptrace(PTRACE_SETREGSET, pid, (void *)NT_PRSTATUS, &my_iovec) == -1) {
        fprintf(stderr, "error: ptrace set register(), error during restore the application state\n");
    }
detach:
    ptrace(PTRACE_DETACH, pid, NULL, NULL);

exit:
    free(oldcode);
    return ret;
}

int 
injectScope(int pid, char *path)
{
    int glibc = FALSE;
    void *dlopenAddr = NULL;
    uint64_t remLib, remAddr;
    elf_buf_t *ebuf;
    char remproc[PATH_MAX];

    if (((remLib = findLibrary("libc.so", pid, FALSE, remproc, sizeof(remproc))) == 0) &&
        ((remLib = findLibrary("ld-musl", pid, FALSE, remproc, sizeof(remproc))) == 0) &&
        ((remLib = findLibrary("libc-", pid, FALSE, remproc, sizeof(remproc))) == 0)) {
        fprintf(stderr, "error: can't resolve libc.so in the remote process");
        return EXIT_FAILURE;
    }

    ebuf = getElf(remproc);
    if (((dlopenAddr = getDynSymbol(ebuf->buf, "dlopen")) == NULL) &&
        ((dlopenAddr = getDynSymbol(ebuf->buf, "__libc_dlopen_mode")) == NULL)) {
        return EXIT_FAILURE;
    }

    remAddr = remLib + (uint64_t)dlopenAddr;

    //printf("dlopen %p remLib 0x%lx remAddr 0x%lx libscope %s\n", dlopenAddr, remLib, remAddr, path);

    // inject libscope.so into the target process
    return inject(pid, REM_CMD_DLOPEN, (uint64_t)remAddr, path, glibc);
}

int
injectFirstAttach(int pid, uint64_t remaddr)
{
    // execute the attach command in the target process
    return inject(pid, REM_CMD_REATTACH, remaddr, NULL, 0);
}
