/*
 * Load and run static executables
 *
 * objcopy -I binary -O elf64-x86-64 -B i386 ./lib/linux/libscope.so ./lib/linux/libscope.o
 * gcc -Wall -g src/scope.c -ldl -lrt -o scope ./lib/linux/libscope.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/mman.h>
#include <elf.h>
#include <stddef.h>
#include <sys/wait.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <limits.h>
#include <errno.h>
#include <getopt.h>

#include "scopestdlib.h"
#include "fn.h"
#include "dbg.h"
#include "scopeelf.h"
#include "scopetypes.h"
#include "utils.h"
#include "inject.h"
#include "os.h"
#include "gocontext.h"

#define UNKNOWN_GO_VER (-1)
#define MIN_SUPPORTED_GO_HTTP2 (17)
#define GO_ENV_VAR "GODEBUG"
#define GO_ENV_SERVER_VALUE "http2server"
#define GO_ENV_CLIENT_VALUE "http2client"
static int g_go_major_ver;
static char *go_ver;
static char g_go_build_ver[7];

// If possible, we want to set GODEBUG=http2server=0,http2client=0
// This tells go not to upgrade to http2, which allows
// our http1 protocol capture stuff to do it's thing.
// We consider this temporary, because when we support http2
// it will not be necessary.
static void
setGoHttpEnvVariable(void)
{
    if (checkEnv("SCOPE_GO_HTTP1", "false") == TRUE) return;

    char *cur_val = getenv(GO_ENV_VAR);

    // If GODEBUG isn't set, try to set it to http2server=0,http2client=0
    if (!cur_val) {
        if (setenv(GO_ENV_VAR, GO_ENV_SERVER_VALUE "=0," GO_ENV_CLIENT_VALUE "=0", 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        return;
    }

    // GODEBUG is set.
    // If http2server wasn't specified, let's append ",http2server=0"
    if (!scope_strstr(cur_val, GO_ENV_SERVER_VALUE)) {
        char *new_val = NULL;
        if ((scope_asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_SERVER_VALUE) == -1)) {
            scope_perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) scope_free(new_val);
    }

    cur_val = getenv(GO_ENV_VAR);

    // If http2client wasn't specified, let's append ",http2client=0"
    if (!scope_strstr(cur_val, GO_ENV_CLIENT_VALUE)) {
        char *new_val = NULL;
        if ((scope_asprintf(&new_val, "%s,%s=0", cur_val, GO_ENV_CLIENT_VALUE) == -1)) {
            scope_perror("setGoHttpEnvVariable:asprintf");
            return;
        }
        if (setenv(GO_ENV_VAR, new_val, 1)) {
            perror("setGoHttpEnvVariable:setenv");
        }
        if (new_val) scope_free(new_val);
    }
}


// Use go_str() whenever a "go string" type needs to be interpreted.
// The resulting go_str will need to be passed to free_go_str() when it is no
// longer needed.
// Don't use go_str() for byte arrays.
static char *
go_str(void *go_str)
{
    // Go 17 and higher use "c style" null terminated strings instead of a string and a length
    if (g_go_major_ver >= 17) {
       // We need to deference the address first before casting to a char *
       if (!go_str) return NULL;
       return (char *)*(uint64_t *)go_str;
    }

    gostring_t* go_str_tmp = (gostring_t *)go_str;
    if (!go_str_tmp || go_str_tmp->len <= 0) return NULL;

    char *c_str;
    if ((c_str = scope_calloc(1, go_str_tmp->len+1)) == NULL) return NULL;
    scope_memmove(c_str, go_str_tmp->str, go_str_tmp->len);
    c_str[go_str_tmp->len] = '\0';

    return c_str;
}

static void
free_go_str(char *str) {
    // Go 17 and higher use "c style" null terminated strings instead of a string and a length
    if (g_go_major_ver >= 17) {
        return;
    }
    if(str) scope_free(str);
}

static void *
getGoVersionAddr(const char* buf)
{
    int i;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name;
    const char *sec_data;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;
    const char magic[0xe] = "\xff Go buildinf:";
    void *go_build_ver_addr = NULL;
 
    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        sec_data = (const char *)buf + sections[i].sh_offset;
        // Since go1.13, the .go.buildinfo section has been added to
        // identify where runtime.buildVersion exists, for the case where
        // go apps have been stripped of their symbols.

        // offset into sec_data     field contents
        // -----------------------------------------------------------
        // 0x0                      build info magic = "\xff Go buildinf:"
        // 0xe                      binary ptrSize
        // 0xf                      endianness
        // 0x10                     pointer to string runtime.buildVersion
        // 0x10 + ptrSize           pointer to runtime.modinfo
        // 0x10 + 2 * ptr size      pointer to build flags

        if (!scope_strcmp(sec_name, ".go.buildinfo") &&
            (sections[i].sh_size >= 0x18) &&
            (!scope_memcmp(&sec_data[0], magic, sizeof(magic))) &&
            (sec_data[0xe] == 0x08)) {  // 64 bit executables only

            // debug/buildinfo/buildinfo.go
            // If the endianness has the 2 bit set, then the pointers are zero
            // and the 32-byte header is followed by varint-prefixed string data
            // for the two string values we care about.
            if (sec_data[0xf] == 0x00) {  // little-endian
                uint64_t *addressPtr = (uint64_t*)&sec_data[0x10];
                go_build_ver_addr = (void*)*addressPtr;
            } else if (sec_data[0xf] == 0x02) {
                scope_memmove(g_go_build_ver, (char*)&sec_data[0x21], 6);
                g_go_build_ver[6] = '\0';
                go_build_ver_addr = &g_go_build_ver;
            }
        }
    }
    return go_build_ver_addr;
}


// Extract the Major version from a Go version string
static int
go_major_version(const char *go_runtime_version)
{
    if (!go_runtime_version) return UNKNOWN_GO_VER;

    char buf[256] = {0};
    scope_strncpy(buf, go_runtime_version, sizeof(buf)-1);

    char *token = scope_strtok(buf, ".");
    token = scope_strtok(NULL, ".");
    if (!token) {
        return UNKNOWN_GO_VER;
    }

    scope_errno = 0;
    long val = scope_strtol(token, NULL, 10);
    if (scope_errno || val <= 0 || val > INT_MAX) {
        return UNKNOWN_GO_VER;
    }

    return val;
}

int
getBaseAddress(uint64_t *addr) {
    uint64_t base_addr = 0;
    char perms[5];
    char offset[20];
    char buf[1024];
    char pname[1024];
    FILE *fp;

    if (osGetProcname(pname, sizeof(pname)) == -1) return -1;

    if ((fp = scope_fopen("/proc/self/maps", "r")) == NULL) {
        return -1;
    }

    while (scope_fgets(buf, sizeof(buf), fp) != NULL) {
        uint64_t addr_start;
        scope_sscanf(buf, "%lx-%*x %s %*s %s %*d", &addr_start, perms, offset);
        if (scope_strstr(buf, pname) != NULL) {
            base_addr = addr_start;
            break;
        }
    }

    scope_fclose(fp);
    if (base_addr) {
        *addr = base_addr;
        return 0;
    }
    return -1;
}

// Detect the Go Version of an executable
static void
getGoVersion(elf_buf_t *ebuf) {
    //check ELF type
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)ebuf->buf;
    // if it's a position independent executable, get the base address from /proc/self/maps
    uint64_t base = 0LL;
    // A dynamic executable has a type ET_EXEC. A PIE mode executable has a type ET_DYN.
    /*
    if (ehdr->e_type == ET_DYN && (scopeGetGoAppStateStatic() == FALSE)) {
        if (getBaseAddress(&base) != 0) {
//            sysprint("ERROR: can't get the base address\n");
            return; // don't install our hooks
        }
        Elf64_Shdr* textSec = getElfSection(ebuf->buf, ".text");
//        sysprint("base %lx %lx %lx\n", base, (uint64_t)ebuf->text_addr, textSec->sh_offset);
        base = base - (uint64_t)ebuf->text_addr + textSec->sh_offset;
    }
    */

//        Elf64_Shdr* textSec = getElfSection(ebuf->buf, ".text");
        Elf64_Shdr* dataSec = getElfSection(ebuf->buf, ".data");
//        Elf64_Shdr* rodataSec = getElfSection(ebuf->buf, ".rodata");


    uint64_t go_ver_str_sym = (uint64_t)getSymbol(ebuf->buf, "runtime.buildVersion.str");
    if (go_ver_str_sym) {
        char *go_ver_sym_1 = (char *)((uint64_t)ebuf->buf + (go_ver_str_sym - (uint64_t)dataSec->sh_addr) + (uint64_t)dataSec->sh_offset);
        go_ver = (char *)(go_ver_sym_1 + base);
    } else {
        void *go_ver_sym = getSymbol(ebuf->buf, "runtime.buildVersion");
        if (go_ver_sym) {
            // Start of file + Offset into rodata + Offset of rodata in the file
            uint64_t *go_ver_sym_1 = (uint64_t *)((uint64_t)ebuf->buf + (go_ver_sym - (uint64_t)dataSec->sh_addr) + (uint64_t)dataSec->sh_offset);

            uint64_t go_ver_sym_2 = ((uint64_t )ebuf->buf + (*go_ver_sym_1 - (uint64_t)dataSec->sh_addr) + (uint64_t)dataSec->sh_offset);

            gostring_t go_str_tmp = {
               .len = (int)*((char *)go_ver_sym_1 + 0x8),
               .str = (char *)go_ver_sym_2,
            };

            char *c_str;
            if ((c_str = scope_calloc(1, go_str_tmp.len+1)) == NULL) return;
            scope_memmove(c_str, go_str_tmp.str, go_str_tmp.len);
            c_str[go_str_tmp.len] = '\0';

            go_ver = c_str;
        }
    }

    if (!go_ver) {
        // runtime.buildVersion symbol not found, probably dealing with a stripped binary
        // try to retrieve the version symbol address from the .go.buildinfo section
        // if g_go_build_ver is set we know we're dealing with a char *
        // if it is not set, we know we're dealing with a "go string"
        void *ver_addr = getGoVersionAddr(ebuf->buf);
        if (g_go_build_ver[0] != '\0') {
            go_ver = (char *)((uint64_t)ver_addr);
        } else {
            go_ver = go_str((void *)((uint64_t)ver_addr + base));
        }
    }
    
    if (!go_ver) {
        scope_fprintf(scope_stderr, "error: could not get go version. Note: We do not support \
                static stripped executables built with versions of Go prior to 1.13\n");
        return;
    }
    
    printf("go_runtime_version = %s\n", go_ver);
    g_go_major_ver = go_major_version(go_ver);
}

static void
showUsage(char *prog)
{
    scope_printf(
      "\n"
      "Cribl AppScope Dynamic Loader %s\n"
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n"
      "usage: %s [OPTIONS] --lib LIBRARY [--] EXECUTABLE [ARGS...]\n"
      "       %s [OPTIONS] --attach PID\n"
      "\n"
      "options:\n"
      "  -u, -h, --usage, --help  display this info\n"
      "  -a, --attach PID         attach to the specified process ID\n"
      "\n"
      "Unless you are an AppScope developer, you are likely in the wrong place.\n"
      "See `scope` or `ldscope` instead.\n"
      "\n"
      "User docs are at https://appscope.dev/docs/. The project is hosted at\n"
      "https://github.com/criblio/appscope. Please direct feature requests and\n"
      "defect reports there.\n"
      "\n",
      SCOPE_VER, prog, prog
    );
}

// long aliases for short options
static struct option options[] = {
    {"help",    no_argument,       0, 'h'},
    {"usage",   no_argument,       0, 'u'},
    {"attach",  required_argument, 0, 'a'},
    {0, 0, 0, 0}
};

int
main(int argc, char **argv, char **env)
{
    // process command line
    char *attachArg = 0;
    for (;;) {
        int index = 0;
        int opt = getopt_long(argc, argv, "+:uha:", options, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'u':
            case 'h':
                showUsage(scope_basename(argv[0]));
                return EXIT_SUCCESS;
            case 'a':
                attachArg = optarg;
                break;
            case ':':
                // options missing their value end up here
                switch (optopt) {
                    default:
                        scope_fprintf(scope_stderr, "error: missing value for -%c option\n", optopt);
                        showUsage(scope_basename(argv[0]));
                        return EXIT_FAILURE;
                }
                break;
            case '?':
            default:
                scope_fprintf(scope_stderr, "error: invalid option: -%c\n", optopt);
                showUsage(scope_basename(argv[0]));
                return EXIT_FAILURE;
        }
    }

    // either --attach or an executable is required
    if (!attachArg && optind >= argc) {
        scope_fprintf(scope_stderr, "error: missing --attach or EXECUTABLE argument\n");
        showUsage(scope_basename(argv[0]));
        return EXIT_FAILURE;
    }

    // use --attach, ignore executable and args
    if (attachArg && optind < argc) {
        scope_fprintf(scope_stderr, "warning: ignoring EXECUTABLE argument with --attach option\n");
    }

    // SCOPE_LIB_PATH environment variable is required
    char* scopeLibPath = getenv("SCOPE_LIB_PATH");
    if (!scopeLibPath) {
        scope_fprintf(scope_stderr, "error: SCOPE_LIB_PATH must be set to point to libscope.so\n");
        return EXIT_FAILURE;
    }
    if (scope_access(scopeLibPath, R_OK|X_OK)) {
        scope_fprintf(scope_stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        return EXIT_FAILURE;
    }

    elf_buf_t *ebuf;
    int (*sys_exec)(elf_buf_t *, const char *, int, char **, char **);
    pid_t pid;
    void *handle = NULL;

    // Use dlsym to get addresses for everything in g_fn
    initFn();
    setPidEnv(scope_getpid());

    if (attachArg) {
        char *exe_path = NULL;
        elf_buf_t *ebuf;

        int pid = scope_atoi(attachArg);
        if (pid < 1) {
            scope_fprintf(scope_stderr, "error: invalid PID for --attach\n");
            return EXIT_FAILURE;
        }

        if (osGetExePath(pid, &exe_path) == -1) {
            scope_fprintf(scope_stderr, "error: can't get path to executable for pid %d\n", pid);
            return EXIT_FAILURE;
        }

        if ((ebuf = getElf(exe_path)) == NULL) {
            scope_free(exe_path);
            scope_fprintf(scope_stderr, "error: can't read the executable %s\n", exe_path);
            return EXIT_FAILURE;
        }

        if (is_static(ebuf->buf) == TRUE) {
            scope_fprintf(scope_stderr, "error: can't attach to the static executable: %s\nNote that the executable can be 'scoped' using the command 'scope run -- %s'\n", exe_path, exe_path);
            scope_free(exe_path);
            freeElf(ebuf->buf, ebuf->len);
            return EXIT_FAILURE;
        }

        scope_free(exe_path);
        freeElf(ebuf->buf, ebuf->len);

        scope_printf("Attaching to process %d\n", pid);
        int ret = injectScope(pid, scopeLibPath);

        // remove the config that `ldscope`
        char path[PATH_MAX];
        scope_snprintf(path, sizeof(path), "/scope_attach_%d.env", pid);
        scope_shm_unlink(path);

        // done
        return ret;
    }

    char *inferior_command = getpath(argv[optind]);
    if (!inferior_command) {
        scope_fprintf(scope_stderr,"%s could not find or execute command `%s`.  Exiting.\n", argv[0], argv[optind]);
        exit(EXIT_FAILURE);
    }

    ebuf = getElf(inferior_command);

    if (ebuf && (is_go(ebuf->buf) == TRUE)) {
        if (setenv("SCOPE_APP_TYPE", "go", 1) == -1) {
            perror("setenv");
            goto err;
        }

//        getGoVersion(ebuf); 
//        if (g_go_major_ver < MIN_SUPPORTED_GO_HTTP2) {
//            setGoHttpEnvVariable();
//        }

    } else {
        if (setenv("SCOPE_APP_TYPE", "native", 1) == -1) {
            perror("setenv");
            goto err;
        }
    }

    if ((ebuf == NULL) || (!is_static(ebuf->buf))) {
        // Dynamic executable path
        if (ebuf) freeElf(ebuf->buf, ebuf->len);

        if (setenv("LD_PRELOAD", scopeLibPath, 0) == -1) {
            perror("setenv");
            goto err;
        }

        if (setenv("SCOPE_EXEC_TYPE", "dynamic", 1) == -1) {
            perror("setenv");
            goto err;
        }
        
        pid = fork();
        if (pid == -1) {
            perror("fork");
            goto err;
        } else if (pid > 0) {
            int status;
            int ret;
            do {
                ret = waitpid(pid, &status, 0);
            } while (ret == -1 && errno == EINTR);

            if (WIFEXITED(status)) exit(WEXITSTATUS(status));
            exit(EXIT_FAILURE);
        } else {
            execve(inferior_command, &argv[optind], environ);
            perror("execve");
            goto err;
        }
    }

    if (setenv("SCOPE_EXEC_TYPE", "static", 1) == -1) {
        perror("setenv");
        goto err;
    }

    // Static executable path
    if (getenv("LD_PRELOAD") != NULL) {
        unsetenv("LD_PRELOAD");
        execve(argv[0], argv, environ);
    }

    program_invocation_short_name = scope_basename(argv[1]);

    if (!is_go(ebuf->buf)) {
        // We're getting here with upx-encoded binaries
        // and any other static native apps...
        // Start here when we support more static binaries
        // than go.
        execve(argv[optind], &argv[optind], environ);
    }

    if ((handle = dlopen(scopeLibPath, RTLD_LAZY)) == NULL) {
        goto err;
    }

    sys_exec = dlsym(handle, "sys_exec");
    if (!sys_exec) {
        goto err;
    }

    sys_exec(ebuf, inferior_command, argc-optind, &argv[optind], env);

    return 0;
err:
    if (ebuf) scope_free(ebuf);
    exit(EXIT_FAILURE);
}
