#define _GNU_SOURCE
#include "capstone/capstone.h"
#include <dlfcn.h>
#include <elf.h>
#include <malloc.h>

#include "bashmem.h"
#include "dbg.h"
#include "fn.h"
#include "os.h"
#include "scopetypes.h"
#include "utils.h"

typedef struct {
    void *(*malloc)(size_t size);
    void *(*realloc)(void *ptr, size_t size);
    void (*free)(void *ptr);
    void *(*memalign)(size_t alignment, size_t size);
    void (*cfree)(void *ptr);
    // Needed for our capstone disassembler
    void *(*calloc)(size_t nmemb, size_t size);
    int (*vsnprintf)(char *str, size_t size, const char *format, va_list ap);
} bash_mem_fn_t;

bash_mem_fn_t g_mem_fn;

// fn prototypes
static void *bash_internal_malloc(size_t, const char *, int, int);
static void *bash_internal_realloc(void *, size_t, const char *, int, int);
static void bash_internal_free(void *, const char *, int, int);
static void *bash_internal_memalign(size_t, size_t, const char *, int, int);
static void bash_internal_cfree(void *, const char *, int, int);

typedef struct {
    // constant at build time
    const char *name;
    const void *fn_ptr;

    // set at runtime
    void *external_addr;
    void *internal_addr;
} patch_info_t;

patch_info_t bash_mem_func[] = {
    {"malloc", bash_internal_malloc, NULL, NULL},     {"realloc", bash_internal_realloc, NULL, NULL}, {"free", bash_internal_free, NULL, NULL},
    {"memalign", bash_internal_memalign, NULL, NULL}, {"cfree", bash_internal_cfree, NULL, NULL},
};
const int bash_mem_func_count = sizeof(bash_mem_func) / sizeof(bash_mem_func[0]);

static int
glibcMemFuncsFound(void)
{
    g_mem_fn.malloc = dlsym(RTLD_NEXT, "malloc");
    g_mem_fn.realloc = dlsym(RTLD_NEXT, "realloc");
    g_mem_fn.free = dlsym(RTLD_NEXT, "free");
    g_mem_fn.memalign = dlsym(RTLD_NEXT, "memalign");
    g_mem_fn.cfree = dlsym(RTLD_NEXT, "cfree");
    // Needed for our capstone disassembler
    g_mem_fn.calloc = dlsym(RTLD_NEXT, "calloc");
    g_mem_fn.vsnprintf = dlsym(RTLD_NEXT, "vsnprintf");

    return g_mem_fn.malloc && g_mem_fn.realloc && g_mem_fn.free && g_mem_fn.memalign && g_mem_fn.cfree &&
        // Needed for our capstone disassembler
        g_mem_fn.calloc && g_mem_fn.vsnprintf;
}

int
func_found_in_executable(const char *symbol, const char *exe)
{
    int func_found = FALSE;
    char *exe_with_preceeding_slash = NULL;

    // open the executable (as opposed to a specific shared lib)
    void *exe_handle = g_fn.dlopen(NULL, RTLD_LAZY);
    if (!exe_handle)
        goto out;

    void *symbol_ptr = dlsym(exe_handle, symbol);
    if (!symbol_ptr)
        goto out;

    Dl_info symbol_info;
    if (!dladdr(symbol_ptr, &symbol_info)) {
        goto out;
    }

    // turns "bash" into "/bash", for example
    if (asprintf(&exe_with_preceeding_slash, "/%s", exe) == -1)
        goto out;
    func_found = endsWith(symbol_info.dli_fname, exe_with_preceeding_slash);

out:
    if (exe_handle)
        dlclose(exe_handle);
    if (exe_with_preceeding_slash)
        free(exe_with_preceeding_slash);
    return func_found;
}

static void *
bash_internal_malloc(size_t bytes, const char *file, int line, int flags)
{
    return g_mem_fn.malloc(bytes);
}

static void *
bash_internal_realloc(void *mem, size_t n, const char *file, int line, int flags)
{
    return g_mem_fn.realloc(mem, n);
}

static void
bash_internal_free(void *mem, const char *file, int line, int flags)
{
    g_mem_fn.free(mem);
}

static void *
bash_internal_memalign(size_t alignment, size_t size, const char *file, int line, int flags)
{
    return g_mem_fn.memalign(alignment, size);
}

// Defined here because it's widely deprecated
extern void cfree(void *__ptr);

static void
bash_internal_cfree(void *p, const char *file, int line, int flags)
{
    g_mem_fn.cfree(p);
}

static int
bashMemFuncsFound()
{
    int num_found = 0;
    csh disass_handle = 0;

    void *exe_handle = g_fn.dlopen(NULL, RTLD_LAZY);
    if (!exe_handle)
        goto out;

    cs_arch arch;
    cs_mode mode;
#if defined(__aarch64__)
    arch = CS_ARCH_ARM64;
    mode = CS_MODE_LITTLE_ENDIAN;
#elif defined(__x86_64__)
    arch = CS_ARCH_X86;
    mode = CS_MODE_64;
#else
    goto out;
#endif
    if (cs_open(arch, mode, &disass_handle) != CS_ERR_OK)
        goto out;

    int i;
    cs_insn *asm_inst = NULL;
    unsigned int asm_count = 0;
    for (i = 0; i < bash_mem_func_count; i++) {
        if (asm_inst) {
            cs_free(asm_inst, asm_count);
            asm_inst = NULL;
            asm_count = 0;
        }
        patch_info_t *func = &bash_mem_func[i];
        void *func_ptr = dlsym(exe_handle, func->name);
        if (!func_ptr) {
            scopeLogError("Couldn't find bash function %s", func->name);
            continue;
        }

        const int DECODE_BYTES = 50;
        asm_count = cs_disasm(disass_handle, func_ptr, DECODE_BYTES, (uint64_t)func_ptr, 0, &asm_inst);
        if (asm_count <= 0) {
            scopeLogError("Couldn't disassemble bash function %s", func->name);
            continue;
        }

        // look for the first jmp instruction
        int j;
        cs_insn *inst;
        for (j = 0; j < asm_count; j++) {
            inst = &asm_inst[j];
            if (!strcmp((const char *)inst->mnemonic, "jmp") && ((inst->size == 5) || (inst->size == 2))) {
                break;
            }
        }
        if (j == asm_count) {
            scopeLogError("For bash function %s, couldn't find "
                          "a jmp instruction in the first %d instructions from 0x%p",
                          func->name, asm_count, func_ptr);
            continue;
        }

        // Calculate the destination of the jmp instruction, and save it
        // as the internal_addr that we want to hook later.  Assumes x86_64.
        int64_t addr = inst->address; // the address of the current inst
        int64_t jmp_offset;
        switch (inst->size) {
            case 5:
                // (05) 0xe927f4ffff -> e9 (jmp) 0xfffff427
                jmp_offset = *(int *)(addr + 1);
                break;
            case 2:
                // (02) 0xebec       -> eb (jmp) 0xec
                jmp_offset = *(char *)(addr + 1);
                break;
            default:
                continue;
        }
        addr += jmp_offset; // the relative offset from inst
        addr += inst->size; // relative to the next instruction ptr

        // Save away what we found
        func->external_addr = func_ptr;
        func->internal_addr = (void *)addr;
        num_found++;
    }

    if (asm_inst) {
        cs_free(asm_inst, asm_count);
    }

out:

    if (exe_handle)
        dlclose(exe_handle);
    if (disass_handle)
        cs_close(&disass_handle);
    return num_found == bash_mem_func_count;
}

static int
replaceBashMemFuncs()
{
    int i, rc;
    int num_patched = 0;

    funchook_t *funchook = funchook_create();
    if (!funchook) {
        scopeLogError("funchook_create failed");
        return FALSE;
    }

    // Setting funchook_set_debug_file is not possible when patching memory.
    // If funchook has a debug file, then it does a fopen of this file which
    // calls malloc to create a memory buffer.  In this scenario, the memory
    // buffer is created with bash's memory subsystem, but then after the
    // patching is complete, the fclose of this file will attempt to free the
    // memory buffer with a different memory subsystem than created it.
    // No bueno.
    //
    // funchook_set_debug_file(DEFAULT_LOG_PATH);

    for (i = 0; i < bash_mem_func_count; i++) {
        patch_info_t *func = &bash_mem_func[i];
        void *addr_to_patch = func->internal_addr;
        rc = funchook_prepare(funchook, (void **)&addr_to_patch, (void *)func->fn_ptr);
        if (rc) {
            scopeLogError("funchook_prepare failed for %s at 0x%p", func->name, func->internal_addr);
        } else {
            num_patched++;
        }
    }

    rc = funchook_install(funchook, 0);
    if (rc) {
        scopeLogError("ERROR: failed to install run_bash_mem_fix. (%s)\n", funchook_error_message(funchook));
    }

    return !rc && (num_patched == bash_mem_func_count);
}

int
run_bash_mem_fix(void)
{
    int successful = FALSE;

    // fill in g_mem_fn by looking up glibc funcs
    if (!glibcMemFuncsFound())
        goto out;

    // Take charge of what memory functions capstone, our disassembler,
    // uses.  We can't use bash's memory functions and switch away from
    // them at the same time.  This CS_OPT_MEM call ensures that capstone
    // uses glibc's memory subsystem instead of bash's.  It's important
    // to make this call before before calls to bashMemFuncsFound() and
    // replaceBashMemFuncs(), since these use the disassembler.
    cs_opt_mem capstone_mem = {.malloc = g_mem_fn.malloc, .calloc = g_mem_fn.calloc, .realloc = g_mem_fn.realloc, .free = g_mem_fn.free, .vsnprintf = g_mem_fn.vsnprintf};
    if ((cs_option(0, CS_OPT_MEM, (size_t)&capstone_mem)) != 0)
        goto out;

    // fill in bash_mem_func by looking up external bash mem funcs
    // then finding where they jmp to within bash (internal bash mem funcs)
    if (!bashMemFuncsFound())
        goto out;

    // using bash_mem_func structure, redirect bash internal funcs to ours
    // which will use glibc equivalents.  Voilla!  Now old bashes have their
    // memory subsystem upgraded to glibcs (which is thread safe and supports
    // the new libscope.so thread)
    if (!replaceBashMemFuncs())
        goto out;

    successful = TRUE;
out:
{
    scopeLogError("run_bash_mem_fix was run %s", (successful) ? "successfully" : "but failed");
}
    return successful;
}
