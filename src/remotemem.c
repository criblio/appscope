#define _GNU_SOURCE

#include "remotemem.h"
#include "vector.h"
#include "scopeelf.h"

// mem_library_t represents library which residents in the process memory
typedef vector mem_library_t;

// mem_region_t represents single region of library which residents in the process memory

typedef struct {
    uint64_t vAddrBegin;        // virtual address begin
    uint64_t vAddrEnd;          // virtual address end
    uint64_t vAddrSize;         // virtual address size
    void *   buf;               // content of the memory
} mem_region_t;

typedef struct {
    Elf64_Word nbucket;
    Elf64_Word nchain;
    /* an array of buckets followed by an array of chains */
    Elf64_Word entries[];
} Elf64_Hash_t;

typedef struct {
    unsigned long vaddr;
    unsigned long size;

    const Elf64_Dyn *entries;
    unsigned long count;
} dynamic_section_t;

typedef struct {
    unsigned long vaddr;
    unsigned long size;

    const char *strings;
} dynstr_section_t;

typedef struct {
    unsigned long vaddr;
    unsigned long syment;

    const Elf64_Sym *symbols;
    unsigned long count;
} dynsym_section_t;

typedef struct {
    unsigned long vaddr;

    const Elf64_Hash_t *table;
} hash_section_t;

typedef struct {
    unsigned long base_vaddr;
    dynstr_section_t dynstr;
    dynsym_section_t dynsym;
    hash_section_t hash;
} symbol_table_t;

 /*
 * Creates the memory region structure based on begin and end arguments.
 *
 * Returns memory region, NULL on failure.
 */
static mem_region_t *
memRegionCreate(uint64_t begin, uint64_t end) {
    mem_region_t *region = scope_calloc(1, sizeof(mem_region_t));
    if (!region) {
        return NULL;
    }
    region->vAddrBegin = begin;
    region->vAddrEnd = end;
    region->vAddrSize = end - begin;
    region->buf = scope_malloc(sizeof(char) * (region->vAddrSize));
    if (!region->buf) {
        scope_free(region);
        return NULL;
    }
    return region;
}

 /*
 * Destroys memory region.
 */
static void
memRegionDestroy(mem_region_t *region) {
    scope_free(region->buf);
    scope_free(region);
}

 /*
 * Initiates memory library
 */
static bool
memLibInit(mem_library_t *memLib) {
    return vecInit(memLib);
}

static mem_region_t *
memLibGetRegion(mem_library_t *memLib, unsigned regionIndex) {
    return vecGet(memLib, regionIndex);
}

 /*
 * Destroys memory library.
 */
static void
memLibDestroy(mem_library_t * memLib) {
    for (unsigned i = 0; i < memLib->size; ++i) {
        memRegionDestroy(memLibGetRegion(memLib, i));
    }

    vecDelete(memLib);
}

static const void*
resolve_vaddr(unsigned long vaddr, mem_library_t *memLib)
{
    for (unsigned i = 0; i < memLib->size; ++i) {
        mem_region_t *region = memLibGetRegion(memLib, i);
        if (region->vAddrBegin <= vaddr && vaddr < region->vAddrEnd)
            return region->buf + (vaddr - region->vAddrBegin);
    }
    scope_fprintf(scope_stderr, "\nresolve_vaddr failed");
    return NULL;
}


 /*
 * Append the memory region to library structure on begin and end arguments.
 *
 * Returns status of operation - TRUE on success, FALSE on failure.
 */
static bool
memLibAppendRegion(mem_library_t *memLib, uint64_t regionBegin, uint64_t regionEnd) {
    // create new memory region 
    mem_region_t *region = memRegionCreate(regionBegin, regionEnd);
    if (!region) {
        return FALSE;
    }

    return vecAdd(memLib, region);
}

 /*
 * Parse specific memory map file and allocate memmory for library mapping.
 *
 * Returns status of operation - TRUE on success, FALSE on failure.
 */
static bool
memoryLibMap(pid_t pid, mem_library_t *memLib) {
    char filename[PATH_MAX];
    char line[1024];
    FILE *fd;

    scope_snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
    if ((fd = scope_fopen(filename, "r")) == NULL) {
        return FALSE;
    }

    while (scope_fgets(line, sizeof(line), fd)) {
        uint64_t vAddrBegin = 0;
        uint64_t vAddrEnd = 0;
        char pathFromLine[512] = {0};
        char perms[5] = {0};
        if ((scope_sscanf(line, "%lx-%lx %s %*x %*s %*d %512s", &vAddrBegin, &vAddrEnd, perms, pathFromLine) == 4)) {
            // scope_fprintf(scope_stderr, "\nBegin: %lx End: %lx lib: %s", vAddrBegin, vAddrEnd, pathFromLine);
            if ((!scope_strstr(pathFromLine, "/libc-")) && (!scope_strstr(pathFromLine, "/ld-musl-"))) {
                continue;
            }

            // Add only readable segment
            if (!scope_strstr(perms, "r")) {
                continue;
            }

            if (memLibAppendRegion(memLib, vAddrBegin, vAddrEnd) == FALSE) {
                scope_fclose(fd);
                return FALSE;
            }
        }
    }
    scope_fclose(fd);
    return TRUE;
}

typedef struct {
    const Elf64_Phdr *headers;
    uint16_t count;
} program_headers_t;

static const Elf64_Phdr *
find_pt_dynamic_header(program_headers_t*ph)
{
    /*
     * Actually, ELF also has a wacky special case for when there are
     * more than PN_XNUM program headers, but we do not handle it here
     * as this will require access to section headers.
     */
    for (uint16_t i = 0; i < ph->count; i++) {
        if (ph->headers[i].p_type == PT_DYNAMIC) {
            return &ph->headers[i];
        }
    }
    return NULL;
}

static int
locate_program_headers(mem_library_t *memLib, program_headers_t *ph) {
    
    mem_region_t *reg = memLibGetRegion(memLib, 0);

    unsigned long base_vaddr = reg->vAddrBegin;
    const Elf64_Ehdr *ehdr = reg->buf;

    if (ehdr->e_phoff == 0 || ehdr->e_phnum == 0) {
        return -1;
    }

    ph->count = ehdr->e_phnum;
    ph->headers = resolve_vaddr(base_vaddr + ehdr->e_phoff, memLib);

    if (!ph->headers) {
        return -1;
    }

    return 0;
}

static int
locate_dynamic_section(mem_library_t *memLib, dynamic_section_t *dynamic) {
    program_headers_t ph;

    /*
     * In order to locate the ".dynamic" section we have to use program
     * headers, not section headers. Usually the section headers are not
     * mapped into memory as they are not necessary for library loading.
     */

    int resH = locate_program_headers(memLib, &ph);

    if (resH < 0) {
        scope_fprintf(scope_stderr, "\nlocate_program_headers failed");
        return -1;
	}

    /*
     * Scan through the headers to find the PT_DYNAMIC one. This is the
     * program header that describes where exactly the .dynamic section
     * is located when loaded.
     */
    const Elf64_Phdr *pt_dynamic = find_pt_dynamic_header(&ph);
    if (!pt_dynamic) {
        scope_fprintf(scope_stderr, "\nfind_pt_dynamic_header failed");
        return -1;
    }

    /*
     * Now we can locate the section data in our memory mapping.
     * The 'p_vaddr' field actually contains virtual address _offset_
     * of the ".dynamic" section, not its absolute virtual address.
     */
    mem_region_t *reg = memLibGetRegion(memLib, 0);
    unsigned long base_vaddr = reg->vAddrBegin;
    dynamic->vaddr = base_vaddr + pt_dynamic->p_vaddr;
    dynamic->size = pt_dynamic->p_memsz;
    dynamic->entries = resolve_vaddr(dynamic->vaddr, memLib);
    dynamic->count = dynamic->size / sizeof(Elf64_Dyn);

    if (!dynamic->entries) {
        scope_fprintf(scope_stderr, "\ndynamic->entries empty");
        return -1;
    }

    return 0;
}

static int locate_dynstr_section(dynamic_section_t *dynamic, mem_library_t *mapping, dynstr_section_t *dynstr)
{
    dynstr->vaddr = 0;
    dynstr->size = 0;

    for (unsigned long i = 0; i < dynamic->count; i++) {
        if (dynamic->entries[i].d_tag == DT_STRTAB)
            dynstr->vaddr = dynamic->entries[i].d_un.d_ptr;

        if (dynamic->entries[i].d_tag == DT_STRSZ)
            dynstr->size = dynamic->entries[i].d_un.d_val;

        if (dynamic->entries[i].d_tag == DT_NULL)
            break;
    }

    if (!dynstr->vaddr) {
        return -1;
    }
    if (!dynstr->size) {
        return -1;
    }

    dynstr->strings = resolve_vaddr(dynstr->vaddr, mapping);

    if (!dynstr->strings) {
        return -1;
    }

    return 0;
}

static int locate_dynsym_section(dynamic_section_t *dynamic, mem_library_t *mapping, dynsym_section_t *dynsym)
{
    dynsym->vaddr = 0;
    dynsym->syment = 0;

    for (unsigned long i = 0; i < dynamic->count; i++) {
        if (dynamic->entries[i].d_tag == DT_SYMTAB)
            dynsym->vaddr = dynamic->entries[i].d_un.d_ptr;

        if (dynamic->entries[i].d_tag == DT_SYMENT)
            dynsym->syment = dynamic->entries[i].d_un.d_val;

        if (dynamic->entries[i].d_tag == DT_NULL)
            break;
    }

    if (!dynsym->vaddr) {
        return -1;
    }
    if (!dynsym->syment) {
        return -1;
    }

    dynsym->symbols = resolve_vaddr(dynsym->vaddr, mapping);
    dynsym->count = 0;

    if (!dynsym->symbols) {
        return -1;
    }

    return 0;
}

static int
locate_hash_section(dynamic_section_t *dynamic, mem_library_t *mapping, hash_section_t *hash) {
    hash->vaddr = 0;

    for (unsigned long i = 0; i < dynamic->count; i++) {
        if (dynamic->entries[i].d_tag == DT_HASH)
            hash->vaddr = dynamic->entries[i].d_un.d_ptr;

        if (dynamic->entries[i].d_tag == DT_NULL)
            break;
    }

    if (!hash->vaddr) {
        return -1;
    }

    hash->table = resolve_vaddr(hash->vaddr, mapping);

    if (!hash->table) {
        return -1;
    }

    return 0;
}

static symbol_table_t *
findDynamicSymbolTable(mem_library_t *memLib) {
    if (memLib->size < 1) {
        return NULL;
    }

    const mem_region_t *begin = memLibGetRegion(memLib, 0);
    // Compare header of library to be ELF based
    const Elf64_Ehdr *ehdr = (Elf64_Ehdr *)begin->buf;
    if (!(ehdr->e_ident[EI_MAG0] == ELFMAG0 &&
          ehdr->e_ident[EI_MAG1] == ELFMAG1 &&
          ehdr->e_ident[EI_MAG2] == ELFMAG2 &&
          ehdr->e_ident[EI_MAG3] == ELFMAG3)) {
        return NULL;
    }

    dynamic_section_t dynamic;

    if (locate_dynamic_section(memLib, &dynamic) < 0) {
        scope_fprintf(scope_stderr, "\nlocate_dynamic_section failed");
        return NULL;
    }

    dynstr_section_t dynstr;
    dynsym_section_t dynsym;
    hash_section_t hash;

    if (locate_dynstr_section(&dynamic, memLib, &dynstr) < 0) {
        scope_fprintf(scope_stderr, "\nlocate_dynstr_section failed");
        return NULL;
    }
    if (locate_dynsym_section(&dynamic, memLib, &dynsym) < 0) {
        scope_fprintf(scope_stderr, "\nlocate_dynsym_section failed");
        return NULL;
    }
    if (locate_hash_section(&dynamic, memLib, &hash) < 0) {
        scope_fprintf(scope_stderr, "\nlocate_hash_section failed");
        return NULL;
    }

    dynsym.count = hash.table->nchain;

    symbol_table_t *table = scope_calloc(1, sizeof(*table));
    if (!table) {
        return NULL;
    }

    table->base_vaddr = begin->vAddrBegin;
    table->dynstr = dynstr;
    table->dynsym = dynsym;
    table->hash = hash;

    return table;
}

 /*
 * Copy virtual mapping from remote process
 *
 * Returns status of operation - TRUE on success, FALSE on failure.
 */
static bool
memoryLibRead(pid_t pid, mem_library_t *memLib) {
    bool status = FALSE;
    ssize_t vmReadBytes = 0;
    size_t libSizeBytes = 0;

    struct iovec *localIov = scope_calloc(memLib->size, sizeof(struct iovec));
    if (!localIov) {
        return status;
    }
    struct iovec *remoteIov = scope_calloc(memLib->size, sizeof(struct iovec));
    if (!remoteIov) {
        goto freeLocalIov;
    }

    for (unsigned i = 0; i < memLib->size; ++i) {
        mem_region_t *reg = memLibGetRegion(memLib, i);

        localIov[i].iov_base = reg->buf;
        localIov[i].iov_len = reg->vAddrSize;
        remoteIov[i].iov_base = (void*) reg->vAddrBegin;
        remoteIov[i].iov_len = reg->vAddrSize;
        libSizeBytes += reg->vAddrSize;
    }

    vmReadBytes = scope_process_vm_readv(pid, localIov, memLib->size, remoteIov, memLib->size, 0);
    if (vmReadBytes < 0) {
        goto freeRemoteIov;
    }

    if (libSizeBytes != vmReadBytes) {
        goto freeRemoteIov;
    }

    status = TRUE;

freeRemoteIov:
    scope_free(remoteIov);
   
freeLocalIov:
    scope_free(localIov);
    return status;
}

static unsigned long symbol_address(const char *name,
        const Elf64_Sym *symbol, symbol_table_t *symbols)
{
    uint8_t bind = ELF64_ST_BIND(symbol->st_info);
    uint8_t type = ELF64_ST_TYPE(symbol->st_info);

    if (symbol->st_shndx == STN_UNDEF) {
        return 0;
    }

    if ((bind != STB_GLOBAL) && (bind != STB_WEAK)) {
        return 0;
    }

    if ((type != STT_FUNC) && (type != STT_OBJECT)) {
        return 0;
    }

    return symbols->base_vaddr + symbol->st_value;
}


static Elf64_Word elf_hash(const char *name)
{
    Elf64_Word h = 0;
    
    while (*name) {
        h = (h << 4) + *name++;
    
        Elf64_Word g = h & 0xF0000000;

        if (g)
            h ^= g >> 24;
    
        h &= ~g;
    }

    return h;
}

static const char* symbol_name(const Elf64_Sym *symbol, symbol_table_t *symbols)
{
    if (!symbol->st_name)
        return "";

    return &symbols->dynstr.strings[symbol->st_name];
}

unsigned long resolveSymbol(const char *name, symbol_table_t *symbols)
{
    Elf64_Word nbucket = symbols->hash.table->nbucket;
    const Elf64_Word *buckets = &symbols->hash.table->entries[0];
    const Elf64_Word *chains = &symbols->hash.table->entries[nbucket];

    Elf64_Word hash = elf_hash(name);
    Elf64_Word bucket = hash % nbucket;
    Elf64_Word index = buckets[bucket];
    
    for (;;) {
        if (index > symbols->dynsym.count) {
            return 0;
        }

        const Elf64_Sym *symbol = &symbols->dynsym.symbols[index];

        if (scope_strcmp(symbol_name(symbol, symbols), name) == 0)
            return symbol_address(name, symbol, symbols);
    
        index = chains[index];

        if (index == STN_UNDEF) {
            return 0;
        }
    }
}

void free_symbol_table(symbol_table_t *table)
{
    scope_free(table);
}

 /*
 * Find the addresss of specific symbol in remote process
 *
 * Returns address of the symbol, NULL on failure.
 */
uint64_t
remoteProcSymbolAddr(pid_t pid, const char *symbolName) {
    uint64_t symAddr = 0;
    mem_library_t memLib;
    if (memLibInit(&memLib) == FALSE) {
        return symAddr;
    }

    if (memoryLibMap(pid, &memLib) == FALSE) {
        scope_fprintf(scope_stderr, "\nmemoryLibMap failed");
        goto destroyLib;
    }

    if (memoryLibRead(pid, &memLib) == FALSE) {
        scope_fprintf(scope_stderr, "\nmemoryLibRead failed");
        goto destroyLib;
    }

    symbol_table_t *libSymTable = findDynamicSymbolTable(&memLib);
    if (!libSymTable) {
        scope_fprintf(scope_stderr, "\nfindDynamicSymbolTable failed");
        goto destroyLib;
    }

    symAddr = resolveSymbol(symbolName, libSymTable);

    if (symAddr == 0) {
        scope_fprintf(scope_stderr, "\nresolveSymbol failed");
    }

    free_symbol_table(libSymTable);

destroyLib:
    memLibDestroy(&memLib);

    return symAddr;
}
