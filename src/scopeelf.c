#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "dbg.h"
#include "os.h"
#include "fn.h"
#include "scopeelf.h"

#define GOPCLNTAB_MAGIC_112 0xfffffffb
#define GOPCLNTAB_MAGIC_116 0xfffffffa

void
freeElf(char *buf, size_t len)
{
    if (!buf) return;

    if (munmap(buf, len) == -1) {
        scopeLog("freeElf: munmap failed", -1, CFG_LOG_ERROR);
    }
}

static void
setTextSizeAndLenFromElf(elf_buf_t *ebuf)
{
    int i;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *sec_name;
    const char *section_strtab = NULL;
    ehdr = (Elf64_Ehdr *)ebuf->buf;
    sections = (Elf64_Shdr *)((char *)ebuf->buf + ehdr->e_shoff);
    section_strtab = (char *)ebuf->buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;

        if (!strcmp(sec_name, ".text")) {
            ebuf->text_addr = (unsigned char *)sections[i].sh_addr;
            ebuf->text_len = sections[i].sh_size;
            char msg[128];
            snprintf(msg, sizeof(msg), "%s:%d %s addr %p - %p\n",
                     __FUNCTION__, __LINE__, sec_name, ebuf->text_addr, ebuf->text_addr + ebuf->text_len);
            scopeLog(msg, -1, CFG_LOG_DEBUG);
        }
    }
}

static bool
app_type(char *buf, const uint32_t sh_type, const char *sh_name)
{
    int i = 0;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        //printf("section %s type = %d \n", sec_name, sections[i].sh_type);
        if (sections[i].sh_type == sh_type && strcmp(sec_name, sh_name) == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

Elf64_Shdr*
getElfSection(char *buf, const char *sh_name)
{
    int i = 0;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        if (strcmp(sec_name, sh_name) == 0) {
            return &sections[i];
        }
    }
    return NULL;
}

elf_buf_t *
getElf(char *path)
{
    int fd = -1;
    elf_buf_t *ebuf = NULL;
    Elf64_Ehdr *elf;
    struct stat sbuf;
    int get_elf_successful = FALSE;

    if (!g_fn.open || !g_fn.close) {
        scopeLog("getElf: open/close can't be found", -1, CFG_LOG_ERROR);
        goto out;
    }

    if ((ebuf = calloc(1, sizeof(elf_buf_t))) == NULL) {
        scopeLog("getElf: memory alloc failed", -1, CFG_LOG_ERROR);
        goto out;
    }

    if ((fd = g_fn.open(path, O_RDONLY)) == -1) {
        scopeLog("getElf: open failed", -1, CFG_LOG_ERROR);
        goto out;
    }

    if (fstat(fd, &sbuf) == -1) {
        scopeLog("getElf: fstat failed", fd, CFG_LOG_ERROR);
        goto out;
    }


    char * mmap_rv = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                          PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (mmap_rv == MAP_FAILED) {
        scopeLog("getElf: mmap failed", fd, CFG_LOG_ERROR);
        goto out;
    }

    ebuf->cmd = path;
    ebuf->buf = mmap_rv;
    ebuf->len = sbuf.st_size;

    elf = (Elf64_Ehdr *)ebuf->buf;
    if((elf->e_ident[EI_MAG0] != 0x7f) ||
       strncmp((char *)&elf->e_ident[EI_MAG1], "ELF", 3) ||
       (elf->e_ident[EI_CLASS] != ELFCLASS64) ||
       (elf->e_ident[EI_DATA] != ELFDATA2LSB) ||
       (elf->e_machine != EM_X86_64)) {
        char emsg[64];
        snprintf(emsg, sizeof(emsg), "%s:%d ERROR: %s is not a viable ELF file\n",
                 __FUNCTION__, __LINE__, path);
        scopeLog(emsg, fd, CFG_LOG_ERROR);
        goto out;
    }

    if ((elf->e_type != ET_EXEC) && (elf->e_type != ET_DYN)) {
        char emsg[128];
        snprintf(emsg, sizeof(emsg), "%s:%d %s with type %d is not an executable\n",
                 __FUNCTION__, __LINE__, path, elf->e_type);
        scopeLog(emsg, fd, CFG_LOG_ERROR);
        goto out;
    }

    setTextSizeAndLenFromElf(ebuf);

    get_elf_successful = TRUE;

out:
    if (fd != -1) g_fn.close(fd);
    if (!get_elf_successful && ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
        ebuf = NULL;
    }
    return ebuf;
}

/*
 * Find the GOT ptr as defined in RELA/DYNSYM (.rela.dyn) for symbol & hook it.
 * .rela.dyn section:
 * The address of relocation entries associated solely with the PLT.
 * The relocation table's entries have a one-to-one correspondence with the PLT.
 */
int
doGotcha(struct link_map *lm, got_list_t *hook, Elf64_Rela *rel, Elf64_Sym *sym, char *str, int rsz, int attach)
{
    int i, match = -1;
    char buf[128];

    for (i = 0; i < rsz / sizeof(Elf64_Rela); i++) {
        /*
         * Index into the dynamic symbol table (not the 'symbol' table)
         * with the index from the current relocation entry and get the
         * sym tab offset (st_name). The index is calculated with the
         * ELF64_R_SYM macro, which shifts the elf value >> 32. The dyn sym
         * tab offset is added to the start of the string table (str) to get
         * the string name. Compare the str entry with a symbol in the list.
         *
         * Note, it would be nice to check the array bounds before indexing the
         * sym[] table. However, DT_SYMENT gives the byte size of a single entry.
         * According to the ELF spec there is no size/number of entries for the
         * symbol table at the program header table level. This is not needed at
         * runtime as the symbol lookup always go through the hash table; ELF64_R_SYM.
         */
        if (!strcmp(sym[ELF64_R_SYM(rel[i].r_info)].st_name + str, hook->symbol)) {
            uint64_t *gfn = hook->gfn;
            uint64_t *gaddr = (uint64_t *)(rel[i].r_offset + lm->l_addr);
            int page_size = getpagesize();
            size_t saddr = ROUND_DOWN((size_t)gaddr, page_size);
            int prot = osGetPageProt((uint64_t)gaddr);

            if (prot != -1) {
                if ((prot & PROT_WRITE) == 0) {
                    // mprotect if write perms are not set
                    if (mprotect((void *)saddr, (size_t)16, PROT_WRITE | prot) == -1) {
                        scopeLog("doGotcha: mprotect failed", -1, CFG_LOG_DEBUG);
                        return -1;
                    }
                }
            } else {
                /*
                 * We don't have a valid protection setting for the GOT page.
                 * It's "almost assuredly" safe to set perms to RD | WR as this is
                 * a GOT page; we know what settings are expected. However, it
                 * may be safest to just bail out at this point.
                 */
                return -1;
            }

            /*
             * The offset from the matching relocation entry defines the GOT
             * entry associated with 'symbol'. ELF docs describe that this
             * is an offset and not a virtual address. Take the load address
             * of the shared module as defined in the link map's l_addr + offset.
             * as in: rel[i].r_offset + lm->l_addr
             */
            if (!attach) *gfn = *gaddr;
            uint64_t prev = *gaddr;
            *gaddr = (uint64_t)hook->func;
            snprintf(buf, sizeof(buf), "%s:%d sym=%s offset 0x%lx GOT entry %p saddr 0x%lx, prev=0x%lx, curr=%p",
                     __FUNCTION__, __LINE__, hook->symbol, rel[i].r_offset, gaddr, saddr, prev, hook->func);
            scopeLog(buf, -1, CFG_LOG_DEBUG);

            if ((prot & PROT_WRITE) == 0) {
                // if we didn't mod above leave prot settings as is
                if (mprotect((void *)saddr, (size_t)16, prot) == -1) {
                    scopeLog("doGotcha: mprotect failed", -1, CFG_LOG_DEBUG);
                    return -1;
                }
            }
            match = 0;
            break;
        }
    }

    return match;
}

// Locate the needed elf entries from a given link map.
int
getElfEntries(struct link_map *lm, Elf64_Rela **rel, Elf64_Sym **sym, char **str, int *rsz)
{
    Elf64_Dyn *dyn = NULL;
    char *got = NULL; // TODO; got is not needed, debug, remove
    char buf[256];

    for (dyn = lm->l_ld; dyn->d_tag != DT_NULL; dyn++) {
        if (dyn->d_tag == DT_SYMTAB) {
            *sym = (Elf64_Sym *)((char *)(dyn->d_un.d_ptr + lm->l_addr));
        } else if (dyn->d_tag == DT_STRTAB) {
            *str = (char *)(dyn->d_un.d_ptr + lm->l_addr);
        } else if (dyn->d_tag == DT_JMPREL) {
            *rel = (Elf64_Rela *)((char *)(dyn->d_un.d_ptr + lm->l_addr));
        } else if (dyn->d_tag == DT_PLTRELSZ) {
            *rsz = dyn->d_un.d_val;
        } else if (dyn->d_tag == DT_PLTGOT) {
            got = (char *)(dyn->d_un.d_ptr + lm->l_addr);
        }
    }

    snprintf(buf, sizeof(buf), "%s:%d name: %s dyn %p sym %p rel %p str %p rsz %d got %p laddr 0x%lx\n",
             __FUNCTION__, __LINE__, lm->l_name, dyn, *sym, *rel, *str, *rsz, got, lm->l_addr);
    scopeLog(buf, -1, CFG_LOG_TRACE);

    if (*sym == NULL || *rel == NULL || (*rsz < sizeof(Elf64_Rela))) {
        return -1;
    }

    return 0;
}

void *
getSymbol(const char *buf, char *sname)
{
    int i, nsyms = 0;
    Elf64_Addr symaddr = 0;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    Elf64_Sym *symtab = NULL;
    const char *section_strtab = NULL;
    const char *strtab = NULL;
    const char *sec_name = NULL;

    if (!buf || !sname) return NULL;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)((char *)buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;

        if (sections[i].sh_type == SHT_SYMTAB) {
            symtab = (Elf64_Sym *)((char *)buf + sections[i].sh_offset);
            nsyms = sections[i].sh_size / sections[i].sh_entsize;
        } else if (sections[i].sh_type == SHT_STRTAB && strcmp(sec_name, ".strtab") == 0) {
            strtab = (const char *)(buf + sections[i].sh_offset);
        }

        if ((strtab != NULL) && (symtab != NULL)) break;
        /*printf("section %s type = %d flags = 0x%lx addr = 0x%lx-0x%lx, size = 0x%lx off = 0x%lx\n",
               sec_name,
               sections[i].sh_type,
               sections[i].sh_flags,
               sections[i].sh_addr,
               sections[i].sh_addr + sections[i].sh_size,
               sections[i].sh_size,
               sections[i].sh_offset);*/
    }

    for (i=0; i < nsyms; i++) {
        if (strcmp(sname, strtab + symtab[i].st_name) == 0) {
            symaddr = symtab[i].st_value;
            char buf[256];
            snprintf(buf, sizeof(buf), "symbol found %s = 0x%08lx\n", strtab + symtab[i].st_name, symtab[i].st_value);
            scopeLog(buf, -1, CFG_LOG_TRACE);
            break;
        }
    }

    return (void *)symaddr;
}

void *
getGoSymbol(const char *buf, char *sname)
{
    int i;
    Elf64_Addr symaddr = 0;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    if (!buf || !sname) return NULL;

    ehdr = (Elf64_Ehdr *)buf;
    sections = (Elf64_Shdr *)((char *)buf + ehdr->e_shoff);
    section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        if (strcmp(".gopclntab", sec_name) == 0) {
            const void *pclntab_addr = buf + sections[i].sh_offset;
            /*
            Go symbol table is stored in the .gopclntab section
            More info: https://docs.google.com/document/d/1lyPIbmsYbXnpNj57a261hgOYVpNRcgydurVQIyZOz_o/pub
            */
            uint32_t magic = *((const uint32_t *)(pclntab_addr));
            if (magic == GOPCLNTAB_MAGIC_112) {
                uint64_t sym_count      = *((const uint64_t *)(pclntab_addr + 8));
                const void *symtab_addr = pclntab_addr + 16;

                for(i=0; i<sym_count; i++) {
                    uint64_t sym_addr     = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset  = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset  = *((const uint32_t *)(pclntab_addr + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + name_offset);

                    if (strcmp(sname, func_name) == 0) {
                        symaddr = sym_addr;
                        char buf[512];
                        snprintf(buf, sizeof(buf), "symbol found %s = 0x%08lx\n", func_name, sym_addr);
                        scopeLog(buf, -1, CFG_LOG_TRACE);
                        break;
                    }
                    symtab_addr += 16;
                }
            } else if (magic == GOPCLNTAB_MAGIC_116) {
                uint64_t sym_count      = *((const uint64_t *)(pclntab_addr + 8));
                uint64_t funcnametab_offset = *((const uint64_t *)(pclntab_addr + (3 * 8)));
                uint64_t pclntab_offset = *((const uint64_t *)(pclntab_addr + (7 * 8)));
                const void *symtab_addr = pclntab_addr + pclntab_offset;
                for (i = 0; i < sym_count; i++) {
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    if (strcmp(sname, func_name) == 0) {
                        symaddr = sym_addr;
                        char buf[512];
                        snprintf(buf, sizeof(buf), "symbol found %s = 0x%08lx\n", func_name, sym_addr);
                        scopeLog(buf, -1, CFG_LOG_TRACE);
                        break;
                    }
                    symtab_addr += 16;
                }
            } else {
                scopeLog("Invalid header in .gopclntab", -1, CFG_LOG_DEBUG);
                break;
            }
            break;
        }
    }

    return (void *)symaddr;
}

void *
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
        // 0xf                      endianess
        // 0x10                     pointer to string runtime.buildVersion
        // 0x10 + ptrSize           pointer to runtime.modinfo
        // 0x10 + 2 * ptr size      pointer to build flags

        if (!strcmp(sec_name, ".go.buildinfo") &&
            (sections[i].sh_size >= 0x18) &&
            (!memcmp(&sec_data[0], magic, sizeof(magic))) &&
            (sec_data[0xe] == 0x08) &&  // 64 bit executables only
            (sec_data[0xf] == 0x00)) {  // little-endian

            uint64_t *addressPtr = (uint64_t*)&sec_data[0x10];

            go_build_ver_addr = (void*)*addressPtr;
        }
    }

    return go_build_ver_addr;
}

bool
is_static(char *buf)
{
    int i;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_DYNAMIC) || (phead[i].p_type == PT_INTERP)) {
            return FALSE;
        }
    }

    return TRUE;
}

bool
is_go(char *buf)
{
    if (buf && (app_type(buf, SHT_PROGBITS, ".gosymtab") ||
                app_type(buf, SHT_PROGBITS, ".gopclntab") ||
                app_type(buf, SHT_NOTE, ".note.go.buildid"))) {
        return TRUE;
    }

    return FALSE;
}

bool
is_musl(char *buf)
{
    int i;
    char *ldso = NULL;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char *exld = (char *)&buf[phead[i].p_offset];

            ldso = strdup(exld);
            if (ldso) {
                if (strstr(ldso, "musl") != NULL) {
                    free(ldso);
                    return TRUE;
                }
                free(ldso);
            } else {
                DBG(NULL); // not expected
            }
            break;
        }
    }

    return FALSE;
}
