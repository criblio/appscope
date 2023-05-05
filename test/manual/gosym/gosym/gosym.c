/*
Utility for listing golang symbols from the .gopclntab section of an ELF file

gcc -o gosym ./test/manual/gosym.c
*/

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


#define GOPCLNTAB_MAGIC_112 0xfffffffb
#define GOPCLNTAB_MAGIC_116 0xfffffffa
#define GOPCLNTAB_MAGIC_118 0xfffffff0
#define GOPCLNTAB_MAGIC_120 0xfffffff1

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

struct sym_status {
    const char *name;
    bool present;
};

// List of the functions used by the AppScope
static struct sym_status sym_table[] = {
  {.name = "syscall.Syscall"},
  {.name = "syscall.RawSyscall"},
  {.name = "syscall.Syscall6"},
  {.name = "net/http.(*persistConn).readResponse"},
  {.name = "net/http.persistConnWriter.Write"},
  {.name = "net/http.(*connReader).Read"},
  {.name = "net/http.checkConnErrorWriter.Write"},
  {.name = "net/http.(*http2clientConnReadLoop).run"},
  {.name = "net/http.http2stickyErrWriter.Write"},
  {.name = "net/http.(*http2serverConn).readFrames"},
  {.name = "net/http.(*http2serverConn).Flush"},
  {.name = "net/http.(*http2serverConn).readPreface"},
  {.name = "runtime.exit"},
  {.name = "runtime.dieFromSignal"},
  {.name = "runtime.sighandler"},
  {.name = "runtime/internal/syscall.Syscall6"},
};

static bool validate = false;

#define printf_info(format,args...) do {\
        if (validate == false) { \
            printf(format, ## args); \
        } \
    } while(0)

#define printf_validate(format,args...) do {\
        if (validate == true) { \
            printf(format, ## args); \
        } \
    } while(0)

static void updateSymStatus(const char* funcName) {
    for (int i = 0; i < ARRAY_SIZE(sym_table); ++i) {
        if (strcmp(funcName, sym_table[i].name) == 0) {
            sym_table[i].present = true;
        }
    }
}

static void printSymStatus(void) {
    bool allKnownSymbols = true;
    for (int i = 0; i < ARRAY_SIZE(sym_table); ++i) {
        if (sym_table[i].present == false) {
            allKnownSymbols = false;
            printf_validate("[INFO] Missing symbol %s\n", sym_table[i].name);
        }
    }
    if (allKnownSymbols) {
        printf_validate("[INFO] There are no missing symbols\n");
    }
}

int printSymbols(const char *fname)
{
    int fd;
    struct stat st;

    if ((fd = open(fname, O_RDONLY)) < 0) {
        perror("open failed");
        return -1;
    }
    if (fstat(fd, &st) < 0) {
        perror("stat failed");
        close(fd);
        return -1;
    }
    uint8_t *buf = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;

    if (buf == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return -1;
    }

    //check ELF magic
    if (buf[0] != 0x7f && strcmp(&buf[1], "ELF")) {
        fprintf(stderr, "%s is not an ELF file\n", fname);
        munmap(buf, st.st_size);
        close(fd);
        return -1;
    }

    Elf64_Shdr *sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    const char *section_strtab = (char *)buf + sections[ehdr->e_shstrndx].sh_offset;

    for (int i = 0; i < ehdr->e_shnum; i++) {
        const char *sec_name = section_strtab + sections[i].sh_name;
        if (strcmp(".gopclntab", sec_name) == 0) {
            const void *pclntab_addr = buf + sections[i].sh_offset;
            /*
            Go symbol table is stored in the .gopclntab section
            More info: https://docs.google.com/document/d/1lyPIbmsYbXnpNj57a261hgOYVpNRcgydurVQIyZOz_o/pub
            */
            uint32_t magic = *((const uint32_t *)(pclntab_addr));
            if (magic == GOPCLNTAB_MAGIC_112) {
                printf_validate("[INFO] gopclntab was recognized\n");
                uint64_t sym_count = *((const uint64_t *)(pclntab_addr + 8));
                const void *symtab_addr = pclntab_addr + 16;
                printf_info("Symbol count = %ld\n", sym_count);
                printf_info("Address\t\tSymbol Name\n");
                printf_info("---------------------------\n");
                for (i = 0; i < sym_count; i++) {
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + name_offset);
                    printf_info("0x%lx\t%s\n", sym_addr, func_name);

                    symtab_addr += 16;
                    updateSymStatus(func_name);
                }
            } else if (magic == GOPCLNTAB_MAGIC_116) {
                printf_validate("[INFO] gopclntab was recognized\n");
                // the layout of pclntab:
                //
                //  .gopclntab/__gopclntab [elf/macho section]
                //    runtime.pclntab
                //      Carrier symbol for the entire pclntab section.
                //
                //      runtime.pcheader  (see: runtime/symtab.go:pcHeader)
                //        8-byte magic
                //        nfunc [thearch.ptrsize bytes]
                //        nfiles [thearch.ptrsize bytes]
                //
                //        offset to runtime.funcnametab from the beginning of runtime.pcheader
                //        offset to runtime..cutab from the beginning of runtime.pcheader
                //        offset to runtime.filetab from the beginning of runtime.pcheader
                //        offset to runtime.pctab from the beginning of runtime.pcheader
                //        offset to runtime.pclntab from the beginning of runtime.pcheader
                //
                //      runtime.funcnametab
                //        []list of null terminated function names
                //
                //      runtime.cutab
                //        for i=0..#CUs
                //          for j=0..#max used file index in CU[i]
                //            uint32 offset into runtime.filetab for the filename[j]
                //
                //      runtime.filetab
                //        []null terminated filename strings
                //
                //      runtime.pctab
                //        []byte of deduplicated pc data.
                //
                //      runtime.functab
                //        function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
                //        end PC [thearch.ptrsize bytes]
                //        func structures, pcdata offsets, func data.
                uint64_t sym_count = *((const uint64_t *)(pclntab_addr + 8));

                uint64_t funcnametab_offset = *((const uint64_t *)(pclntab_addr + (3 * 8)));
                uint64_t pclntab_offset = *((const uint64_t *)(pclntab_addr + (7 * 8)));

                const void *symtab_addr = pclntab_addr + pclntab_offset;
                printf_info("Symbol count = %ld\n", sym_count);
                printf_info("Address\t\tSymbol Name\n");
                printf_info("---------------------------\n");
                for (i = 0; i < sym_count; i++) {
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    printf_info("0x%lx\t%s\n", sym_addr, func_name);

                    symtab_addr += 16;
                    updateSymStatus(func_name);
                }
            } else if ((magic == GOPCLNTAB_MAGIC_118) || (magic == GOPCLNTAB_MAGIC_120)) {
                printf_validate("[INFO] gopclntab was recognized\n");
                uint64_t sym_count = *((const uint64_t *)(pclntab_addr + 8));
                uint64_t funcnametab_offset = *((const uint64_t *)(pclntab_addr + (4 * 8)));
                uint64_t pclntab_offset = *((const uint64_t *)(pclntab_addr + (8 * 8)));
                uint64_t text_start = *((const uint64_t *)(pclntab_addr + (3 * 8)));

                const void *symtab_addr = pclntab_addr + pclntab_offset;

                printf_info("Symbol count = %ld\n", sym_count);
                printf_info("Address\t\tSymbol Name\n");
                printf_info("---------------------------\n");
                for (i = 0; i < sym_count; i++) {
                    uint32_t func_offset = *((uint32_t *)(symtab_addr + 4));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 4));
                    func_offset = *((uint32_t *)(symtab_addr));
                    uint64_t sym_addr = (uint64_t)(func_offset + text_start);
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    printf_info("0x%lx\t%s\n", sym_addr, func_name);
                    symtab_addr += 8;
                    updateSymStatus(func_name);
                }
            } else {
                fprintf(stderr, "[ERROR] Unknown header in .gopclntab\n");
                munmap(buf, st.st_size);
                close(fd);
                return -1;
            }
            break;
        }
    }

    printSymStatus();

    munmap(buf, st.st_size);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "v")) != -1) {
        switch (opt)
        {
            case 'v':
                validate = true;
                break;
            default:
                printf("Usage: %s elf-file [-v]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (argc < 2) {
        printf("Utility for listing golang symbols from the .gopclntab section of an ELF file\n");
        printf("Usage: %s elf-file [-v]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    printSymbols(argv[optind]);

    exit(EXIT_SUCCESS);
}
