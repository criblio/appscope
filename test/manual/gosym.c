/*
Utility for listing golang symbols from the .gopclntab section of an ELF file

gcc -o gosym ./test/manual/gosym.c
*/
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <elf.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/limits.h>

#define GOPCLNTAB_MAGIC_112 0xfffffffb
#define GOPCLNTAB_MAGIC_116 0xfffffffa

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
                uint64_t sym_count = *((const uint64_t *)(pclntab_addr + 8));
                const void *symtab_addr = pclntab_addr + 16;

                printf("Symbol count = %ld\n", sym_count);
                printf("Address\t\tSymbol Name\n");
                printf("---------------------------\n");
                for (i = 0; i < sym_count; i++) {
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + name_offset);
                    printf("0x%lx\t%s\n", sym_addr, func_name);
                    symtab_addr += 16;
                }
            } else if (magic == GOPCLNTAB_MAGIC_116) {
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

                printf("Symbol count = %ld\n", sym_count);
                printf("Address\t\tSymbol Name\n");
                printf("---------------------------\n");
                for (i = 0; i < sym_count; i++) {
                    uint64_t sym_addr = *((const uint64_t *)(symtab_addr));
                    uint64_t func_offset = *((const uint64_t *)(symtab_addr + 8));
                    uint32_t name_offset = *((const uint32_t *)(pclntab_addr + pclntab_offset + func_offset + 8));
                    const char *func_name = (const char *)(pclntab_addr + funcnametab_offset + name_offset);
                    printf("0x%lx\t%s\n", sym_addr, func_name);
                    symtab_addr += 16;
                }
            } else {
                fprintf(stderr, "Invalid header in .gopclntab\n");
                munmap(buf, st.st_size);
                close(fd);
                return -1;
            }
            break;
        }
    }

    munmap(buf, st.st_size);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc<2) {
        printf("Utility for listing golang symbols from the .gopclntab section of an ELF file\n");
        printf("Usage: gosym elf-file\n");
        exit(0);
    }
    printSymbols(argv[1]);
    exit(0);
}