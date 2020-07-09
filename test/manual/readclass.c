/*

An utility for listing the contents of a java class file

Compile:
gcc test/manual/readclass.c src/javabci.c -I src -o readclass

Sample usage:
./readclass test/manual/JavaTest.class

*/
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdint.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <endian.h>
#include "javabci.h"

char *cp_tag_name[] = {
    "",
    "Utf8", 
    "",
    "Integer",
    "Float",
    "Long",
    "Double",
    "Class",
    "String",
    "Fieldref",
    "Methodref",
    "InterfaceMethodref",
    "NameAndType",
    "",
    "",
    "MethodHandle",
    "MethodType",
    "Dynamic",
    "InvokeDynamic",
    "Module",
    "Package",
};

void 
printCode(unsigned char *addr, int len)
{
    for (int i = 0; i < len; i++) {
        printf("%02x ", *(addr + i));
        if ((i + 1) % 4 == 0) printf("  ");
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

void 
getMethodrefTagInfo(java_class_t *info, unsigned char *addr, char *buf, size_t bufSize)
{
    // CONSTANT_Methodref_info {
    //   u1 tag;
    //   u2 class_index;
    //   u2 name_and_type_index;
    // }
    unsigned char tag = *addr;
    uint16_t class_index = be16toh(*((uint16_t *)(addr + 1)));
    uint16_t name_and_type_index = be16toh(*((uint16_t *)(addr + 3)));
    snprintf(buf, bufSize, "class_index=%d, name_and_type_index=%d", class_index, name_and_type_index);
}

void 
getClassTagInfo(java_class_t *info, unsigned char *addr, char *buf, size_t bufSize)
{
    // CONSTANT_Class_info {
    //   u1 tag;
    //   u2 name_index;
    // }
    unsigned char tag = *addr;
    uint16_t name_index = be16toh(*((uint16_t *)(addr + 1)));
    char *name = javaGetUtf8String(info, name_index);
    snprintf(buf, bufSize,"name_index=%d, name=%s", name_index, name);
    free(name);
}

void 
getNameAndTypeTagInfo(java_class_t *info, unsigned char *addr, char *buf, size_t bufSize)
{
    // CONSTANT_NameAndType_info {
    //   u1 tag;
    //   u2 name_index;
    //   u2 descriptor_index;
    // }
    unsigned char tag = *addr;
    uint16_t name_index = be16toh(*((uint16_t *)(addr + 1)));
    uint16_t desc_index = be16toh(*((uint16_t *)(addr + 3)));
    char *name = javaGetUtf8String(info, name_index);
    char *desc = javaGetUtf8String(info, desc_index);
    snprintf(buf, bufSize, "name_index=%d, desc_index=%d, name=%s, desc=%s", name_index, desc_index, name, desc);
    free(name);
    free(desc);
}

void 
printClassInfo(java_class_t *info)
{
    printf("Version %d.%d\n", info->major_version, info->minor_version);
    printf("CONSTANT POOL, count = %d\n", info->constant_pool_count);
    printf("IDX\tTAG\t%-20s\tLEN\tCONTENTS\n", "NAME");
    for (int i = 1; i < info->constant_pool_count; i++) {
        unsigned char *cp_info = info->constant_pool[i - 1];
        unsigned char tag = *cp_info;
        if (tag == CONSTANT_Double || tag == CONSTANT_Long) i++;

        uint16_t len = javaGetTagLength(cp_info);
        char buf[1000];
        memset(buf, 0, sizeof(buf));
        if (tag == CONSTANT_Utf8) {
            snprintf(buf, len + 1, "%s", (char *)(cp_info + 3));
        } else if (tag == CONSTANT_Methodref) {
            getMethodrefTagInfo(info, cp_info, buf, sizeof(buf));
        } else if (tag == CONSTANT_Class) {
            getClassTagInfo(info, cp_info, buf, sizeof(buf));
        } else if (tag == CONSTANT_NameAndType) {
            getNameAndTypeTagInfo(info, cp_info, buf, sizeof(buf));    
        }
        printf("%3d\t%2d\t%-20s\t%d\t%s\n", i, tag, cp_tag_name[tag], len, buf);
    }
    printf("\nINTERFACES count=%d\n", info->interfaces_count);

    printf("\nFIELDS count=%d\n\n", info->fields_count);
    printf("FLAGS\tNAME IDX\tDESC IDX\tATTR COUNT\n");
    for (int i = 0; i < info->fields_count; i++) {
        unsigned char *off         = info->fields[i];
        uint16_t access_flags      = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t name_index        = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t descriptor_index  = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t attributes_count  = be16toh(*((uint16_t *)off)); off += 2;
        printf("0x%04x\t%d\t\t%d\t\t%d\n", access_flags, name_index, descriptor_index, attributes_count);
        for (int j = 0; j < attributes_count; j++) {
            uint16_t attribute_name_index = be16toh(*((uint16_t *)off)); off += 2;
            uint32_t attribute_length     = be32toh(*((uint32_t *)off)); off += 4;
            off += attribute_length;
            printf("attribute_name_index=%d, attribute_length=%d\n", attribute_name_index, attribute_length);
        }
    }

    printf("\nMETHODS count=%d\n", info->methods_count);

    for (int i = 0; i < info->methods_count; i++) {
        unsigned char *off        = info->methods[i];
        uint16_t access_flags     = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t name_index       = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t descriptor_index = be16toh(*((uint16_t *)off)); off += 2;
        uint16_t attributes_count = be16toh(*((uint16_t *)off)); off += 2;
        char *method_name = javaGetUtf8String(info, name_index);
        char *method_desc = javaGetUtf8String(info, descriptor_index);

        printf("\nMETHOD name=%s, access_flags=0x%04x, name_index=%d, descriptor=%s, attributes_count=%d\n",
            method_name, access_flags, name_index, method_desc, attributes_count);
        free(method_name);
        free(method_desc);

        for (int j = 0; j < attributes_count; j++) {
            uint16_t attr_name_index = be16toh(*((uint16_t *)off)); off += 2;
            uint32_t attr_length     = be32toh(*((uint32_t *)off)); off += 4;
            char *attr_name = javaGetUtf8String(info, attr_name_index);

            if (strcmp(attr_name, "Code") == 0)  {
                /*
                Code_attribute {
                    u2 attribute_name_index; 
                    u4 attribute_length;
                    u2 max_stack;
                    u2 max_locals;
                    u4 code_length;
                    u1 code[code_length];
                    u2 exception_table_length; 
                    { 
                        u2 start_pc;
                        u2 end_pc;
                        u2 handler_pc; 
                        u2 catch_type;
                    } exception_table[exception_table_length]; 
                    u2 attributes_count;
                    attribute_info attributes[attributes_count];
                }
                */
                uint16_t max_stack              = be16toh(*((uint16_t *)off));
                uint16_t max_locals             = be16toh(*((uint16_t *)(off + 2)));
                uint32_t code_length            = be32toh(*((uint32_t *)(off + 4)));
                uint16_t exception_table_length = be32toh(*((uint32_t *)(off + 8 + code_length)));

                printf("CODE name_index=%d, length=%d, max_stack=%d, max_locals=%d, code_lendth=%d, exception_table_length=%d\n",
                       attr_name_index, attr_length, max_stack, max_locals, code_length, exception_table_length);
                printCode(off + 8, code_length);
            } else {
                printf("attribute name=%s, name_index=%d, length=%d\n", attr_name, attr_name_index, attr_length);
            }
            off += attr_length;
            free(attr_name);
        }
    }

    printf("\nATTRIBUTES count=%d\n", info->attributes_count);

    for (int i = 0; i < info->attributes_count; i++) {
        unsigned char *off = info->attributes[i];
        uint16_t attribute_name_index = be16toh(*((uint16_t *)off)); off += 2;
        uint32_t attribute_length     = be32toh(*((uint32_t *)off)); off += 4;
        off += attribute_length;
        printf("attribute_name_index=%d, attribute_length=%d\n", attribute_name_index, attribute_length);
    }
}

int main(int argc, char **argv)
{
    int fd;
    struct stat st;
    java_class_t *classInfo;
    char *src = argv[1];
    char *dst = argv[2];

    if ((fd = open(src, O_RDONLY)) < 0) {
        perror("open");
        exit(-1);
    }
    if (fstat(fd, &st) < 0) {
        perror("stat");
        close(fd);
        exit(-1);
    }
    uint8_t *buf = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if ((classInfo = javaReadClass(buf)) == NULL) {
        fprintf(stderr, "%s is not a Java class file\n", src);
        munmap(buf, st.st_size);
        close(fd);
        exit(-1);
    }

    printClassInfo(classInfo);

    javaDestroy(&classInfo);
    munmap(buf, st.st_size);
    close(fd);
    exit(0);
}