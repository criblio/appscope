/*
An utility for listing the contents of a java class
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

char * cp_tag_name[] = {
  "",
  "Utf8",//                     = 1,
  "",
  "Integer",//                  = 3,
  "Float",//                    = 4,
  "Long",//                     = 5,
  "Double",//                   = 6,
  "Class",//                    = 7,
  "String",//                   = 8,
  "Fieldref",//                 = 9,
  "Methodref",//                = 10, 
  "InterfaceMethodref",//       = 11,
  "NameAndType",//              = 12,
  "",
  "",
  "MethodHandle",//             = 15,
  "MethodType",//               = 16,
  "Dynamic",//                  = 17,
  "InvokeDynamic",//            = 18,
  "Module",//                   = 19,
  "Package",//                  = 20
};

void printCode(void *addr, int len) {
  for(int i=0;i<len;i++) {
    printf("%02x ", *((uint8_t *)(addr + i)) );
    if ((i+1) % 4==0) printf("  ");
    if ((i+1) % 16==0) printf("\n");
  }
  printf("\n");
}

void printMethodrefTag(java_class_t *info, void *addr) {
  // CONSTANT_Methodref_info { 
  //   u1 tag;
  //   u2 class_index;
  //   u2 name_and_type_index; 
  // }
  uint8_t tag                  = readUInt8(addr);
  uint16_t class_index         = readUInt16(addr + 1);
  uint16_t name_and_type_index = readUInt16(addr + 3);
  printf(" -- Methodref class_index=%d, name_and_type_index=%d\n", class_index, name_and_type_index);
}

void printClassTag(java_class_t *info, void *addr) {
  // CONSTANT_Class_info { 
  //   u1 tag;
  //   u2 name_index; 
  // }
  uint8_t tag         = readUInt8(addr);
  uint16_t name_index = readUInt16(addr + 1);
  char *name = javaGetUtf8String(info, name_index);
  printf(" -- Class name_index=%d, name=%s\n", name_index, name);
  free(name);
}

void printNameAndTypeTag(java_class_t *info, void *addr) {
  // CONSTANT_NameAndType_info { 
  //   u1 tag;
  //   u2 name_index;
  //   u2 descriptor_index;
  // }
  uint8_t tag         = readUInt8(addr);
  uint16_t name_index = readUInt16(addr + 1);
  uint16_t desc_index = readUInt16(addr + 3);
  char *name = javaGetUtf8String(info, name_index);
  char *desc = javaGetUtf8String(info, desc_index);
  printf(" -- NameAndType name_index=%d, desc_index=%d, name=%s, desc=%s\n", name_index, desc_index, name, desc);
  free(name);
  free(desc);
}

void printClassInfo(java_class_t *info) {
  printf("version %d.%d\n", info->major_version, info->minor_version);
  printf("num tags = %d\n", info->constant_pool_count);
  for(int i=1;i<info->constant_pool_count;i++) {
    cp_info_t *cp_info = info->constant_pool[i - 1];
    uint8_t tag = cp_info->tag;

    uint16_t len = javaGetTagLength(cp_info);
    if (tag == CONSTANT_Utf8) {
      char buf[1000];
      snprintf(buf, len + 1, "%s", (char *)(cp_info + 3));
      printf("idx=%3d, tag = %2d, name=%s, len=%d, %s\n", i, tag, cp_tag_name[tag], len, buf);
    } else {
      printf("idx=%3d, tag = %2d, name=%s, len=%d\n", i, tag, cp_tag_name[tag], len);
      if (tag == CONSTANT_Methodref) {
        printMethodrefTag(info, cp_info);
      } else if (tag == CONSTANT_Class) {
        printClassTag(info, cp_info);
      } else if (tag == CONSTANT_NameAndType) {
        printNameAndTypeTag(info, cp_info);
      }
    }
  }
  printf("INTERFACES count=%d\n", info->interfaces_count);

  printf("FIELDS count=%d\n\n", info->fields_count);
  for (int i=0;i<info->fields_count;i++) {
    void *off = info->fields[i];
    uint16_t access_flags     = readUInt16(off); off += 2;
    uint16_t name_index       = readUInt16(off); off += 2;
    uint16_t descriptor_index = readUInt16(off); off += 2;
    uint16_t attributes_count = readUInt16(off); off += 2;
    for (int j=0;j<attributes_count;j++) {
      uint16_t attribute_name_index = readUInt16(off); off += 2;
      uint32_t attribute_length = readUInt32(off);     off += 4;
      off += attribute_length;
      printf("attribute_name_index=%d, attribute_length=%d\n", attribute_name_index, attribute_length);
    }
  }

  printf("METHODS count=%d\n", info->methods_count);

  for (int i=0;i<info->methods_count;i++) {
    void *off = info->methods[i];

    uint16_t access_flags     = readUInt16(off); off += 2;
    uint16_t name_index       = readUInt16(off); off += 2;
    uint16_t descriptor_index = readUInt16(off); off += 2;
    uint16_t attributes_count = readUInt16(off); off += 2;
    char *method_name = javaGetUtf8String(info, name_index);
    char *method_desc = javaGetUtf8String(info, descriptor_index);

    printf("\nMETHOD name=%s, access_flags=0x%04x, name_index=%d, descriptor=%s, attributes_count=%d\n", 
      method_name, access_flags, name_index, method_desc, attributes_count);
    free(method_name);
    free(method_desc);

    for (int j=0;j<attributes_count;j++) {
      uint16_t attr_name_index = readUInt16(off); off += 2;
      uint32_t attr_length =     readUInt32(off); off += 4;
      char *attr_name = javaGetUtf8String(info, attr_name_index);
      
      if (strcmp(attr_name, "Code")==0) {
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
        uint16_t max_stack = readUInt16(off); 
        uint16_t max_locals = readUInt16(off + 2); 
        uint32_t code_length = readUInt32(off + 4);
        uint16_t exception_table_length = readUInt32(off + 8 + code_length);

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

  for (int i=0;i<info->attributes_count;i++) {
    void *off = info->attributes[i];
    uint16_t attribute_name_index = readUInt16(off); off += 2;
    uint32_t attribute_length = readUInt32(off); off += 4;
    off += attribute_length;
    printf("attribute_name_index=%d, attribute_length=%d\n", attribute_name_index, attribute_length);
  }
}

int main(int argc, char **argv) {
  int fd;
  struct stat st;
  java_class_t* classInfo;
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
    fprintf(stderr, "%s is not a class file\n", src);
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