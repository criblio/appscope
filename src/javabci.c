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

static char magic[] = { 0xca, 0xfe, 0xba, 0xbe };

static char cp_tag_size[] = {
  5,  // 3,
  5,  // 4,
  9,  // 5,
  9,  // 6,
  3,  // 7,
  3,  // 8,
  5,  // 9,
  5,  // 10, 
  5,  // 11,
  5,  // 12,
  0,
  0,
  4,  // 15,
  3,  // 16,
  5,  // 17,
  5,  // 18,
  3,  // 19,
  3,  // 20  
};

uint8_t readUInt8(void *addr) {
  return *((uint8_t *)(addr));
}

uint16_t readUInt16(void *addr) {
  return be16toh(*((uint16_t *)addr));
}

uint32_t readUInt32(void *addr) {
  return be32toh(*((uint32_t *)addr));
}

char* javaGetUtf8String(java_class_t *info, int tagIndex) {
  cp_info_t *cp = info->constant_pool[tagIndex - 1];
  uint16_t len = readUInt16(cp + 1);
  char *buf = malloc(len + 1);
  memcpy(buf, cp + 3, len);
  buf[len] = 0;
  return buf;
}

uint16_t javaGetTagLength(void *addr) {
  uint16_t len = 0;
  uint8_t tag = readUInt8(addr);
  if (tag == CONSTANT_Utf8) {
    len = readUInt16(addr + 1) + 3;
  } else {
    len = cp_tag_size[tag - 3];
  }
  return len;
}

uint16_t addTag(java_class_t *info, cp_info_t *tag) {
  uint16_t idx = info->constant_pool_count - 1;
  info->constant_pool_count++;
  info->constant_pool[idx] = tag;
  return idx + 1;
}

uint16_t addUtf8Tag(java_class_t *info, const char *str) {
  size_t len = strlen(str);
  size_t bufsize = len + 3;
  cp_info_t *utf8Tag = malloc(bufsize);
  utf8Tag->tag = CONSTANT_Utf8;
  *((uint16_t *)(utf8Tag + 1)) = htobe16(len);
  memcpy(utf8Tag + 3, str, len);
  info->length += bufsize;
  return addTag(info, utf8Tag);
}

uint16_t javaAddNameAndTypeTag(java_class_t *info, const char *name, const char *desc) {
  uint16_t nameIndex = addUtf8Tag(info, name);
  uint16_t descIndex = addUtf8Tag(info, desc);
  size_t bufsize = 5;
  cp_info_t *tag = malloc(bufsize);
  tag->tag = CONSTANT_NameAndType;
  *((uint16_t *)(tag + 1)) = htobe16(nameIndex);
  *((uint16_t *)(tag + 3)) = htobe16(descIndex);
  info->length += bufsize;
  return addTag(info, tag);
}

uint16_t javaAddMethodRefTag(java_class_t *info, uint16_t classIndex, uint16_t nameAndTypeIndex) {
  size_t bufsize = 5;
  cp_info_t *tag = malloc(bufsize);
  tag->tag = CONSTANT_Methodref;
  *((uint16_t *)(tag + 1)) = htobe16(classIndex);
  *((uint16_t *)(tag + 3)) = htobe16(nameAndTypeIndex);
  info->length += bufsize;
  return addTag(info, tag);
}

uint32_t getAttributesLength(void *addr) {
  uint16_t attr_count = readUInt16(addr);
  void *off = addr + 2;
  for (int j=0;j<attr_count;j++) {
    uint32_t attr_length = readUInt32(off + 2);
    off += attr_length + 6;
  }
  return (uint32_t)(off - addr);
}

uint32_t javaGetMethodLength(void *addr) {
  return getAttributesLength(addr + 6) + 6;
}

void * getCodeAttributeAddress(java_class_t *info, void *method) {
  uint16_t attributes_count = readUInt16(method + 6);
  
  void *code = NULL;
  void *off = method + 8;
  for (int j=0;j<attributes_count && code == NULL;j++) {
    uint16_t attr_name_index = readUInt16(off);
    uint32_t attr_length     = readUInt32(off + 2);
    char *attr_name          = javaGetUtf8String(info, attr_name_index);
    if (strcmp(attr_name, "Code")==0) {
      code = off;
    }
    free(attr_name);
    if (code != NULL) break;
    off += attr_length + 6;
  }
  return code;
}

uint16_t addStringTag(java_class_t *info, const char* str) {
  uint16_t idx = addUtf8Tag(info, str);
  cp_info_t *stringTag = malloc(3);
  stringTag->tag = CONSTANT_String;
  *((uint16_t *)(stringTag + 1)) = htobe16(idx);
  return addTag(info, stringTag);
}

int javaFindClassIndex(java_class_t *info, const char *className) {
  int idx = -1;
  for(int i=1;i<info->constant_pool_count;i++) {
    cp_info_t *cp_info = info->constant_pool[i - 1];
    uint8_t tag = cp_info->tag;
    if(tag == CONSTANT_Class) {
      uint16_t name_index = readUInt16(cp_info + 1);
      char *name = javaGetUtf8String(info, name_index);
      if (strcmp(name, className) == 0) {
        idx = i;
      }
      free(name);
    }
    if (idx != -1) break;
  }
  return idx;
}

int javaFindMethodIndex(java_class_t *info, const char *method, const char *signature) {
  int idx = -1;
  for (int i=0;i<info->methods_count;i++) {
    void *addr = info->methods[i];
    uint16_t name_index       = readUInt16(addr + 2);
    uint16_t descriptor_index = readUInt16(addr + 4);
    
    char *method_name = javaGetUtf8String(info, name_index);
    char *method_desc = javaGetUtf8String(info, descriptor_index);

    if (strcmp(method, method_name) == 0 && strcmp(signature, method_desc) == 0) {
      idx = i;
    }
    free(method_name);
    free(method_desc);
    if (idx != -1) break;
  }
  return idx;
}

void javaCopyMethod(java_class_t *info, void *method, const char *newName) {
  uint32_t len = javaGetMethodLength(method);
  void *dest = malloc(len);
  memcpy(dest, method, len);
  uint16_t nameIndex = addUtf8Tag(info, newName);
   *((uint16_t *)(dest + 2)) = htobe16(nameIndex);
  info->methods_count++;
  info->methods[info->methods_count - 1] = dest;
  info->length += len;
}

void javaConvertMethodToNative(java_class_t *info, int methodIndex) {
  void *methodAddr = info->methods[methodIndex];
  uint32_t len     = javaGetMethodLength(methodAddr);
  size_t bufsize   = 8;
  void *addr       = malloc(bufsize);
  info->methods[methodIndex] = addr;

  memcpy(addr, methodAddr, bufsize);

  uint16_t accessFlags     = readUInt16(methodAddr); 
  uint16_t attributesCount = 0;

  *((uint16_t *)addr)        = htobe16(accessFlags | ACC_NATIVE | ACC_PUBLIC);
  *((uint16_t *)(addr + 6))  = htobe16(attributesCount);

  info->length += bufsize - len;
}

void javaAddMethod(java_class_t *info, const char* name, const char* descriptor,  uint16_t accessFlags, 
  uint16_t maxStack, uint16_t maxLocals, uint8_t *code, uint32_t codeLen) {

  size_t codeAttrLen = 12 + codeLen;
  size_t bufsize = 8 + 6 + codeAttrLen;

  if (code == NULL) {
    codeAttrLen = 0;
    bufsize = 8;
  }

  void *addr = malloc(bufsize);
  info->methods_count++;
  info->methods[info->methods_count - 1] = addr;

  uint16_t nameIndex = addUtf8Tag(info, name);
  uint16_t descriptorIndex = addUtf8Tag(info, descriptor);
  uint16_t attrCount = 0;
  uint16_t codeNameIndex = 0;
  
  if (code != NULL) {
    attrCount = 1;
    codeNameIndex = addUtf8Tag(info, "Code");
  }
  
  //write method info
  *((uint16_t *)addr) = htobe16(accessFlags);      addr += 2;
  *((uint16_t *)addr) = htobe16(nameIndex);        addr += 2;
  *((uint16_t *)addr) = htobe16(descriptorIndex);  addr += 2;
  *((uint16_t *)addr) = htobe16(attrCount);        addr += 2;

  if (code != NULL) {
    //write code attribute
    uint16_t exceptionLen = 0;
    uint16_t codeAttrCount = 0;
    *((uint16_t *)addr) = htobe16(codeNameIndex);    addr += 2;
    *((uint32_t *)addr) = htobe32(codeAttrLen);      addr += 4;
    *((uint16_t *)addr) = htobe16(maxStack);         addr += 2;
    *((uint16_t *)addr) = htobe16(maxLocals);        addr += 2;
    *((uint32_t *)addr) = htobe32(codeLen);          addr += 4;
    memcpy(addr, code, codeLen);                     addr += codeLen;
    *((uint16_t *)addr) = htobe16(exceptionLen);     addr += 2;
    *((uint16_t *)addr) = htobe16(codeAttrCount);    
  }

  info->length += bufsize;
}

void javaInjectCode(java_class_t *classInfo, void *method, uint8_t *code, size_t len) {
  void *codeAttr = getCodeAttributeAddress(classInfo, method);
  uint32_t codeLen = readUInt32(codeAttr + 10);
  memset(codeAttr + 14, 0, codeLen - 1);
  memcpy(codeAttr + 14, code, len);
} 

void javaWriteClass(void *dest, java_class_t *info) {
  void *addr = dest;
  memcpy(addr, magic, 4);                                     addr += 4;
  *((uint16_t *)addr) = htobe16(info->minor_version);         addr += 2;
  *((uint16_t *)addr) = htobe16(info->major_version);         addr += 2;
  *((uint16_t *)addr) = htobe16(info->constant_pool_count);   addr += 2;
  for(int i=0;i<info->constant_pool_count - 1;i++) {
    cp_info_t *cp = info->constant_pool[i];
    uint16_t size = javaGetTagLength(cp);
    memcpy(addr, cp, size);
    addr += size;
  }
  *((uint16_t *)addr) = htobe16(info->access_flags);          addr += 2;
  *((uint16_t *)addr) = htobe16(info->this_class);            addr += 2;
  *((uint16_t *)addr) = htobe16(info->super_class);           addr += 2;
  *((uint16_t *)addr) = htobe16(info->interfaces_count);      addr += 2;
  memcpy(addr, info->interfaces, info->interfaces_count * 2); addr += info->interfaces_count * 2;

  *((uint16_t *)addr) = htobe16(info->fields_count); addr += 2;
  for (int i=0;i<info->fields_count;i++) {
    void *field = info->fields[i];
    uint32_t size = getAttributesLength(field + 6) + 6;
    memcpy(addr, info->fields[i], size);
    addr += size;
  }
  *((uint16_t *)addr) = htobe16(info->methods_count); addr += 2;
  //write methods
  for (int i=0;i<info->methods_count;i++) {
    void *method = info->methods[i];
    uint32_t size = javaGetMethodLength(method);
    memcpy(addr, info->methods[i], size);
    addr += size;
  }
  //write attributes
  *((uint16_t *)addr) = htobe16(info->attributes_count); addr += 2;
  for (int i=0;i<info->attributes_count;i++) {
    void *attr = info->attributes[i];
    uint32_t attribute_length = readUInt32(attr + 2); 
    size_t size = attribute_length + 6;
    memcpy(addr, info->attributes[i], size);
    addr += size;
  }
}

java_class_t* javaReadClass(void *buf) {
  java_class_t *classInfo = malloc(sizeof(java_class_t));
  if (memcmp(buf, magic, sizeof(magic)) != 0) {
    return NULL;
  }
  void *off = buf + 4;
  classInfo->minor_version       = readUInt16(off); off += 2;
  classInfo->major_version       = readUInt16(off); off += 2;
  classInfo->constant_pool_count = readUInt16(off); off += 2;
  classInfo->_constant_pool_count = classInfo->constant_pool_count;
  classInfo->constant_pool       = (cp_info_t **) calloc(100 + (classInfo->constant_pool_count - 1), sizeof(cp_info_t *));
  for(int i=1;i<classInfo->constant_pool_count;i++) {
    classInfo->constant_pool[i - 1] = (cp_info_t *)off;
    off += javaGetTagLength(off);
  }

  classInfo->access_flags        = readUInt16(off); off += 2;
  classInfo->this_class          = readUInt16(off); off += 2;
  classInfo->super_class         = readUInt16(off); off += 2;
  classInfo->interfaces_count    = readUInt16(off); off += 2;
  classInfo->interfaces          = (uint16_t *)off;
  off += classInfo->interfaces_count * 2;

  //fields
  classInfo->fields_count        = readUInt16(off); off += 2;
  classInfo->fields              = (void **) calloc(classInfo->fields_count, sizeof(void *));
  for (int i=0;i<classInfo->fields_count;i++) {
    classInfo->fields[i] = off;
    off += getAttributesLength(off + 6) + 6;
  }

  classInfo->methods_count       = readUInt16(off); off += 2;
  classInfo->_methods_count      = classInfo->methods_count;
  classInfo->methods             = (void **) calloc(100 + classInfo->methods_count, sizeof(void *));
  for (int i=0;i<classInfo->methods_count;i++) {
    classInfo->methods[i] = off;
    off += javaGetMethodLength(off);
  }

  classInfo->attributes_count    = readUInt16(off); off += 2;
  classInfo->attributes          = (void **) calloc(classInfo->attributes_count, sizeof(void *));

  for (int i=0;i<classInfo->attributes_count;i++) {
    classInfo->attributes[i] = off;
    uint32_t attribute_length = readUInt32(off + 2);
    off += attribute_length + 6;
  }
  classInfo->length = off - buf;
  return classInfo;
}

void javaDestroy(java_class_t **classInfo) {
  int i;
  for(i = (*classInfo)->_constant_pool_count - 1;i<(*classInfo)->constant_pool_count - 1;i++) {
    free((*classInfo)->constant_pool[i]);
  }
  for(i = (*classInfo)->_methods_count;i<(*classInfo)->methods_count;i++) {
    free((*classInfo)->methods[i]);
  }
  free((*classInfo)->constant_pool);
  free((*classInfo)->fields);
  free((*classInfo)->methods);
  free((*classInfo)->attributes);
  free(*classInfo);
  *classInfo = NULL;
}
