#define _GNU_SOURCE
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

// constant pool entry sizes in bytes
static char cp_tag_size[] = {
    0,  // CONSTANT_Utf8
    0,
    0,
    5,  // CONSTANT_Integer,
    5,  // CONSTANT_Float,
    9,  // CONSTANT_Long,
    9,  // CONSTANT_Double,
    3,  // CONSTANT_Class,
    3,  // CONSTANT_String,
    5,  // CONSTANT_Fieldref,
    5,  // CONSTANT_Methodref, 
    5,  // CONSTANT_InterfaceMethodref,
    5,  // CONSTANT_NameAndType,
    0,
    0,
    4,  // CONSTANT_MethodHandle,
    3,  // CONSTANT_MethodType,
    5,  // CONSTANT_Dynamic,
    5,  // CONSTANT_InvokeDynamic,
    3,  // CONSTANT_Module,
    3,  // CONSTANT_Package  
};

char* 
javaGetUtf8String(java_class_t *info, int tagIndex) 
{
    unsigned char *cp = info->constant_pool[tagIndex - 1];
    uint16_t len = be16toh(*((uint16_t *)(cp + 1)));
    char *buf = malloc(len + 1);
    memcpy(buf, cp + 3, len);
    buf[len] = 0;
    return buf;
}

uint16_t 
javaGetTagLength(unsigned char *addr) 
{
    uint16_t len = 0;
    uint8_t tag = *((uint8_t *)(addr));
    if (tag == CONSTANT_Utf8) {
        len = be16toh(*((uint16_t *)(addr + 1))) + 3;
    } else {
        len = cp_tag_size[tag];
    }
    return len;
}

/*
Adds a tag to the constant pool and returns the tag's index
*/
static uint16_t 
addTag(java_class_t *info, unsigned char *tag) 
{
    uint16_t idx = info->constant_pool_count - 1;
    info->constant_pool_count++;
    info->constant_pool[idx] = tag;
    return idx + 1;
}

/*
Adds a utf8 tag to the constant pool
see: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.7
*/
static uint16_t 
addUtf8Tag(java_class_t *info, const char *str) 
{
    size_t len = strlen(str);
    size_t bufsize = len + 3;
    unsigned char *utf8Tag = malloc(bufsize);
    *((uint8_t *)utf8Tag)        = CONSTANT_Utf8;
    *((uint16_t *)(utf8Tag + 1)) = htobe16(len);
    memcpy(utf8Tag + 3, str, len);
    info->length += bufsize;
    return addTag(info, utf8Tag);
}

/*
Adds a name and type tag to the constant pool
see: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.6
*/
uint16_t 
javaAddNameAndTypeTag(java_class_t *info, const char *name, const char *desc) 
{
    uint16_t nameIndex = addUtf8Tag(info, name);
    uint16_t descIndex = addUtf8Tag(info, desc);
    size_t bufsize = 5;
    unsigned char *tag = malloc(bufsize);
    *((uint8_t *)tag)        = CONSTANT_NameAndType;
    *((uint16_t *)(tag + 1)) = htobe16(nameIndex);
    *((uint16_t *)(tag + 3)) = htobe16(descIndex);
    info->length += bufsize;
    return addTag(info, tag);
}

/*
Adds a method ref tag to the contant pool
see: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.2
*/
uint16_t 
javaAddMethodRefTag(java_class_t *info, uint16_t classIndex, uint16_t nameAndTypeIndex) 
{
    size_t bufsize = 5;
    unsigned char *tag = malloc(bufsize);
    *((uint8_t *)tag)        = CONSTANT_Methodref;
    *((uint16_t *)(tag + 1)) = htobe16(classIndex);
    *((uint16_t *)(tag + 3)) = htobe16(nameAndTypeIndex);
    info->length += bufsize;
    return addTag(info, tag);
}

/*
Adds a string tag to the constant pool
see: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.3
*/
uint16_t 
javaAddStringTag(java_class_t *info, const char* str)
{
    uint16_t idx = addUtf8Tag(info, str);
    unsigned char *tag = malloc(3);
    *((uint8_t *)tag)        = CONSTANT_String;
    *((uint16_t *)(tag + 1)) = htobe16(idx);
    return addTag(info, tag);
}

static uint32_t 
getAttributesLength(unsigned char *addr) 
{
    uint16_t attr_count = be16toh(*((uint16_t *)addr));
    unsigned char *off = addr + 2;
    int j;
    for (j=0;j<attr_count;j++) {
        uint32_t attr_length = be32toh(*((uint32_t *)(off + 2)));
        off += attr_length + 6;
    }
    return (uint32_t)(off - addr);
}

uint32_t 
javaGetMethodLength(unsigned char *addr) 
{
    return getAttributesLength(addr + 6) + 6;
}

static unsigned char *
getCodeAttributeAddress(java_class_t *info, unsigned char *method) 
{
    uint16_t attributes_count = be16toh(*((uint16_t *)(method + 6)));
    
    unsigned char *code = NULL;
    unsigned char *off = method + 8;
    int j;
    for (j=0;j<attributes_count && code == NULL;j++) {
        uint16_t attr_name_index = be16toh(*((uint16_t *)off));
        uint32_t attr_length     = be32toh(*((uint32_t *)(off + 2)));
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

int 
javaFindClassIndex(java_class_t *info, const char *className) 
{
    int idx = -1;
    int i;
    for(i=1;i<info->constant_pool_count;i++) {
        unsigned char *cp_info = info->constant_pool[i - 1];
        uint8_t tag = *((uint8_t *)cp_info);
        if(tag == CONSTANT_Class) {
            uint16_t name_index = be16toh(*((uint16_t *)(cp_info + 1)));
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

int 
javaFindMethodIndex(java_class_t *info, const char *method, const char *signature) 
{
    int idx = -1;
    int i;
    for (i=0;i<info->methods_count;i++) {
        unsigned char *addr = info->methods[i];
        uint16_t name_index       = be16toh(*((uint16_t *)(addr + 2)));
        uint16_t descriptor_index = be16toh(*((uint16_t *)(addr + 4)));
        
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

void 
javaCopyMethod(java_class_t *info, unsigned char *method, const char *newName) 
{
    uint32_t len = javaGetMethodLength(method);
    unsigned char *dest = malloc(len);
    memcpy(dest, method, len);
    uint16_t nameIndex = addUtf8Tag(info, newName);
    *((uint16_t *)(dest + 2)) = htobe16(nameIndex);
    info->methods_count++;
    info->methods[info->methods_count - 1] = dest;
    info->length += len;
}

void 
javaConvertMethodToNative(java_class_t *info, int methodIndex) 
{
    unsigned char *methodAddr = info->methods[methodIndex];
    uint32_t len   = javaGetMethodLength(methodAddr);
    size_t bufsize = 8;
    unsigned char *addr        = malloc(bufsize);
    info->methods[methodIndex] = addr;

    memcpy(addr, methodAddr, bufsize);

    uint16_t accessFlags      = be16toh(*((uint16_t *)methodAddr));
    uint16_t attributesCount  = 0;

    *((uint16_t *)addr)       = htobe16(accessFlags | ACC_NATIVE);
    *((uint16_t *)(addr + 6)) = htobe16(attributesCount);

    info->length += bufsize - len;
}

void 
javaAddMethod(java_class_t *info, const char* name, const char* descriptor, 
              uint16_t accessFlags, uint16_t maxStack, uint16_t maxLocals, uint8_t *code, uint32_t codeLen) 
{
    size_t codeAttrLen = 12 + codeLen;
    /*
    total buf size = method info (8 bytes) + 
                    attribute index (2 bytes) + 
                    attribute length (4 bytes) + 
                    length of the buffer which holds the code attribute
    
    Code attribute specs : https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.7.3
    Method info specs: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.6
    */
    size_t bufsize = 8 + 2 + 4 + codeAttrLen;  

    if (code == NULL) {
        codeAttrLen = 0;
        bufsize = 8;
    }

    unsigned char *addr = malloc(bufsize);
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
        *((uint16_t *)addr) = htobe16(codeNameIndex);  addr += 2;
        *((uint32_t *)addr) = htobe32(codeAttrLen);    addr += 4;
        *((uint16_t *)addr) = htobe16(maxStack);       addr += 2;
        *((uint16_t *)addr) = htobe16(maxLocals);      addr += 2;
        *((uint32_t *)addr) = htobe32(codeLen);        addr += 4;
        memcpy(addr, code, codeLen);                   addr += codeLen;
        *((uint16_t *)addr) = htobe16(exceptionLen);   addr += 2;
        *((uint16_t *)addr) = htobe16(codeAttrCount);    
    }

    info->length += bufsize;
}

void
javaAddField(java_class_t *info, const char* name, const char* descriptor, uint16_t accessFlags)
{
    uint16_t nameIndex       = addUtf8Tag(info, name);
    uint16_t descriptorIndex = addUtf8Tag(info, descriptor);
    uint16_t attrCount = 0;

    size_t bufsize = 4 * 2;
    unsigned char *buf = malloc(bufsize);
    info->fields_count++;
    info->fields[info->fields_count - 1] = buf;
    info->length += bufsize;

    *((uint16_t *)buf) = htobe16(accessFlags);      buf += 2;
    *((uint16_t *)buf) = htobe16(nameIndex);        buf += 2;
    *((uint16_t *)buf) = htobe16(descriptorIndex);  buf += 2;
    *((uint16_t *)buf) = htobe16(attrCount);        buf += 2;
}

void 
javaInjectCode(java_class_t *classInfo, unsigned char *method, uint8_t *code, size_t len) 
{
    unsigned char *codeAttr = getCodeAttributeAddress(classInfo, method);
    uint32_t codeLen = be32toh(*((uint32_t *)(codeAttr + 10)));
    memset(codeAttr + 14, 0, codeLen - 1);
    memcpy(codeAttr + 14, code, len);
} 

void 
javaWriteClass(unsigned char *dest, java_class_t *info) 
{
    unsigned char *addr = dest;
    memcpy(addr, magic, sizeof(magic));                         addr += sizeof(magic);
    *((uint16_t *)addr) = htobe16(info->minor_version);         addr += 2;
    *((uint16_t *)addr) = htobe16(info->major_version);         addr += 2;
    *((uint16_t *)addr) = htobe16(info->constant_pool_count);   addr += 2;
    int i;
    for(i=0;i<info->constant_pool_count - 1;i++) {
        unsigned char *cp = info->constant_pool[i];
        unsigned char tag = *((unsigned char *)cp);
        uint16_t size = javaGetTagLength(cp);
        memcpy(addr, cp, size);
        addr += size;
        if (tag == CONSTANT_Double || tag == CONSTANT_Long) {
            //this is what JVM spec says here: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.5
            //If a CONSTANT_Long_info or CONSTANT_Double_info structure is the entry at index n in the constant_pool table, 
            //then the next usable entry in the table is located at index n+2. The constant_pool index n+1 must be valid but is 
            //considered unusable.
            i++;
        }
    }
    *((uint16_t *)addr) = htobe16(info->access_flags);          addr += 2;
    *((uint16_t *)addr) = htobe16(info->this_class);            addr += 2;
    *((uint16_t *)addr) = htobe16(info->super_class);           addr += 2;
    *((uint16_t *)addr) = htobe16(info->interfaces_count);      addr += 2;
    memcpy(addr, info->interfaces, info->interfaces_count * 2); addr += info->interfaces_count * 2;
    //write fields
    *((uint16_t *)addr) = htobe16(info->fields_count); addr += 2;
    for (i=0;i<info->fields_count;i++) {
        unsigned char *field = info->fields[i];
        uint32_t size = getAttributesLength(field + 6) + 6;
        memcpy(addr, info->fields[i], size);
        addr += size;
    }
    *((uint16_t *)addr) = htobe16(info->methods_count); addr += 2;
    //write methods
    for (i=0;i<info->methods_count;i++) {
        unsigned char *method = info->methods[i];
        uint32_t size = javaGetMethodLength(method);
        memcpy(addr, info->methods[i], size);
        addr += size;
    }
    //write attributes
    *((uint16_t *)addr) = htobe16(info->attributes_count); addr += 2;
    for (i=0;i<info->attributes_count;i++) {
        unsigned char *attr = info->attributes[i];
        uint32_t attribute_length = be32toh(*((uint32_t *)(attr + 2)));
        size_t size = attribute_length + 6;
        memcpy(addr, info->attributes[i], size);
        addr += size;
    }
}

java_class_t* 
javaReadClass(const unsigned char* classData) 
{
    java_class_t *classInfo = malloc(sizeof(java_class_t));
    if (memcmp(classData, magic, sizeof(magic)) != 0) {
        free(classInfo);
        return NULL;
    }
    unsigned char *addr = (unsigned char *)classData;
    unsigned char *off = addr + sizeof(magic);
    classInfo->minor_version        = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->major_version        = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->constant_pool_count  = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->_constant_pool_count = classInfo->constant_pool_count;
    //allocate memory for existing constant pool enties and make a room for up to 100 new entries
    classInfo->constant_pool        = (unsigned char **) calloc(100 + (classInfo->constant_pool_count - 1), sizeof(unsigned char *));
    int i;
    for(i=1;i<classInfo->constant_pool_count;i++) {
        classInfo->constant_pool[i - 1] = (unsigned char *)off;
        unsigned char tag = *((unsigned char *)off);
        if (tag == CONSTANT_Double || tag == CONSTANT_Long) {
            //this is what JVM spec says here: https://docs.oracle.com/javase/specs/jvms/se14/html/jvms-4.html#jvms-4.4.5
            //If a CONSTANT_Long_info or CONSTANT_Double_info structure is the entry at index n in the constant_pool table, 
            //then the next usable entry in the table is located at index n+2. The constant_pool index n+1 must be valid but is 
            //considered unusable.
            i++;
        }
        off += javaGetTagLength(off);
    }

    classInfo->access_flags         = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->this_class           = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->super_class          = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->interfaces_count     = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->interfaces           = (uint16_t *)off;
    off += classInfo->interfaces_count * 2;

    //read fields
    classInfo->fields_count         = be16toh(*((uint16_t *)off)); off += 2;
    //allocate memory for existing fields and make a room for up to 100 new fields
    classInfo->fields               = (unsigned char **) calloc(100 + classInfo->fields_count, sizeof(unsigned char *));
    for (i=0;i<classInfo->fields_count;i++) {
        classInfo->fields[i] = off;
        off += getAttributesLength(off + 6) + 6;
    }

    //read methods
    classInfo->methods_count        = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->_methods_count       = classInfo->methods_count;
    //allocate memory for existing methods and make a room for up to 100 new methods
    classInfo->methods              = (unsigned char **) calloc(100 + classInfo->methods_count, sizeof(unsigned char *));
    for (i=0;i<classInfo->methods_count;i++) {
        classInfo->methods[i] = off;
        off += javaGetMethodLength(off);
    }

    //read attributes
    classInfo->attributes_count     = be16toh(*((uint16_t *)off)); off += 2;
    classInfo->attributes           = (unsigned char **) calloc(classInfo->attributes_count, sizeof(unsigned char *));
    for (i=0;i<classInfo->attributes_count;i++) {
        classInfo->attributes[i] = off;
        uint32_t attribute_length = be32toh(*((uint32_t *)(off + 2)));
        off += attribute_length + 6;
    }
    classInfo->length = off - addr;
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
