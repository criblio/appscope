#ifndef __JAVABCI_H__
#define __JAVABCI_H__
#include <stdint.h>

// Set of simple utilities for Java byte code instrumentation based on 
// https://docs.oracle.com/javase/specs/jvms/se14/html/index.html

// constant pool tags
#define CONSTANT_Utf8                 1
#define CONSTANT_Integer              3
#define CONSTANT_Float                4
#define CONSTANT_Long                 5
#define CONSTANT_Double               6
#define CONSTANT_Class                7
#define CONSTANT_String               8
#define CONSTANT_Fieldref             9
#define CONSTANT_Methodref            10 
#define CONSTANT_InterfaceMethodref   11
#define CONSTANT_NameAndType          12
#define CONSTANT_MethodHandle         15
#define CONSTANT_MethodType           16
#define CONSTANT_Dynamic              17
#define CONSTANT_InvokeDynamic        18
#define CONSTANT_Module               19
#define CONSTANT_Package              20

#define OP_ALOAD_0                    0x2a
#define OP_ALOAD_1                    0x2b
#define OP_INVOKEVIRTUAL              0xb6

// access flags
#define ACC_PUBLIC                    0x0001
#define ACC_PRIVATE                   0x0002
#define ACC_PROTECTED                 0x0004
#define ACC_STATIC                    0x0008
#define ACC_FINAL                     0x0010
#define ACC_SYNCHRONIZED              0x0020
#define ACC_BRIDGE                    0x0040
#define ACC_VARARGS                   0x0080
#define ACC_NATIVE                    0x0100
#define ACC_ABSTRACT                  0x0400
#define ACC_STRICT                    0x0800
#define ACC_SYNTHETIC                 0x1000


typedef struct {
  uint8_t        magic[4];
	uint16_t       minor_version;
	uint16_t       major_version;
	uint16_t       constant_pool_count;
  unsigned char  **constant_pool;
  uint16_t       access_flags;
  uint16_t       this_class;
  uint16_t       super_class;
  uint16_t       interfaces_count;
  uint16_t       *interfaces;
  uint16_t       fields_count;
  unsigned char  **fields;
  uint16_t       methods_count;
  unsigned char  **methods;
  uint16_t       attributes_count;
  unsigned char  **attributes;
  uint32_t       length;

  uint16_t       _constant_pool_count;
  uint16_t       _methods_count;
} java_class_t;


java_class_t*   javaReadClass(const unsigned char* classData);
void            javaWriteClass(unsigned char *dest, java_class_t *info);
void            javaDestroy(java_class_t **classInfo);

int             javaFindClassIndex(java_class_t *info, const char *className);
int             javaFindMethodIndex(java_class_t *info, const char *method, const char *signature);
void            javaCopyMethod(java_class_t *info, unsigned char *method, const char *newName);
void            javaAddMethod(java_class_t *info, const char* name, const char* descriptor, 
                              uint16_t accessFlags, uint16_t maxStack, uint16_t maxLocals, 
                              uint8_t *code, uint32_t codeLen);
void            javaInjectCode(java_class_t *classInfo, unsigned char *method, uint8_t *code, size_t len);

uint16_t        javaAddStringTag(java_class_t *info, const char* str);
uint16_t        javaAddNameAndTypeTag(java_class_t *info, const char *name, const char *desc);
uint16_t        javaAddMethodRefTag(java_class_t *info, uint16_t classIndex, uint16_t nameAndTypeIndex);

char*           javaGetUtf8String(java_class_t *info, int tagIndex);
uint16_t        javaGetTagLength(unsigned char *addr);
uint32_t        javaGetMethodLength(unsigned char *addr);
void            javaConvertMethodToNative(java_class_t *info, int methodIndex);

#endif // __JAVABCI_H__