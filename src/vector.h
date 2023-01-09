#ifndef __VECTOR_H__
#define __VECTOR_H__

#include "scopetypes.h"

#define VECTOR_INIT_SIZE (4)

typedef struct vector {
    void **items;
    unsigned size;
    unsigned capacity;
} vector;

bool vecInit(vector *);
unsigned vecSize(vector *);
bool vecAdd(vector *, void *);
void *vecGet(vector *, unsigned);
void vecDelete(vector *);

#endif // __VECTOR_H__
