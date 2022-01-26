#ifndef __STRSET_H__
#define __STRSET_H__

#include <stdbool.h>

// This was written to keep track of a set of strings which should
// not be allowed to have duplicate values.  (Originally written
// for managing metric field names).  When strings are added to the
// set, this implementation saves pointers rather than allocating
// copies of values.
//

typedef struct _strset_t strset_t;

#define DEFAULT_SET_SIZE ( 64 )

strset_t *strSetCreate(unsigned int initialCapacity);
void strSetDestroy(strset_t **);

bool strSetAdd(strset_t *, const char *);

bool strSetContains(strset_t *, const char *);

unsigned int strSetEntryCount(strset_t *);

#endif // __STRSET_H__
