#include "scopetypes.h"
#include "strset.h"
#include <stdlib.h>
#include <string.h>

typedef struct _strset_t {
    const char **str;
    unsigned int count;
    unsigned int alloc;
} strset_t;

strset_t *
strSetCreate(unsigned int initialCapacity)
{
    strset_t *set = calloc(1, sizeof(*set));
    const char **str = calloc(initialCapacity, sizeof(char *));
    if (!set || !str) goto err;

    set->str = str;
    set->count = 0;
    set->alloc = initialCapacity;

    return set;

err:
    if (set) free(set);
    if (str) free(str);
    return NULL;
}

void
strSetDestroy(strset_t **set_ptr)
{
    if (!set_ptr || !*set_ptr) return;

    strset_t *set = *set_ptr;
    free(set->str);
    free(set);
    *set_ptr = NULL;
}

bool
strSetAdd(strset_t *set, const char *str)
{
    if (!set || !str) return FALSE;

    // enforce that no dup values are allowed
    if (strSetContains(set, str)) return FALSE;

    // grow if needed
    if (set->count >= set->alloc) {
        unsigned int new_alloc = (set->alloc) ? set->alloc * 4 : 2;
        const char **new_str = realloc(set->str, sizeof(char *) * new_alloc);
        if (!new_str) {
            return FALSE;
        }
        memset(&new_str[set->count], 0, sizeof(char*) * (new_alloc-set->count));
        set->str = new_str;
        set->alloc = new_alloc;
    }

    // Add str to the set
    set->str[set->count++] = str;
    return TRUE;
}

bool
strSetContains(strset_t *set, const char *str)
{
    if (!set || !str) return FALSE;

    unsigned int i;
    for (i=0; i < set->count; i++) {
        if (strcmp(set->str[i], str) == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

unsigned int
strSetEntryCount(strset_t *set)
{
    return (set) ? set->count : 0;
}
