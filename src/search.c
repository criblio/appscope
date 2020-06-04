#include <stdlib.h>
#include <string.h>
#include "scopetypes.h"
#include "search.h"

#define ASIZE 256

struct _needle_t
{
    int nlen;
    unsigned char* str;
    int bmBc[ASIZE];
};

/*
 * This is an implementation of the Horspool
 * string search algorithm.
 * ref: https://www-igm.univ-mlv.fr/~lecroq/string/node18.html
 *
 * Pre-compute the array from the needle.
 */
needle_t*
needleCreate(const char *needle_str)
{
    if (!needle_str) return NULL;

    needle_t* handle = calloc(1, sizeof(needle_t));
    if (!handle) goto failed;
    handle->nlen = strlen(needle_str);
    if (!handle->nlen) goto failed;
    handle->str = (unsigned char*)strdup(needle_str);
    if (!handle->str) goto failed;

    int i;
    for (i = 0; i < ASIZE; ++i)
        handle->bmBc[i] = handle->nlen;
    for (i = 0; i < handle->nlen - 1; ++i)
        handle->bmBc[handle->str[i]] = handle->nlen - i - 1;

    return handle;

failed:
    needleDestroy(&handle);
    return handle;
}

void
needleDestroy(needle_t **needle_ptr)
{
    if (!needle_ptr || !*needle_ptr) return;
    needle_t *needle = *needle_ptr;
    if (needle && needle->str) free(needle->str);
    if (needle) free(needle);
    *needle_ptr = NULL;
}

int
needleLen(needle_t *needle)
{
    if (!needle) return 0;
    return needle->nlen;
}

int
needleFind(needle_t *needle, char *haystack, int hlen)
{
    int j;
    unsigned char c;

    if (!needle || !haystack || hlen < 0) return -1;

    /* Searching */
    j = 0;
    while (j <= hlen - needle->nlen) {
        c = haystack[j + needle->nlen - 1];
        if (needle->str[needle->nlen - 1] == c &&
            memcmp(needle->str, haystack + j, needle->nlen - 1) == 0) {
            return j;
        }

        j += needle->bmBc[c];
    }

    return -1;
}
