#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include "scopetypes.h"
#include "search.h"
#include "scopestdlib.h"

#define ASIZE 256

struct _search_t
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
 * Pre-compute the array from the input_str.
 */
search_t*
searchComp(const char *input_str)
{
    if (!input_str) return NULL;

    search_t* handle = scope_calloc(1, sizeof(search_t));
    if (!handle) goto failed;
    handle->nlen = scope_strlen(input_str);
    if (!handle->nlen) goto failed;
    handle->str = (unsigned char*)scope_strdup(input_str);
    if (!handle->str) goto failed;

    int i;
    for (i = 0; i < ASIZE; ++i)
        handle->bmBc[i] = handle->nlen;
    for (i = 0; i < handle->nlen - 1; ++i)
        handle->bmBc[handle->str[i]] = handle->nlen - i - 1;

    return handle;

failed:
    searchFree(&handle);
    return handle;
}

void
searchFree(search_t **handle_ptr)
{
    if (!handle_ptr || !*handle_ptr) return;
    search_t *handle = *handle_ptr;
    if (handle && handle->str) scope_free(handle->str);
    if (handle) scope_free(handle);
    *handle_ptr = NULL;
}

int
searchLen(search_t *handle)
{
    if (!handle) return 0;
    return handle->nlen;
}

int
searchExec(search_t *handle, char *haystack, int hlen)
{
    int j;
    unsigned char c;

    if (!handle || !haystack || hlen < 0) return -1;

    /* Searching */
    j = 0;
    while (j <= hlen - handle->nlen) {
        c = haystack[j + handle->nlen - 1];
        if (handle->str[handle->nlen - 1] == c &&
            scope_memcmp(handle->str, haystack + j, handle->nlen - 1) == 0) {
            return j;
        }

        j += handle->bmBc[c];
    }

    return -1;
}
