#define _GNU_SOURCE

#include "scopestdlib.h"
#include "vector.h"

bool
vecInit(vector *v) {
    v->capacity = VECTOR_INIT_SIZE;
    v->size = 0;
    v->items = scope_malloc(sizeof(void*) * v->capacity);
    return (v->items) ? TRUE : FALSE;
}

unsigned
vecSize(vector *v) {
    return v->size;
}

static bool
vecReserve(vector *v, unsigned capacity) {
    void **items = scope_realloc(v->items, sizeof(void *) * capacity);
    if (items) {
        v->items = items;
        v->capacity = capacity;
        return TRUE;
    }
    return FALSE;
}

bool
vecAdd(vector *v, void *item) {
    if (v->capacity == v->size) {
        if (vecReserve(v, v->capacity * 2) == FALSE) {
            return FALSE;
        }
    }
    v->items[v->size++] = item;
    return TRUE;
}

void *
vecGet(vector *v, unsigned index) {
    return (index < v->size) ? v->items[index] : NULL;
}

void
vecDelete(vector *v) {
    scope_free(v->items);
    v->items = 0;
    v->capacity = 0;
}
