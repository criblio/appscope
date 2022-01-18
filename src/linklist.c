#define _GNU_SOURCE
#include <stdlib.h>
#include "atomic.h"
#include "linklist.h"
#include "scopestdlib.h"

#define TRUE 1
#define FALSE 0

typedef struct _list_element_t {
    list_key_t               key;
    void                    *data;
    struct _list_element_t  *next;
} list_element_t;

typedef struct _list_t {
    delete_fn_t              delete_fn;
    list_element_t          *head;
} list_t;


// This realtime-safe linked list implementation is based on
// the algorithm by Timothy L. Harris in
// "A Pragmatic Implementation of Non-Blocking Linked-Lists".
//
// https://www.cl.cam.ac.uk/research/srg/netos/papers/2001-caslists.pdf
//
// It is a singly linked list, ordered by key, that prohibits
// duplicate key values.
//
// This particular algorithm was chosen because it is known to
// handle 1) concurrent deletes and 2) concurrent inserts and deletes.
//
// The lstDestoy() function is something I added, and have tried to
// make it realtime-safe as well, but this needs more evaluation
// if lists are going to be deleted while they are in use by other
// threads.
//


static inline bool
CAS(list_element_t **ptr, list_element_t *oldval, list_element_t* newval)
{
    return atomicCasU64((uint64_t*)ptr, (uint64_t)oldval, (uint64_t)newval);
}

static int
is_marked_reference(list_element_t *ptr)
{
    return (uintptr_t)ptr & 0x1;
}

static list_element_t*
get_unmarked_reference(list_element_t *ptr)
{
    return (list_element_t*)((uintptr_t)ptr & ~0x1);
}

static list_element_t*
get_marked_reference(list_element_t *ptr)
{
    return (list_element_t*)((uintptr_t)ptr | 0x1);
}

static list_element_t *
search (list_t *list, list_key_t search_key, list_element_t **left_node)
{
    if (!list || !left_node) return NULL;

    list_element_t *left_node_next, *right_node;

    list_element_t *head = list->head;
    if (!head) return NULL;

    *left_node = head;
    left_node_next = head->next;

search_again:
    do {
        list_element_t *t = head;
        list_element_t *t_next = head->next;

        /* 1: Find left_node and right_node */
        do {
            if (!is_marked_reference(t_next)) {
                (*left_node) = t;
                left_node_next = t_next;
            }
            t = get_unmarked_reference(t_next);
            if (!t) break; // at the end
            t_next = t->next;
        } while (is_marked_reference(t_next) || (t->key<search_key)); /*B1*/
        right_node = t;

        /* 2: Check nodes are adjacent */
        if (left_node_next == right_node) {
            if ((right_node) && is_marked_reference(right_node->next)) {
                goto search_again; /*G1*/
            } else {
                return right_node; /*R1*/
            }
        }

        /* 3: Remove one or more marked nodes */
        if (CAS (&(*left_node)->next, left_node_next, right_node)) { /*C1*/
            if ((right_node) && is_marked_reference(right_node->next)) {
                goto search_again; /*G2*/
            } else {
                return right_node; /*R2*/
            }
        }
    } while (TRUE);

    return NULL;
}

list_t*
lstCreate(delete_fn_t delete_fn)
{
    list_t *list = scope_calloc(1, sizeof(list_t));
    list_element_t* head = scope_calloc(1, sizeof(list_element_t));
    if (!list || !head) {
        if (list) scope_free(list);
        if (head) scope_free(head);
        return NULL;
    }
    list->head = head;
    list->delete_fn = delete_fn;
    return list;
}

int
lstInsert (list_t *list, list_key_t key, void* data)
{
    if (!list) return FALSE;

    list_element_t *new_node = scope_calloc(1, sizeof(list_element_t));
    if (!new_node) return FALSE;
    new_node->key = key;
    new_node->data = data;

    list_element_t *right_node, *left_node;

    do {
        right_node = search (list, key, &left_node);
        if ((right_node) && (right_node->key == key)) { /*T1*/
            scope_free(new_node);
            return FALSE;
        }
        new_node->next = right_node;
        if (CAS (&(left_node->next), right_node, new_node)) { /*C2*/
            return TRUE;
        }
    } while (TRUE); /*B3*/

    return FALSE;
}

int
lstDelete (list_t *list, list_key_t search_key)
{
    if (!list) return FALSE;

    list_element_t *right_node, *right_node_next, *left_node;

    do {
        right_node = search (list, search_key, &left_node);
        if ((!right_node) || (right_node->key != search_key)) { /*T1*/
            return FALSE;
        }
        right_node_next = right_node->next;
        if (!is_marked_reference(right_node_next)) {
            if (CAS (&(right_node->next), /*C3*/
                right_node_next, get_marked_reference (right_node_next))) {
                break;
            }
        }
    } while (TRUE); /*B4*/
    if (!CAS (&(left_node->next), right_node, right_node_next)) { /*C4*/
        right_node = search (list, right_node->key, &left_node);
    }

    // Call delete_fn, if defined
    if (list->delete_fn && right_node->data) {
        list->delete_fn(right_node->data);
    }

    if (right_node) scope_free(right_node);

    return TRUE;
}

void*
lstFind (list_t *list, list_key_t search_key)
{
    if (!list) return NULL;

    list_element_t *right_node, *left_node;

    right_node = search (list, search_key, &left_node);
    if ((!right_node) || (right_node->key != search_key)) {
        return NULL;
    } else {
        return right_node->data;
    }
}

void
lstDestroy(list_t** list)
{
    if (!list || !*list) return;

    list_element_t *head = (*list)->head;

    // Delete the first element until there isn't a first element.
    list_element_t *head_next;
    do {
        head_next = get_unmarked_reference(head->next);
        if (head_next) {
            lstDelete(*list, head_next->key);
        }
    } while (!CAS(&head->next, NULL, get_marked_reference(NULL)));

    // Delete the head of the list
    (*list)->head = NULL;
    scope_free(head);

    // Delete the list itself
    scope_free(*list);
    *list = NULL;
}
