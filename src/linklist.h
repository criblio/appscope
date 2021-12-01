#ifndef __LINKLIST_H__
#define __LINKLIST_H__

#include <stdint.h>

typedef uint64_t list_key_t;
typedef struct _list_t list_t;
typedef void (*delete_fn_t)(void *); // signature for optional delete function

//
// Creates a new list object.  This list object can contain an arbitrary
// number of (key, data) elements.  (See lstInsert())
//
// The delete_fn argument provides a way for the list to deallocate data
// during a subsequent lstDelete or lstDestroy() call, if desired.
// If delete_fn is not NULL, then it is called with the argument of data -
// once within a lstDelete() or once for each element during lstDestroy().
//
// Returns NULL if the object can not be created.
list_t *lstCreate(delete_fn_t delete_fn);

// Stores the (key, data) pair as a new list element, provided key isn't
// already in the list.
// Returns true if (key, data) were successfully inserted in the list.
int lstInsert(list_t *list, list_key_t key, void *data);

// Removes the list element identified by key.
// If delete_fn was specified when the list was created, it will be called
// with the argument of data (see lstInsert()).
// Returns true if matching (key, data) pair was found and removed.
int lstDelete(list_t *list, list_key_t search_key);

// Returns data if (key, data) are found in the list.
void *lstFind(list_t *list, list_key_t search_key);

// Destroys a list object and all it's contents by calling lstDelete
// on every element, then freeing the list_t structure when it's complete.
void lstDestroy(list_t **list);

#endif // __LINKLIST_H__
