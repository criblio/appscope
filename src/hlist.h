#ifndef __HLIST_H__
#define __HLIST_H__

#include "../contrib/tls/tls.h"

typedef struct http_list_t {
    struct http_list_t *next;
    uint64_t id;
    void *protocol;
    PRIOMethods *ssl_methods;
    PRIOMethods *ssl_int_methods;
} http_list;

extern http_list *g_hlist;

http_list *hnew();
int hpush(http_list **, http_list *);
int hrem(http_list **, uint64_t);
http_list *hget(http_list *, uint64_t);

#endif // __HLIST_H__
