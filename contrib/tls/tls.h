#ifndef __TLS_H__
#define __TLS_H__

#define XP_UNIX

#include "nss/pprio.h"

struct gnutls_session_def;
typedef struct gnutls_session_def *gnutls_session_t;

struct ssl_session_def;
typedef struct ssl_session_def SSL;

ssize_t gnutls_record_recv(gnutls_session_t, void *, size_t);
ssize_t gnutls_record_send(gnutls_session_t, const void *, size_t);
int     gnutls_transport_get_int(gnutls_session_t);
void    gnutls_transport_get_int2(gnutls_session_t, int *, int *);

int SSL_read(SSL *, void *, int);
int SSL_write(SSL *, const void *, int);
int SSL_get_fd(const SSL *);

PRFileDesc *SSL_ImportFD(PRFileDesc *, PRFileDesc *);

#endif // __TLS_H__
