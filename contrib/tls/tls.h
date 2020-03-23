#ifndef __TLS_H__
#define __TLS_H__

struct gnutls_session_def;
typedef struct gnutls_session_def *gnutls_session_t;

struct ssl_session_def;
typedef struct ssl_session_def SSL;

ssize_t gnutls_record_recv(gnutls_session_t, void *, size_t);
ssize_t gnutls_record_send_early_data(gnutls_session_t, const void *, size_t);

int SSL_read(SSL *, void *, int);
int SSL_write(SSL *, const void *, int);
int SSL_get_fd(const SSL *);

#endif // __TLS_H__
