#include <stdio.h>

// This test was written to manually test the hotpatching code that was
// added in CRIBL-2283 that uses the funchook and distorm libraries.
//
// With it we observed that we could only find and patch the SSL_read
// function if it was published in the dynamic table by the executable
// (as in ./ssl_main_dynamic).  At the time this was surprising to us
// because we thought we'd be able to see any public SSL_read symbol
// that the executable defined.  Stating what's clear now: the dynamic
// linker and it's functions (dlsym, dladdr, dlopen, etc) only deal
// functions that appear in the dynamic section of the elf table.
// This is the normal world of shared libraries, but executables can
// be built to publish symbols in the dynamic section as well.  node
// from nodejs.org is one like this.
//


// Our shared library, and app that depends on it.
//
// gcc -g -DSSL_LIB -shared -fPIC test/manual/SSL_read.c -o libmyssl.so
// gcc -g test/manual/SSL_read.c -L. -lmyssl -o ssl_main_with_myssl_dep
//
// LD_LIBRARY_PATH=`pwd` ldd ./ssl_main_with_myssl_dep
// objdump -T ./ssl_main_with_myssl_dep
// LD_LIBRARY_PATH=`pwd` ./ssl_main_with_myssl_dep


// Our statically linked library, built into app
//
// gcc -g -DSSL_LIB -fPIC -c test/manual/SSL_read.c -o libmyssl.o
// gcc -g test/manual/SSL_read.c libmyssl.o -rdynamic -o ssl_main_dynamic
// gcc -g test/manual/SSL_read.c libmyssl.o -o ssl_main
//
// ldd ./ssl_main_dynamic ./ssl_main
// objdump -T ./ssl_main_dynamic ./ssl_main
// ./ssl_main_dynamic


// Real statically linked library, built into app
//
// apt-get -o Acquire::Check-Valid-Until=false update
// sudo apt install libssl-dev
// gcc -g test/manual/SSL_read.c -L/usr/lib/x86_64-linux-gnu/ -lssl -rdynamic -o ssl_main_real_ssl
// ./ssl_main_real_ssl


// Common definitions
struct ssl_session_def;
typedef struct ssl_session_def SSL;
int SSL_read(SSL *, void *, int );


#ifdef SSL_LIB
int
SSL_read(SSL *ssl, void *ptr, int i)
{
    printf("SSL_read function was called!\n");
}
#endif // SSL_LIB


#ifndef SSL_LIB
int
main()
{
    printf("Running SSL_read.c\n");

    SSL_read(NULL, NULL, 3);

    return 0;
}
#endif // SSL_LIB
