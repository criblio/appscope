#ifndef __FHOOK_H__
#define __FHOOK_H__

#ifndef DISABLE_FUNCHOOK

#include "../../contrib/funchook/include/funchook.h"

#define scope_funchook_t               funchook_t
#define scope_funchook_create          funchook_create
#define scope_funchook_destroy         funchook_destroy
#define scope_funchook_install         funchook_install
#define scope_funchook_error_message   funchook_error_message
#define scope_funchook_prepare         funchook_prepare
#define scope_funchook_set_debug_file  funchook_set_debug_file

#else

#define scope_funchook_t              void

void *fhook_stub_create(void);
int fhook_stub_destroy(void *);
int fhook_stub_install(void *, int);
const char *fhook_stub_error_message(const void *);
int fhook_stub_prepare(void *, void **, void *);
int fhook_stub_set_debug_file(const char *);

#define scope_funchook_create          fhook_stub_create
#define scope_funchook_destroy         fhook_stub_destroy
#define scope_funchook_install         fhook_stub_install
#define scope_funchook_error_message   fhook_stub_error_message
#define scope_funchook_prepare         fhook_stub_prepare
#define scope_funchook_set_debug_file  fhook_stub_set_debug_file

#endif

#endif // __FHOOK_H__
