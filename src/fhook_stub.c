#include "scopestdlib.h"
#include "fhook.h"

void *
fhook_stub_create(void) {
    return scope_malloc(16);
}

int
fhook_stub_destroy(void *funchook) {
    scope_free(funchook);
    return 0;
}

int
fhook_stub_install(void *funchook, int flags) {
    return 0;
}

int
fhook_stub_prepare(void *funchook, void **target_func, void *hook_func) {
    return 0;
}

const char *
fhook_stub_error_message(const void *funchook) {
    return "";
}

int fhook_stub_set_debug_file(const char *name) {
    return 0;
}
