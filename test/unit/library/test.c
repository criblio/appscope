#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "scopestdlib.h"
#include "dbg.h"
#include "test.h"

/*
* Following part of code is required to use memory sanitizers during unit tests
* To correct instrument code we redirect allocations from our internal
* library to allocator from standard library.
* See details in:
* https://github.com/google/sanitizers/wiki/AddressSanitizerIncompatiblity
*
*   make FSAN=1 libtest allows to instrument library unit test code
*
* The memory leak instrumentation is done by "-fsanitize=address"
*/

/*
* In GCC sanitize address is defined when -fsanitize=address is used
* In Clang the code below is recommended way to check this feature
*/
#if defined(__has_feature)
# if __has_feature(address_sanitizer)
#  define __SANITIZE_ADDRESS__ 1
# endif
#endif

#ifdef __SANITIZE_ADDRESS__
#include <stdlib.h>
void * __real_scopelibc_malloc(size_t);
void * __wrap_scopelibc_malloc(size_t size)
{
    return malloc(size);
}

void __real_scopelibc_free(void *);
void __wrap_scopelibc_free(void * ptr)
{
    return free(ptr);
}

void * __real_scopelibc_calloc(size_t, size_t);
void * __wrap_scopelibc_calloc(size_t nelem, size_t size)
{
    return calloc(nelem, size);
}

void * __real_scopelibc_realloc(void *, size_t);
void * __wrap_scopelibc_realloc(void * ptr, size_t size)
{
    return realloc(ptr, size);
}
#endif


int
groupSetup(void** state)
{
    dbgInit();
    return 0;
}

int
groupTeardown(void** state)
{
    // as a policy, we're saying all tests that call groupSetup and
    // groupTeardown should be aware of things that would cause dbg
    // failures, and cleanup after themselves (call dbgInit()) before
    // execution is complete.
    unsigned long long failures = dbgCountAllLines();
    if (failures) {
        dbgDumpAll(stdout);
    }
    dbgDestroy();
    return failures;
}

void
dbgHasNoUnexpectedFailures(void** state)
{
    unsigned long long failures = dbgCountAllLines();
    if (failures) {
        dbgDumpAll(stdout);
    }
    assert_false(failures);
}

void
dbgDumpAllToBuffer(char* buf, int size)
{
    FILE* f = scope_fmemopen(buf, size, "a+");
    assert_non_null(f);
    dbgDumpAll(f);
    if (scope_ftell(f) >= size) {
        fail_msg("size of %d was inadequate for dbgDumpAllToBuffer, "
                 "%ld was needed", size, scope_ftell(f));
    }
    if (scope_fclose(f)) fail_msg("Couldn't close fmemopen'd file");
}

int
writeFile(const char* path, const char* text)
{
    FILE* f = scope_fopen(path, "w");
    if (!f)
        fail_msg("Couldn't open file");

    if (!scope_fwrite(text, scope_strlen(text), 1, f))
        fail_msg("Couldn't write file");

    if (scope_fclose(f))
        fail_msg("Couldn't close file");

    return 0;
}

int
deleteFile(const char* path)
{
    return scope_unlink(path);
}

long
fileEndPosition(const char* path)
{
    FILE* f;
    if ((f = scope_fopen(path, "r"))) {
        scope_fseek(f, 0, SEEK_END);
        long pos = scope_ftell(f);
        scope_fclose(f);
        return pos;
    }
    return -1;
}
