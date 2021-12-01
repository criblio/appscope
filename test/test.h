#ifndef __TEST_H__
#define __TEST_H__

/*
   cmocka.h documents that it's a requirement of cmocka.h that these other
   four header files must be included before cmocka.h.  Instead of doing
   this everywhere, just do it here and have our files use this instead:

#include "test.h"
*/

#include "cmocka.h"
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

// This is a convenient place to stick some helper functions too...

int groupSetup(void **state);
int groupTeardown(void **state);

void dbgHasNoUnexpectedFailures(void **state);
void dbgDumpAllToBuffer(char *buf, int size);

int writeFile(const char *path, const char *text);
int deleteFile(const char *path);
long fileEndPosition(const char *path);

#endif //__TEST_H__
