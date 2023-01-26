#ifndef __TEST_H__
#define __TEST_H__

/*
   cmocka.h documents that it's a requirement of cmocka.h that these other
   four header files must be included before cmocka.h.  Instead of doing
   this everywhere, just do it here and have our files use this instead:

#include "test.h"
*/

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <stdint.h>
#include "cmocka.h"


// Some weak symbols to help with compilation
#ifndef bool
typedef unsigned int bool;
#endif
bool __attribute__((weak)) cmdAttach(void) { return 1; }
bool __attribute__((weak)) cmdDetach(void) { return 1; }


// This is a convenient place to stick some helper functions too...

int groupSetup(void** state);
int groupTeardown(void** state);

void dbgHasNoUnexpectedFailures(void** state);
void dbgDumpAllToBuffer(char* buf, int size);

int writeFile(const char* path, const char* text);
int deleteFile(const char* path);
long fileEndPosition(const char* path);



#endif //__TEST_H__
