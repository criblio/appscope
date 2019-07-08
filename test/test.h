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

#endif //__TEST_H__
