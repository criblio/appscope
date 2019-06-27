#include "os.h"

int osGetProcname(char *pname, int len) {
    strncpy(pname, program_invocation_short_name, len);
    return 0;
}
