#include "os.h"

int osGetProcname(char *pname, size_t len) {
    proc_name(getpid(), pname, len);
    return 0;
}
