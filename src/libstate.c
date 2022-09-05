#include "libstate.h"
#include <fcntl.h>
#include "scopestdlib.h"

static void *loadedStrMap = NULL;
#define LOADED_MAP_SIZE (4096)

/*
 * AppScope state describes the process state in context of the AppScope library.
 * We recognize two states:
 * - Scoped - when all the functions are funchooked
 * - AppScope library is loaded - when execve family functions are funchooked
 * 
 * The distinguish between the AppScope states is based on presence of the "loaded mapping".
 * 
 * Example loaded mapping view in /proc/<PID>/maps:
 * 7fa2b4734000-7fa2b4735000 rw-s 00000000 103:07 23856559  /tmp/scope_loaded.2142471 (deleted)
 */

/*
 * libstateLoaded sets the loaded state for the current process by:
 * - creating a temporary file `/tmp/scope_loaded.<pid>`. 
 * - the file is mapped into the process memory as "loaded mapping"
 * - the file is removed (unlinked)
 */
bool
libstateLoaded(pid_t pid) {
    char path[PATH_MAX] = {0};
    bool res = FALSE;
    /*
    * Switching to "loaded" AppScope state:
    * We are done if mapping is present.
    */
    if (loadedStrMap) {
        return TRUE;
    }

    if (scope_snprintf(path, sizeof(path), "/tmp/scope_loaded.%d", pid) < 0) {
        return res;
    }
    
    int outFd = scope_open(path, O_RDWR | O_CREAT, 0664);
    if (outFd == -1)  {
        return res;
    }

    if (scope_ftruncate(outFd, LOADED_MAP_SIZE) != 0) {
        goto close_file;
    }

    void* dest = scope_mmap(NULL, LOADED_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, outFd, 0);
    if (dest == MAP_FAILED) {
        goto close_file;
    }

    loadedStrMap = dest;

    res = TRUE;

close_file:
    scope_close(outFd);
    if (scope_unlink(path) != 0 ) {
        return FALSE;
    }
    return res;
}

/*
 * libstateScoped sets the scoped state for the current process by unmap the "loaded mapping"
 */
bool
libstateScoped(void) {
    /*
    * Switching to "scoped" AppScope state:
    * We remove "loaded mapping" if present.
    */
    if (loadedStrMap != NULL ) {
        scope_munmap(loadedStrMap, LOADED_MAP_SIZE);
        loadedStrMap = NULL;
    }
    return TRUE;
}
