#define _GNU_SOURCE
#include "snapshot.h"
#include "test.h"
#include "scopestdlib.h"
#include "scopetypes.h"
#include "fn.h"
#include <ftw.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>

static int
rm_callback(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    return (remove(fpath) < 0) ? -1 : 0;
}

static int
rm_recursive(char *path) {
    return nftw(path, rm_callback, 64, FTW_DEPTH | FTW_PHYS);
}

struct snap_prefix {
    char name[64];
    bool present;
};

#define PREFIX_NO 3

static void
snapshotSigSegvTest(void **state)
{
    char dirPath[PATH_MAX] = {0};
    struct stat fileState;
    DIR *dirp;
    struct dirent *entry;
    pid_t cpid = fork();
    assert_int_not_equal(-1, cpid);
    if (cpid == 0){
        /*
        * Child process will perform following operations:
        *
        * - ignore SIGSEGV signal
        * - perform `snapshotSignalHandler` function which will stop the process for 5 seconds
        */
        siginfo_t info = {.si_signo = SIGSEGV, .si_code = SEGV_MAPERR};
        signal(SIGSEGV, SIG_IGN);
        snapshotSetCoredump(TRUE);
        snapshotSetBacktrace(TRUE);
        snapshotSignalHandler(-1, &info, NULL);
        exit(EXIT_SUCCESS);
    } else {
        /*
        * Parent process will perform following operations:
        *
        * - will suspends execution of itself until if detects that child process was stopped (`snapshotSignalHandler`)
        */
        int wstatus;
        pid_t wpid;
        wpid = waitpid(cpid, &wstatus, WUNTRACED);
        if (wpid == -1) {
            exit(EXIT_FAILURE);
        }
    }
    scope_snprintf(dirPath, sizeof(dirPath), "/tmp/appscope/%d/", cpid);

    // Verify if specifed action was performed
    int res = scope_stat(dirPath, &fileState);
    assert_int_equal(res, 0);
    assert_true(S_ISDIR(fileState.st_mode));
    // cfg is missing becase  g_cfg.cfgstr is not initialized
    struct snap_prefix snapFilePrefixes[PREFIX_NO] = {
                                                {"info_", FALSE},
                                                {"backtrace_", FALSE},
                                                {"core_", FALSE}
                                            };


    dirp = scope_opendir(dirPath);
    assert_non_null(dirp);

    while ((entry = scope_readdir(dirp)) != NULL) {
        if (entry->d_type == DT_REG) {
            for (int i = 0; i < PREFIX_NO; ++i) {
                if (scope_strstr(entry->d_name, snapFilePrefixes[i].name)) {
                    snapFilePrefixes[i].present = TRUE;
                }
            }
        }
    }
    scope_closedir(dirp);

    for (int i = 0; i < PREFIX_NO; ++i) {
        assert_true(snapFilePrefixes[i].present);
    }

    // clean it up
    res = rm_recursive(dirPath);
    assert_int_equal(res, 0);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(snapshotSigSegvTest),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
