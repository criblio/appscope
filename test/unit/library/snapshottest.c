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

static void
snapshotSigSegvTest(void **state)
{
    char dirPath[PATH_MAX] = {0};
    struct stat fileState;
    pid_t cpid = fork();
    assert_int_not_equal(-1, cpid);
    if (cpid == 0){
        /*
        * Child process will perform following operations:
        *
        * - ignore SIGSEGV signal
        * - perform `snapshotSignalHandler` function which will stop the process using SIGSTOP 
        * - SIGCONT delivered from parent will allow child process to continue
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
        * - send SIGCONT signal to child process allowing to continue
        */
        int wstatus;
        pid_t wpid;
        wpid = waitpid(cpid, &wstatus, WUNTRACED);
        if (wpid == -1) {
            exit(EXIT_FAILURE);
        }

        if (WIFSTOPPED(wstatus)) {
            kill(cpid, SIGCONT);
            wpid = waitpid(cpid, &wstatus, WCONTINUED);
            if (wpid == -1) {
                exit(EXIT_FAILURE);
            }
        } else {
            assert_non_null(NULL);
        }
    }
    scope_snprintf(dirPath, sizeof(dirPath), "/tmp/appscope/%d/", cpid);

    // Verify if specifed action was performed
    int res = scope_stat(dirPath, &fileState);
    assert_int_equal(res, 0);
    assert_true(S_ISDIR(fileState.st_mode));
    // cfg is missing becase  g_cfg.cfgstr is not initialized
    char *snapFileNames[] = {"info", "backtrace", "core"};
    for (int i = 0; i < sizeof(snapFileNames)/ sizeof(snapFileNames[0]); ++i) {
        char filePath[PATH_MAX] = {0};
        scope_snprintf(filePath, sizeof(filePath), "%s%s", dirPath, snapFileNames[i]);
        int res = scope_stat(filePath, &fileState);
        assert_int_equal(res, 0);
        assert_true(S_ISREG(fileState.st_mode));
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
