// gcc -g -o sighandler sighandler.c

// This test app was created to handle a set of signals that the library
// also has an interest in.  It can register for signals in different ways:
// signal() or sigaction().
//
// The goal is to test that our library can manage to call original
// test app signal handlers after the library's own handlers.
// This will be of interest in a couple senarios.  When:
//     1) the library is preloaded and
//     2) when our library is injected at a later time.

#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const char *progname;

#define toStdout(str) (write(STDOUT_FILENO, str, strlen(str)))
#define failure(str) do { printf(str); exit(1); } while (0)


typedef enum {SIGNAL, SIGACTION} func_t;
static const char * const funcToName[] = {
    [SIGNAL]          = "signal",
    [SIGACTION]       = "sigaction",
};
#define FUNC_COUNT (sizeof(funcToName)/sizeof(funcToName[0]))


// a handler that just prints something to stdout to show it's been run
void
handleit(char *signame)
{
    // Ugly, but is a sig-safe alternative to:
    // printf("  Handling %s from %s\n", signame, progname);
    toStdout("  Handling ");
    toStdout(signame);
    toStdout(" from ");
    toStdout(progname);
    toStdout("\n");
    fsync(STDOUT_FILENO);
}

void handleIll(int signum) {handleit("SIGILL");}
void handleBus(int signum) {handleit("SIGBUS");}
void handleFpe(int signum) {handleit("SIGFPE");}
void handleSegv(int signum) {handleit("SIGSEGV");}
void handleUsr2(int signum) {handleit("SIGUSR2");}

typedef struct {
    int num;
    char *str;
    sighandler_t fn;
} sigList_t;

// These are the ones which libscope.so is interested in.
// So, these are ones we should test for interactions
sigList_t sigList[] = {
    {.num = SIGILL, .str = "SIGILL", .fn = handleIll},
    {.num = SIGBUS, .str = "SIGBUS", .fn = handleBus},
    {.num = SIGFPE, .str = "SIGFPE", .fn = handleFpe},
    {.num = SIGSEGV, .str = "SIGSEGV", .fn = handleSegv},
    {.num = SIGUSR2, .str = "SIGUSR2", .fn = handleUsr2},
};
#define SIG_COUNT (sizeof(sigList)/sizeof(sigList[0]))

// registers the handler, using the func specified
void
registerSigHandlers(func_t func)
{
    printf("  Executing %s using %s()\n", __func__, funcToName[func]);

    int i;
    for (i=0; i<SIG_COUNT; i++) {
        printf("    Registering %s\n", sigList[i].str);
        if (func == SIGNAL) {
            if (signal(sigList[i].num, sigList[i].fn) == SIG_ERR) {
                failure("signal() call failed\n");
            }
        } else if (func == SIGACTION) {
            struct sigaction act = {.sa_handler = sigList[i].fn,
                                    .sa_mask = 0,
                                    .sa_flags = 0};
            struct sigaction oldact;
            if (sigaction(sigList[i].num, &act, &oldact) == -1) {
                failure("sigaction() call failed\n");
            }
        } else {
            failure("Unexpected function\n");
        }
    }
}


int
main(int argc, char *argv[])
{
    progname = argv[0];

    // Read command line argument to determine which function to use
    // when registering our signal handler
    printf("Starting execution of %s\n", progname);
    if (argc != 2) {
        failure("expected one argument, either \"signal\" or \"sigaction\"\n");
    }

    func_t function;
    int i;
    for (i=0; i<FUNC_COUNT; i++) {
        if (!strcmp(argv[1], funcToName[i])) {
            function = i;
            break;
        }
    }
    if (i >= FUNC_COUNT) {
        failure("expected one argument, either \"signal\" or \"sigaction\"\n");
    }

    // register signal handler using the specified function
    printf("Registering to handle signals\n");
    registerSigHandlers(function);

    // wait a while for signal from the outside
    struct timespec time = {.tv_sec = 90, .tv_nsec = 0};
    nanosleep(&time, NULL);

    printf("Ending execution of %s\n", progname);
    return 0;
}
