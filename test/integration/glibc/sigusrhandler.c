// gcc -g -o sigusrhandler sigusrhandler.c

// This test app was created to handle a SIGUSR2 that the library
// also has an interest in.
// It can register SIGUSR2 in different ways:
// signal()/sigaction().
// It can trigger SIGUSR2 in different ways:
// timer_create()/raise().
//
// The goal is to test that our library can manage to call original
// test app signal handlers after the library's own handlers.


#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const char *progname;

#define toStdout(str) (write(STDOUT_FILENO, str, strlen(str)))

void sigUsr2Handler(int signum) {
    toStdout("  Handling SIGUSR2");
    toStdout(" from ");
    toStdout(progname);
    toStdout("\n");
    fsync(STDOUT_FILENO);
}

typedef struct {
    int num;
    char *str;
    sighandler_t fn;
} sigList_t;

static int
registerHandler(const char *mode) {
    if (strcmp(mode, "signal") == 0) {
        return (signal(SIGUSR2, sigUsr2Handler) == SIG_ERR) ? -1 : 0;
    } else if (strcmp(mode, "sigaction") == 0) {
        struct sigaction sa = { 0 };
        sa.sa_handler = sigUsr2Handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        return sigaction(SIGUSR2, &sa, NULL);
    }
    fprintf (stderr, "Unknown mode\n");
    return -1;
}

static int
setTrigger(const char *trigger) {
    if (strcmp(trigger, "raise") == 0) {
        return raise(SIGUSR2);
    } else if (strcmp(trigger, "timer") == 0) {
        timer_t timerId;
        struct sigevent sev;
        struct itimerspec its;

        sev.sigev_notify = SIGEV_SIGNAL;
        sev.sigev_signo = SIGUSR2;
        if (timer_create(CLOCK_MONOTONIC, &sev, &timerId) == -1) {
            fprintf(stderr, "timer_create fails\n");
            return -1;
        }
        its.it_interval.tv_sec = 0;
        its.it_interval.tv_nsec = 0;
        its.it_value.tv_sec = 1;
        its.it_value.tv_nsec = 0;

        return timer_settime(timerId, 0, &its, NULL);
    }
    fprintf(stderr, "Unknown trigger\n");
    return -1;
}

static void
createAndDeleteDummyFile(void) {
    FILE *file;
    char *filename = "dummy_file.txt";

    file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to create the file.\n");
        return;
    }

    fprintf(file, "Lorem Ipsum");

    fclose(file);

    unlink(filename);
}


int
main(int argc, char *argv[])
{
    progname = argv[0];

    // Read command line argument to determine which function to use
    // when registering our signal handler
    printf("Starting execution of %s\n", progname);
    if (argc != 3) {
        fprintf(stderr,"Usage: %s [signal|sigaction] [raise|timer]\n", progname);
        exit(EXIT_FAILURE);
    }
    const char *mode= argv[1];
    const char *trigger= argv[2];

    if (registerHandler(mode) != 0) {
       fprintf (stderr, "registerHandler fails\n");
       exit(EXIT_FAILURE);
    }

    if (setTrigger(trigger) != 0) {
       fprintf (stderr, "setTrigger fails\n");
       exit(EXIT_FAILURE);
    }

    //Fake sleep  
    sleep(4);

    printf("Ending execution of %s\n", progname);

    // To generate scope data
    createAndDeleteDummyFile();

    return 0;
}
