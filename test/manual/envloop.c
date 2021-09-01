/*
 * envloop - Event Loop Tool
 *
 * A simple program to display any `SCOPE_*` environment variables in a loop so
 * we can test setting them when we "attach".
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>

extern char **environ;

static volatile int running = 1;

void intHandler(int dummy) {
    printf("info: exiting on ctrl-c\n");
    running = 0;
}

int main(int argc, char **argv, char **env)
{
    signal(SIGINT, intHandler);

    while (running) {
        int i;
        for (i = 0; environ[i]; i++) {
            if (strlen(environ[i]) > 6 && strncmp(environ[i], "SCOPE_", 6) == 0) {
                printf("  %s\n", environ[i]);
            }
        }
        printf("--\n");
        sleep(3);
    }
    
    return EXIT_SUCCESS;
}
