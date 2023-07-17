/*
 * Testing unshare CLONE_NEWNET and impact for scope
 *
 * gcc test/manual/namespace/ns_unsharetest.c -o unsharetest
 * sudo SCOPE_CRIBL_ENABLE=FALSE ldscope -- ./unsharetest
 */
#define _GNU_SOURCE
#include <linux/limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#define TEST_FILE "file.txt"
#define TEST_FILE_NEW "file_new.txt"
#define EXAMPLE_STR "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
#define EXAMPLE_STR_NEW "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

static void
eventGenBeforeSwitchNs(void) {
    for (int i = 0 ; i < 3; ++i) {
        FILE *fp = fopen(TEST_FILE,"w");
        if (!fp)
            exit(EXIT_FAILURE);
        fwrite(EXAMPLE_STR, sizeof(char), sizeof(EXAMPLE_STR), fp);
        fclose(fp);
        unlink(TEST_FILE);
        sleep(1);
    }
}

static void
eventGenAfterSwitchNs(void) {
    for (int i = 0 ; i < 30; ++i) {
        FILE *fp = fopen(TEST_FILE_NEW,"w");
        if (!fp)
            exit(EXIT_FAILURE);
        fwrite(EXAMPLE_STR_NEW, sizeof(char), sizeof(EXAMPLE_STR_NEW), fp);
        fclose(fp);
        unlink(TEST_FILE_NEW);
        sleep(1);
    }
}


int main(int argc, char *argv[], char *envp[]) {
    printf("Begin to generate event before switching namespace %d \n", getpid());

    eventGenBeforeSwitchNs();

    printf("End generate event before switching namespace %d \n", getpid());

    // Switch to new network namespace
    if (unshare(CLONE_NEWNS) == -1) {
        printf("unshare failed\n");
        return EXIT_FAILURE;
    }

    sleep(10);

    printf("Begin to generate event after switching namespace %d \n", getpid());

    eventGenAfterSwitchNs();

    printf("End generate event after switching namespace %d \n", getpid());

    return EXIT_SUCCESS;
}
