/*
 * Testing setns CLONE_NEWNET and impact for scope
 *
 * Start docker 
 * docker run --privileged --name testcont --rm -it ubuntu:22.04 bash
 * In separate terminal
 * Retrieve pid of Docker process via docker top testcont to access docker's network namespace
 * gcc test/manual/namespace/ns_setnstest.c -o setnstest
 * sudo SCOPE_CRIBL_ENABLE=FALSE ldscope -- ./setnstest <docker_pid>
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
#define EXAMPLE_STR_NEW "New lorem ipsum"

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
    char path[PATH_MAX] = {0};
    if (argc != 2) {
        printf("Usage: %s <pid for different network ns>\n", argv[0]);
        return EXIT_FAILURE;
    }
    snprintf(path, PATH_MAX, "/proc/%s/ns/net", argv[1]);
    printf("Begin to generate event before switching namespace %d \n", getpid());

    eventGenBeforeSwitchNs();

    printf("End generate event before switching namespace %d \n", getpid());

    // Switch namespace
    int fd = open(path, O_RDONLY);
    if (fd == -1 ) {
        fprintf(stderr, "open");
        return EXIT_FAILURE;
    }

    if (setns(fd, CLONE_NEWNET) == -1) {
        fprintf(stderr, "setns");
        return EXIT_FAILURE;
    }
    close(fd);

    sleep(10);

    printf("Begin to generate event after switching namespace %d \n", getpid());

    eventGenAfterSwitchNs();

    printf("End generate event after switching namespace %d \n", getpid());

    return EXIT_SUCCESS;
}

