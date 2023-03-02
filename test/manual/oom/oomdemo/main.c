#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GB (1024 * 1024 * 1024)
#define ARRAY_SIZE 10000L

int main() {
    char *arr[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = malloc(1 * GB);
        if (!arr[i]) {
            printf("malloc failed in block %d\n", i);
            return 0;
        }
    }

    // Write to the physical memory - to generate OOM
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        memset(arr[i], 0, 1 * GB);
        printf(" %d\n", i);
    }

    return 0;
}
