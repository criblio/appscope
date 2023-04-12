/*
 * Testing the platform cycle clock assembly
 *
 * gcc -g test/manual/cycleclock.c -o cycleclock
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __x86_64__
#include <cpuid.h>

static inline int
rdtscp_supported(void) {
    unsigned a, b, c, d;
    if (__get_cpuid(0x80000001, &a, &b, &c, &d) && (d & (1<<27))) {
        // RDTSCP is supported.
        return 0;
    } else {
        // RDTSCP is not supported.
        return -1;
    }
}
#endif

static inline uint64_t
getTick(void) {
#ifdef __x86_64__
    unsigned low, high;
    if (rdtscp_supported() == 0) {
        asm volatile("rdtscp" : "=a" (low), "=d" (high));
    } else {
        asm volatile("rdtsc" : "=a" (low), "=d" (high));
    }
    return ((uint64_t)low) | (((uint64_t)high) << 32);
#elif defined(__aarch64__)
    uint64_t cnt;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#elif defined(__riscv) && __riscv_xlen == 64
    uint64_t cnt;
    __asm__ volatile ("rdcycle %0" : "=r" (cnt));
    return cnt;
#else
#error Architecture is not defined
#endif
}

int
main(int argc, char **argv) {
    uint64_t starttick, endtick, diff;
    starttick = getTick();
    printf("Tick value: %" PRIu64 "\n", starttick);
    printf("1 sec sleep\n");
    sleep(1);
    endtick = getTick();
    diff = endtick - starttick;
    printf("Tick value: %" PRIu64 "\n", endtick);
    printf("Tick difference: %" PRIu64 "\n", diff);
    return 0;
}
