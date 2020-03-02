#ifndef __TIME_H__
#define __TIME_H__

#include <limits.h>
#include <stdint.h>
#include "scopetypes.h"

typedef struct {
    bool tsc_invariant;
    bool tsc_rdtscp;
    uint64_t freq;
} platform_time_t;

platform_time_t* initTime(void);


// We haven't measured it, but there are concerns about performance
// with calling getTime and getDuration as functions across modules.
// Because of this, we're exposing g_time and making getTime and
// getDuration static inline functions that will use g_time instead
// of hiding this implementation in plattime.c.  There is basically
// no code space cost - it's only 288 bytes per .c file that includes
// this.
extern platform_time_t g_time;

static inline uint64_t
getTime(void) {
    unsigned low, high;

    /*
     * Newer CPUs support a second TSC read instruction.
     * The new instruction, rdtscp, performes a serialization
     * instruction before calling RDTSC. Specifically, rdtscp
     * performs a cpuid instruction then an rdtsc. This is
     * intended to flush the instruction pipeline befiore
     * calling rdtsc.
     *
     * A serializing instruction is used as the order of
     * execution is not guaranteed. It's described as
     * "Out ofOrder Execution". In some cases the read
     * of the TSC can come before the instruction being
     * measured. That scenario is not very likely for us
     * as we tend to measure functions as opposed to
     * statements.
     *
     * If the rdtscp instruction is available, we use it.
     * It takes a bit longer to execute due to the extra
     * serialization instruction (cpuid). However, it's
     * supposed to be more accurate.
     */
    if (g_time.tsc_rdtscp == TRUE) {
        asm volatile("rdtscp" : "=a" (low), "=d" (high));
    } else {
        asm volatile("rdtsc" : "=a" (low), "=d" (high));
    }
    return ((uint64_t)low) | (((uint64_t)high) << 32);
}


// Return the time delta from start to now in nanoseconds
static inline uint64_t
getDuration(uint64_t start)
{
    // before the constructor runs, g_time.freq is zero.
    // Avoid div by zero during this time.
    if (!g_time.freq) return 0ULL;

    /*
     * The clock frequency is in Mhz.
     * In order to get NS resolution we
     * multiply the difference by 1000.
     *
     * If the counter rolls over we adjust
     * by using the max value of the counter.
     * A roll over is rare. But, we should handle it.
     */
    uint64_t now = getTime();
    if (start < now) {
        return ((now - start) * 1000) / g_time.freq;
    } else {
        return (((ULONG_MAX - start) + now) * 1000) / g_time.freq;
    }

}

#endif // __TIME_H__
