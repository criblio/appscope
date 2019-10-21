#ifndef __ATOMIC_H__
#define __ATOMIC_H__

#include <stdint.h>

#ifndef bool
typedef unsigned int bool;
#endif

static inline bool
atomicCasU64(uint64_t* ptr, uint64_t oldval, uint64_t newval)
{
    return __sync_bool_compare_and_swap(ptr, oldval, newval);
}

static inline void
atomicAddU64(uint64_t *ptr, uint64_t val) {
    // Ensure that we don't "add past maxint"...
    uint64_t oldval;
    uint64_t newval;
    do {
        oldval = *ptr;
        newval = oldval + val;
        if (newval < oldval) {
            newval = UINTMAX_MAX;
        }
    } while (!atomicCasU64(ptr, oldval, newval));
}

static inline void
atomicSubU64(uint64_t* ptr, uint64_t val) {
    // Ensure that we don't "subtract past zero"...
    uint64_t oldval;
    uint64_t newval;
    do {
        oldval = *ptr;
        if (val > oldval) {
            newval = 0;
        } else {
            newval = oldval - val;
        }
    } while (!atomicCasU64(ptr, oldval, newval));
}

static inline uint64_t
atomicSwapU64(uint64_t *ptr, uint64_t val) {
    return __sync_lock_test_and_set(ptr, val);
}



static inline bool
atomicCas32(int *ptr, int oldval, int newval)
{
   return __sync_bool_compare_and_swap(ptr, oldval, newval);
}

static inline void
atomicAdd32(int *ptr, int val) {
    (void)__sync_add_and_fetch(ptr, val);
}

static inline void
atomicSub32(int *ptr, int val)
{
    (void)__sync_sub_and_fetch(ptr, val);
}

static inline int
atomicSwap32(int *ptr, int val)
{
    return __sync_lock_test_and_set(ptr, val);
}

#endif // __ATOMIC_H__
