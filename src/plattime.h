#ifndef __TIME_H__
#define __TIME_H__

#include <stdint.h>
#include "scopetypes.h"

typedef struct {
    bool tsc_invariant;
    bool tsc_rdtscp;
    uint64_t freq;
} platform_time_t;

platform_time_t* initTime(void);
uint64_t getTime(void);
uint64_t getDuration(uint64_t start);

#endif // __TIME_H__
