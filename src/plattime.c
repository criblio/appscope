#define _GNU_SOURCE
#include "plattime.h"
#include "os.h"
#include <limits.h>

platform_time_t g_time = {0};

platform_time_t *
initTime(void)
{
    osInitTimer(&g_time);
    return &g_time;
}
