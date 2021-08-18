/*
 * Testing the operation of Interposing Functions in Dependent Libraries
 * The libscope library has intgerposed malloc. Call malloc and see that we get the interposed function.
 *
 * gcc -g -I ~/appscope/src test/manual/armtimer.c -o at
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>

#include "plattime.h"

platform_time_t g_time = {0};

static int
osInitCntr(platform_time_t *cfg)
{
#ifdef __aarch64__
    /* 
     * This uses the General Purpose Timer definiton in an aarch64 instance.
     * The frequency is the lower 32 bits of the CNTFRQ_EL0 register and
     * is defined as HZ. The configured freq is defined in Mhz.
     */
    uint64_t freq;
    __asm__ volatile (
        "mrs x1, CNTFRQ_EL0 \n"
        "mov %0, x1  \n"
        : "=r" (freq)                // output
        :                            // inputs
        :                            // clobbered register
        );

    freq &= 0x0000000ffffffff;
    freq /= 1000000;
    cfg->freq = freq;
    return 0;
#else
    return -1;
#endif
}

/*

    asm("mov    %[result], %[value], ror #1"

           : [result]"=r" (y) // Rotation result
           : [value]"r"   (x) // Rotated value.
           : // No clobbers
    );
    __asm__ volatile (
        "mov %1, %%rdi  \n"
        "callq *%2  \n"
        : "=r"(rc)                    // output
        : "r"(stackptr), "r"(cfunc)   // inputs
        :                             // clobbered register
        );

 */
int
main(int argc, char **argv)
{
    unsigned long freq, cnt, gtime, gdur;

    asm("mrs x1, CNTFRQ_EL0"); //CNTPCT_EL0 CNTFRQ_EL0 CNTKCTL_EL1
    asm("mrs x1, CNTVCT_EL0"); //CNTVCT_EL0 CNTV_CVAL_EL0 CNTV_TVAL_EL0 CNTVCTSS_EL0  CNTP_TVAL_EL0 CNTPCTSS_EL0 CNTHPS_TVAL_EL2 CNTPOFF_EL2

    __asm__ volatile (
        "mrs x1, CNTFRQ_EL0 \n"
        "mov %0, x1  \n"
        : "=r" (freq)                // output
        :                            // inputs
        :                            // clobbered register
        );

    __asm__ volatile (
        "mrs x1, CNTVCT_EL0 \n"
        "mov %0, x1  \n"
        : "=r" (cnt)                // output
        :                            // inputs
        :                            // clobbered register
        );


    freq &= 0x0000000ffffffff;
    freq /= 1000000;
    printf("%s:%d %ld count 0x%lx\n", __FUNCTION__, __LINE__, freq, cnt);

    osInitCntr(&g_time);
    //g_time.gptimer_avail = FALSE;
    g_time.gptimer_avail = TRUE;
    gtime = getTime();
    printf("%s:%d start time 0x%lx\n", __FUNCTION__, __LINE__, gtime);
    gdur = getDuration(gtime);
    printf("%s:%d duration %ldns\n", __FUNCTION__, __LINE__, gdur);
    return 0;
}
