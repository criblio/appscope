static struct first_tls {
	char c;
    char pt[512];
	void *space[16];
} first_tls[1];

static inline uintptr_t __get_tp()
{
    return (uintptr_t)first_tls[0].pt;
}

#define MC_PC gregs[REG_RIP]
