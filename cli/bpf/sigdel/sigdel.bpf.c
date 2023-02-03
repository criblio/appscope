#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>

#define COMM_LEN 128
#define LAST_32_BITS(x) x & 0xFFFFFFFF
#define FIRST_32_BITS(x) x >> 32

/*
 * Signals are filtered. We want to inform any user mode
 * readers about signals that are likley to cause a crash
 * in an application. At this point the signals we pass are:
 * SIGSEGV, SIGBUS, SIGILL and SIGFPE
 */

/*
 * Note: the original SEC macro defined in bpf_helpers.h
 * includes pragmas that enable a compiler warning when
 * the attribute is defined. The current go build has an
 * issue with the use of pragmas in this context. Removing
 * the pragmas allows go build to function properly.
 * It does not appear to have any issue so far.
 */
#define SEC_GO(name) __attribute__((section(name), used))

/*
 * Refer to the following for format definitions:
 * /sys/kernel/tracing/events/signal/signal_deliver/format
 */
struct sigdel_data_t {
    int pid;
    int sig;
    int errno;
    int code;
    unsigned long sa_handler;
    unsigned long sa_flags;
    unsigned char comm[COMM_LEN];
};

struct sigdel_data_t _edt = {0};

struct {
	__uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u32));
} events SEC_GO(".maps");

struct sigdel_args_t {
    unsigned short type;
    unsigned char flags;
    unsigned char preempt_count;
    int pid;
    int sig;
    int errno;
    int code;
    unsigned long sa_handler;
    unsigned long sa_flags;
};

SEC_GO("tracepoint/signal/signal_deliver")
int sig_deliver(struct sigdel_args_t *args)
{
    // SIGSEGV, SIGBUS, SIGILL and SIGFPE
    if ((args->sig != 11) && (args->sig != 7) &&
        (args->sig != 4) && (args->sig != 8)) return -1;

	struct sigdel_data_t sigdel_data = {};
	u64 pid_tgid;

	pid_tgid = bpf_get_current_pid_tgid();
	sigdel_data.pid = LAST_32_BITS(pid_tgid);

    if (bpf_probe_read(&sigdel_data.sig, sizeof(sigdel_data.sig), &args->sig) != 0) {
        bpf_printk("ERROR:sigdel:bpf_probe_read:sig number\n");
        sigdel_data.sig = -1;
    }

    if (bpf_probe_read(&sigdel_data.errno, sizeof(sigdel_data.errno), &args->errno) != 0) {
        bpf_printk("ERROR:sigdel:bpf_probe_read:errno\n");
        sigdel_data.errno = -1;
    }

    if (bpf_probe_read(&sigdel_data.code, sizeof(sigdel_data.code), &args->code) != 0) {
        bpf_printk("ERROR:sigdel:bpf_probe_read:code\n");
        sigdel_data.code = -1;
    }

    if (bpf_probe_read(&sigdel_data.sa_handler, sizeof(sigdel_data.sa_handler), &args->sa_handler) != 0) {
        bpf_printk("ERROR:sigdel:bpf_probe_read:sa_handler\n");
        sigdel_data.sa_handler = (unsigned long)-1;
    }

	if (bpf_get_current_comm(sigdel_data.comm, sizeof(sigdel_data.comm)) != 0) {
        bpf_printk("ERROR:sigdel:bpf_get_current_comm\n");
        sigdel_data.comm[0] = 'X';
        sigdel_data.comm[1] = 'Y';
        sigdel_data.comm[2] = 'Z';
        sigdel_data.comm[3] = '\0';
    }

	if (bpf_perf_event_output(args, &events, BPF_F_CURRENT_CPU,
                              &sigdel_data, sizeof(sigdel_data)) != 0) {
        bpf_printk("ERROR:sigdel:bpf_perf_event_output\n");
    }

	bpf_printk("sigdel happened: %s\n", sigdel_data.comm);

	return 0;
}

char LICENSE[] SEC_GO("license") = "GPL";
