#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>

#define COMM_LEN 128
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
} events SEC(".maps");

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

#define LAST_32_BITS(x) x & 0xFFFFFFFF
#define FIRST_32_BITS(x) x >> 32

SEC("tracepoint/signal/signal_deliver")
int sig_deliver(struct sigdel_args_t *args)
{
	struct sigdel_data_t sigdel_data = {};
	u64 pid_tgid;

	pid_tgid = bpf_get_current_pid_tgid();
	sigdel_data.pid = LAST_32_BITS(pid_tgid);

    bpf_probe_read(&sigdel_data.sig, sizeof(sigdel_data.sig), &args->sig);
    bpf_probe_read(&sigdel_data.errno, sizeof(sigdel_data.errno), &args->errno);
    bpf_probe_read(&sigdel_data.code, sizeof(sigdel_data.errno), &args->code);
    bpf_probe_read(&sigdel_data.sa_handler, sizeof(sigdel_data.errno), &args->sa_handler);
	bpf_get_current_comm(sigdel_data.comm, sizeof(sigdel_data.comm));

	bpf_perf_event_output(args, &events, BPF_F_CURRENT_CPU,
                          &sigdel_data, sizeof(sigdel_data));

	//bpf_printk("sigdel happened\n");

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
