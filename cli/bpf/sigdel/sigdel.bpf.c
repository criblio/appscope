#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>

#define FILTER_SIGS 1
#define COMM_LEN 128
#define LAST_32_BITS(x) x & 0xFFFFFFFF
#define FIRST_32_BITS(x) x >> 32

// Build on ubuntu 18
#ifndef BPF_F_CURRENT_CPU
#define BPF_F_INDEX_MASK 0xffffffffULL
#define BPF_F_CURRENT_CPU BPF_F_INDEX_MASK
#endif

/*
 * Signals are filtered. We want to inform any user mode
 * readers about signals that are likley to cause a crash
 * in an application. At this point the signals we pass are:
 * SIGSEGV, SIGBUS, SIGILL and SIGFPE
 *
 * bpf_printk output:
 * sudo cat /sys/kernel/tracing/trace_pipe
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
    int nspid;
    int sig;
    int errno;
    int code;
    int uid;
    int gid;
    unsigned long sa_handler;
    unsigned long sa_flags;
    unsigned char comm[COMM_LEN];
};

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

char LICENSE[] SEC_GO("license") = "GPL";

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u32));
} events SEC_GO(".maps");

SEC_GO("tracepoint/signal/signal_deliver")
int sig_deliver(struct sigdel_args_t *args)
{
#if FILTER_SIGS > 0
    // SIGSEGV, SIGBUS, SIGILL and SIGFPE
    if ((args->sa_handler == 0) ||
        ((args->sig != 11) && (args->sig != 7) &&
         (args->sig != 4) && (args->sig != 8))) {
        return 1;
    }
#endif
    struct sigdel_data_t sigdel_data = {};
    u64 pid_tgid, uid_gid;
    struct task_struct *task;
    struct pid *pid;
    unsigned int level, nr, hostpid;
    struct upid upid;
    struct ns_common ns;

    /*
     * Note that we are not using bpf_probe_read_kernel_btf
     * here as it does not work on older distros. So, an
     * extra step is required to get the task.
     */
    u64 taskadd = bpf_get_current_task();
    if (!taskadd) {
        bpf_printk("ERROR:%d:sigdel:get current task\n", __LINE__);
        return 0;
    }

    // TODO: add error checking
    bpf_probe_read_kernel(&task, sizeof(task), &taskadd);
    bpf_probe_read_kernel(&pid, sizeof(pid), &task->thread_pid);
    bpf_probe_read_kernel(&level, sizeof(level), &pid->level);
    bpf_probe_read_kernel(&upid, sizeof(upid), &pid->numbers[level]);
    bpf_probe_read_kernel(&ns, sizeof(ns), &upid.ns->ns);
    bpf_probe_read_kernel(&nr, sizeof(nr), &upid.nr);
    bpf_probe_read_kernel(&hostpid, sizeof(hostpid), &task->pid);

    //bpf_printk("sigdel: %s: ino %u nr %u\n", task->comm, ns.inum, nr);

    sigdel_data.nspid = nr;
    sigdel_data.pid = hostpid;

    uid_gid = bpf_get_current_uid_gid();
    sigdel_data.uid = LAST_32_BITS(uid_gid);
    sigdel_data.gid = FIRST_32_BITS(uid_gid);

    if (bpf_get_current_comm(sigdel_data.comm, sizeof(sigdel_data.comm)) != 0) {
        bpf_printk("ERROR:sigdel:bpf_get_current_comm\n");
        sigdel_data.comm[0] = 'X';
        sigdel_data.comm[1] = 'Y';
        sigdel_data.comm[2] = 'Z';
        sigdel_data.comm[3] = '\0';
    }

    sigdel_data.sig = args->sig;
    sigdel_data.errno = args->errno;
    sigdel_data.code = args->code;
    sigdel_data.sa_handler = args->sa_handler;

    if (bpf_perf_event_output(args, &events, BPF_F_CURRENT_CPU,
                              &sigdel_data, sizeof(sigdel_data)) != 0) {
        bpf_printk("ERROR:sigdel:bpf_perf_event_output\n");
    }

    return 0;
}
