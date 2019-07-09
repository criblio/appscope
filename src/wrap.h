#ifndef _CONFIG_H_
#define _CONFIG_H_

#define _GNU_SOURCE
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>

#ifdef __MACOS__
#include "../os/macOS/os.h"
#elif defined (__LINUX__)
#include "../os/linux/os.h"
#endif

#define DEBUG 0
#define TRUE 1
#define FALSE 0
#define EXPORT __attribute__((visibility("default")))
#define EXPORTOFF  __attribute__((visibility("hidden")))
#define EXPORTON __attribute__((visibility("default")))

// Use these only if a config file is not accesible
#define PORT 8125
#define SERVER "172.16.198.1" //"127.0.0.1"

// Initial size of net array for state
#define NET_ENTRIES 1024
#define MAX_FDS 4096
#define PROTOCOL_STR 8
#define MAX_HOSTNAME 255
#define MAX_PROCNAME 128

#define STATSD_OPENPORTS "net.port:%d|g|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n"
#define STATSD_TCPCONNS "net.tcp:%d|g|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n"
#define STATSD_ACTIVECONNS "net.conn:%d|c|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n"

#define STATSD_PROCMEM "proc.mem:%lu|g|#proc:%s,pid:%d,host:%s\n"
#define STATSD_PROCCPU "proc.cpu:%lu|g|#proc:%s,pid:%d,host:%s\n"
#define STATSD_PROCTHREAD "proc.thread:%d|g|#proc:%s,pid:%d,host:%s\n"
#define STATSD_PROCFD "proc.fd:%d|g|#proc:%s,pid:%d,host:%s\n"
#define STATSD_PROCCHILD "proc.child:%d|g|#proc:%s,pid:%d,host:%s\n"

/*
 * NOTE: The following constants are ALL going away
 * They were used for initial signs of life in OSX
 * Left here so that compilations works
 * Using them if we want to see if a function is interposed or called
 */
#define STATSD_READ "cribl.scope.calls.read.bytes:%d|c\n"
#define STATSD_WRITE "cribl.scope.calls.write.bytes:%d|c\n"
#define STATSD_VSYSLOG "cribl.scope.calls.vsyslog|c\n"
#define STATSD_SEND "cribl.scope.calls.send.bytes:%d|c\n"
#define STATSD_SENDTO "cribl.scope.calls.sendto.bytes:%d|c\n"
#define STATSD_SENDMSG "cribl.scope.calls.sendmsg|c\n"
#define STATSD_RECV "cribl.scope.calls.recv.bytes:%d|c\n"


#ifndef bool
typedef unsigned int bool;
#endif

enum metric_t {
    OPEN_PORTS,
    TCP_CONNECTIONS,
    ACTIVE_CONNECTIONS,
    PROC_CPU,
    PROC_MEM,
    PROC_THREAD,
    PROC_FD,
    PROC_CHILD
};

typedef struct operations_info_t {
    unsigned int udp_blocks;
    unsigned int udp_errors;
    unsigned int init_errors;
    unsigned int interpose_errors;
    char *errMsg[64];
} operations_info;

typedef struct net_info_t {
    int fd;
    int type;
    bool listen;
    bool accept;
    in_port_t port;
} net_info;

typedef struct interposed_funcs_t {
    void (*vsyslog)(int, const char *, va_list);
    int (*close)(int);
    int (*shutdown)(int, int);
    int (*socket)(int, int, int);
    int (*listen)(int, int);
    int (*bind)(int, const struct sockaddr *, socklen_t);
    int (*connect)(int, const struct sockaddr *, socklen_t);
    int (*accept)(int, struct sockaddr *, socklen_t *);
    int (*accept4)(int, struct sockaddr *, socklen_t *, int);
    ssize_t (*read)(int, void *, size_t);
    ssize_t (*write)(int, const void *, size_t);
    ssize_t (*send)(int, const void *, size_t, int);
    ssize_t (*sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
    ssize_t (*sendmsg)(int, const struct msghdr *, int);
    ssize_t (*recv)(int, void *, size_t, int);
    ssize_t (*recvfrom)(int sockfd, void *buf, size_t len, int flags,
                                struct sockaddr *src_addr, socklen_t *addrlen);
    ssize_t (*recvmsg)(int, struct msghdr *, int);
    // macOS
    int (*close$NOCANCEL)(int);
    int (*close_nocancel)(int);
    int (*guarded_close_np)(int, void *);
} interposed_funcs;
    
static inline void atomicAdd(int *ptr, int val) {
    (void)__sync_add_and_fetch(ptr, val);
}

static inline void atomicSub(int *ptr, int val)
{
	(void)__sync_sub_and_fetch(ptr, val);
}

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);

#endif // _CONFIG_H_
