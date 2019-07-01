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
#define PORT 8888 //8125
#define SERVER "127.0.0.1"

// Initial size of net array for state
#define NET_ENTRIES 1024
#define MAX_FDS 4096
#define PROTOCOL_STR 8
#define MAX_HOSTNAME 128
#define MAX_PROCNAME 128

#define STATSD_OPENPORTS "net.port:%d|g|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n"

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

static inline void atomicAdd(int *ptr, int val) {
    (void)__sync_add_and_fetch(ptr, val);
}

static inline void atomicSub(int *ptr, int val)
{
	(void)__sync_sub_and_fetch(ptr, val);
}

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);
    
static void (*g_vsyslog)(int, const char *, va_list);
static int (*g_close)(int);
static int (*g_close$NOCANCEL)(int);
static int (*g_close_nocancel)(int);
static int (*g_guarded_close_np)(int, void *);
static int (*g_shutdown)(int, int);
static int (*g_socket)(int, int, int);
static int (*g_listen)(int, int);
static int (*g_bind)(int, const struct sockaddr *, socklen_t);
static ssize_t (*g_read)(int, void *, size_t);
static ssize_t (*g_write)(int, const void *, size_t);
static ssize_t (*g_send)(int, const void *, size_t, int);
static ssize_t (*g_sendto)(int, const void *, size_t, int,
                              const struct sockaddr *, socklen_t);
static ssize_t (*g_sendmsg)(int, const struct msghdr *, int);
static ssize_t (*g_recv)(int, void *, size_t, int);
static ssize_t (*g_recvfrom)(int sockfd, void *buf, size_t len, int flags,
                                struct sockaddr *src_addr, socklen_t *addrlen);
static ssize_t (*g_recvmsg)(int, struct msghdr *, int);

extern int osGetProcname(char *, int);

static int g_sock = 0;
static struct sockaddr_in g_saddr;
static operations_info g_ops;
static net_info *g_netinfo;
static int g_numNinfo;
static char g_hostname[MAX_HOSTNAME];
static char g_procname[MAX_PROCNAME];
static int g_openPorts = 0;
static int g_activeConnections = 0;

// These need to come from a config file
#define LOG_FILE 1  // eventually an enum for file, syslog, shared memory 
static bool g_log = TRUE;
static const char g_logFile[] = "/tmp/scope.log";
static unsigned int g_logOp = LOG_FILE;
static int g_logfd = -1;
