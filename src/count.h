#ifndef __COUNT_H__
#define __COUNT_H__

#include <stddef.h>
#include <stdint.h>
#include <sys/socket.h>

#include "ctl.h"
#include "evt.h"
#include "log.h"
#include "out.h"

// Several control types, used in several areas
typedef enum {
    LOCAL,
    REMOTE,
    PERIODIC,
    EVENT_BASED,
} control_type_t;

typedef enum {
    OPEN_PORTS,
    NET_CONNECTIONS,
    CONNECTION_DURATION,
    PROC_CPU,
    PROC_MEM,
    PROC_THREAD,
    PROC_FD,
    PROC_CHILD,
    NETRX,
    NETTX,
    DNS,
    DNS_DURATION,
    FS_DURATION,
    FS_READ,
    FS_WRITE,
    FS_OPEN,
    FS_CLOSE,
    FS_SEEK,
    TOT_READ,
    TOT_WRITE,
    TOT_RX,
    TOT_TX,
    TOT_SEEK,
    TOT_STAT,
    TOT_OPEN,
    TOT_CLOSE,
    TOT_DNS,
    TOT_PORTS,
    TOT_TCP_CONN,
    TOT_UDP_CONN,
    TOT_OTHER_CONN,
    TOT_FS_DURATION,
    TOT_NET_DURATION,
    TOT_DNS_DURATION,
    NET_ERR_CONN,
    NET_ERR_RX_TX,
    NET_ERR_DNS,
    FS_ERR_OPEN_CLOSE,
    FS_ERR_STAT,
    FS_ERR_READ_WRITE,
} metric_t;

// File types; stream or fd
typedef enum {
    FD,
    STREAM,
} fs_type_t;


// Interfaces
extern log_t* g_log;
extern out_t* g_out;
extern evt_t* g_evt;
extern ctl_t* g_ctl;

extern proc_id_t g_proc;

// Operational parameters (not configuration)
extern int g_urls;
extern int g_blockconn;


void initCount();
void resetCount();
void setCountInterval();
void sendProcessStartMetric();

void scopeLog(const char* msg, int fd, cfg_log_level_t level);
void setVerbosity(unsigned verbosity);
void addSock(int fd, int type);
int doBlockConnection(int fd, const struct sockaddr *addr_arg);
void doErrorMetric(metric_t type, control_type_t source,
           const char *func, const char *name);
void doDNSMetricName(metric_t type, const char *domain, uint64_t duration);
void doProcMetric(metric_t type, long long measurement);
void doStatMetric(const char *op, const char *pathname);
void doFSMetric(metric_t type, int fd, control_type_t source,
           const char *op, ssize_t size, const char *pathname);
void doTotal(metric_t type);
void doTotalDuration(metric_t type);
void doNetMetric(metric_t type, int fd,
           control_type_t source, ssize_t size);
void doSetConnection(int sd, const struct sockaddr *addr, socklen_t len,
           control_type_t endp);
int doSetAddrs(int sockfd);
int doAddNewSock(int sockfd);
int getDNSName(int sd, void *pkt, int pktlen);
int doURL(int sockfd, const void *buf, size_t len, metric_t src);
int doAccessRights(struct msghdr *msg);
int doRecv(int sockfd, ssize_t rc);
int doSend(int sockfd, ssize_t rc);
void doAccept(int sd, struct sockaddr *addr, socklen_t *addrlen, char *func);
void reportFD(int fd, control_type_t source);
void reportAllFds(control_type_t source);
void doRead(int fd, uint64_t initialTime, int success, ssize_t bytes,
           const char* func);
void doWrite(int fd, uint64_t initialTime, int success, const void* buf,
           ssize_t bytes, const char* func);
void doSeek(int fd, int success, const char* func);
void doStatPath(const char* path, int rc, const char* func);
void doStatFd(int fd, int rc, const char* func);
int doDupFile(int newfd, int oldfd, const char *func);
int doDupSock(int oldfd, int newfd);
void doDup(int fd, int rc, const char* func, int copyNet);
void doDup2(int oldfd, int newfd, int rc, const char* func);
void doClose(int fd, const char *func);
void doOpen(int fd, const char *path, fs_type_t type, const char *func);
void doSendFile(int out_fd, int in_fd, uint64_t initialTime, int rc,
           const char* func);
void doCloseAndReportFailures(int fd, int success, const char* func);
void doCloseAllStreams();
int remotePortIsDNS(int sockfd);
int sockIsTCP(int sockfd);


#endif // __COUNT_H__
