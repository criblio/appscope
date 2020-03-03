#ifndef __REPORT_H__
#define __REPORT_H__

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "ctl.h"
#include "mtc.h"

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

// File types: stream or fd
typedef enum {
    FD,
    STREAM,
} fs_type_t;


// Interfaces
extern mtc_t* g_mtc;
extern ctl_t* g_ctl;


void setReportingInterval(int);


void sendProcessStartMetric();
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


#endif // __REPORT_H__
