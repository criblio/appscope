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
    PERIODIC,    // This means to always ignore summarization rules
    EVENT_BASED, // This means to follow summarization rules
} control_type_t;

typedef enum {
    OPEN_PORTS,
    NET_CONNECTIONS,
    CONNECTION_OPEN,
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
    FS_STAT,
    EVT_NET,
    EVT_FS,
    EVT_STAT,
    EVT_ERR,
    EVT_DNS,
    EVT_PROTO,
    EVT_HREQ,
    EVT_HRES,
    EVT_DETECT,
    EVT_PAYLOAD,
    TLSRX,
    TLSTX
} metric_t;

// File types: stream or fd
typedef enum {
    FD,
    STREAM,
} fs_type_t;


// Interfaces
extern mtc_t *g_mtc;
extern ctl_t *g_ctl;

void initReporting(void);
void setReportingInterval(int);
void doErrorMetric(metric_t, control_type_t, const char *, const char *, void *);
void doProcMetric(metric_t, long long);
void doStatMetric(const char *, const char *, void *);
void doTotal(metric_t);
void doTotalDuration(metric_t);
void doEvent(void);
void doPayload(void);

#endif // __REPORT_H__
