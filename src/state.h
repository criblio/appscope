#ifndef __STATE_H__
#define __STATE_H__

#include "report.h"
#include "../contrib/tls/tls.h"

#ifdef __MACOS__
#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif
#endif // __MACOS__

typedef struct http_track_t {
    int status;
    uint64_t start_time;
    uint64_t duration;
    char method[16];
    char uri[256];
} http_track;

typedef struct http_list_t {
    struct http_list_t *next;
    uint64_t id;
    struct http_track_t htrack;
    PRIOMethods *ssl_methods;
    PRIOMethods *ssl_int_methods;
} http_list;

void initState();
void resetState();

void setVerbosity(unsigned);
void addSock(int, int);
int doBlockConnection(int, const struct sockaddr *);
void doSetConnection(int, const struct sockaddr *, socklen_t, control_type_t);
int doSetAddrs(int);
int doAddNewSock(int);
int getDNSName(int, void *, int);
int doURL(int, const void *, size_t, metric_t);
int doRecv(int, ssize_t);
int doSend(int, ssize_t);
void doAccept(int, struct sockaddr *, socklen_t *, char *);
void reportFD(int, control_type_t);
void reportAllFds(control_type_t);
void doRead(int, uint64_t, int, ssize_t, const char *);
void doWrite(int, uint64_t, int, const void *, ssize_t, const char *);
void doSeek(int, int, const char *);
void doStatPath(const char *, int, const char *);
void doStatFd(int, int, const char *);
int doDupFile(int, int, const char *);
int doDupSock(int, int);
void doDup(int, int, const char *, int);
void doDup2(int, int, int, const char *);
void doClose(int, const char *);
void doOpen(int, const char *, fs_type_t, const char *);
void doSendFile(int, int, uint64_t, int, const char *);
void doCloseAndReportFailures(int, int, const char *);
void doCloseAllStreams();
int remotePortIsDNS(int);
int sockIsTCP(int);
void doUpdateState(metric_t, int, ssize_t, const char *, const char *);
int doProtocol(int, void *, size_t, metric_t);

#endif // __STATE_H__
