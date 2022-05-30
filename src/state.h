#ifndef __STATE_H__
#define __STATE_H__

#include <sys/socket.h>
#include "pcre2posix.h"

#include "runtimecfg.h"
#include "linklist.h"
#include "report.h"
#include "../contrib/tls/tls.h"

#ifdef __APPLE__
#ifndef AF_NETLINK
#define AF_NETLINK 16
#endif
#endif // __APPLE__


/**
 * Options for `do*()` indicating how to interpret the `buf` and `len` values.
 */
typedef enum {
    BUF, // `buf` points to raw bytes, `len` is the number of bytes
    MSG, // `buf` points to a `struct msghdr`, `len` is unused
    IOV, // `buf` points to a `struct iovec` array, `len` is the array length
    NONE // `buf` is unavailable, typically NULL but `len` is what?
} src_data_t;

/**
 * File content type
 */
typedef enum {
    FS_CONTENT_UNKNOWN, // File content type undetermined
    FS_CONTENT_BINARY,  // File content type binary
    FS_CONTENT_TEXT     // File content type text
} fs_content_type_t;

void initState();
void resetState();

void setVerbosity(unsigned);
void setCategoryMtcEnable(unsigned, unsigned, unsigned, unsigned, unsigned);
bool fsMtcEnable(void);
bool netMtcEnable(void);
bool httpMtcEnable(void);
bool dnsMtcEnable(void);
bool procMtcEnable(void);
void addSock(int, int, int);
int doBlockConnection(int, const struct sockaddr *);
void doSetConnection(int, const struct sockaddr *, socklen_t, control_type_t);
int doSetAddrs(int);
int doAddNewSock(int);
int getDNSName(int, void *, int);
bool getDNSAnswer(int, char *, size_t, src_data_t);
int doURL(int, const void *, size_t, metric_t);
int doRecv(int, ssize_t, const void *, size_t, src_data_t);
int doSend(int, ssize_t, const void *, size_t, src_data_t);
void doAccept(int, int, struct sockaddr *, socklen_t *, char *);
void reportFD(int, control_type_t);
void reportAllFds(control_type_t);
void doRead(int, uint64_t, int, const void *, ssize_t, const char *, src_data_t, size_t);
void doWrite(int, uint64_t, int, const void *, ssize_t, const char *, src_data_t, size_t);
void doSeek(int, int, const char *);
void doStatPath(const char *, int, const char *);
void doStatFd(int, int, const char *);
int doDupFile(int, int, const char *);
int doDupSock(int, int);
void doDup(int, int, const char *, int);
void doDup2(int, int, int, const char *);
void doDelete(const char *, const char *);
void doClose(int, const char *);
void doOpen(int, const char *, fs_type_t, const char *);
void doSendFile(int, int, uint64_t, int, const char *);
void doCloseAndReportFailures(int, int, const char *);
void doCloseAllStreams();
int remotePortIsDNS(int);
int sockIsTCP(int);
void doUpdateState(metric_t, int, ssize_t, const char *, const char *);
int doProtocol(uint64_t, int, void *, size_t, metric_t, src_data_t);
int doSSL(uint64_t, int, void *, size_t, metric_t, src_data_t, char *);
bool addProtocol(request_t *);
bool delProtocol(request_t *);
void setRemoteClose(int, int);
void setFSContentType(int, fs_content_type_t);
fs_content_type_t getFSContentType(int);

#endif // __STATE_H__
