#ifndef __STATE_H__
#define __STATE_H__

#include "report.h"


// Operational parameters (not configuration)
extern int g_urls;
extern int g_blockconn;


void initState();
void resetState();

void setVerbosity(unsigned verbosity);
void addSock(int fd, int type);
int doBlockConnection(int fd, const struct sockaddr *addr_arg);
void doSetConnection(int sd, const struct sockaddr *addr, socklen_t len,
           control_type_t endp);
int doSetAddrs(int sockfd);
int doAddNewSock(int sockfd);
int getDNSName(int sd, void *pkt, int pktlen);
int doURL(int sockfd, const void *buf, size_t len, metric_t src);
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


#endif // __STATE_H__
