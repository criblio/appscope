#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <sys/epoll.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>

#include "dbg.h"
#include "circbuf.h"
#include "test.h"
#include "state.h"
#include "report.h"
#include "wrap.h"
#include "plattime.h"


// Normally declared by wrap.c
interposed_funcs g_fn;

#define BUFSIZE 500

event_t evtBuf[BUFSIZE] = {0};
int evtBufNext = 0;
event_t mtcBuf[BUFSIZE] = {0};
int mtcBufNext = 0;

void __wrap_ctlSendEvent(ctl_t* ctl, event_t* event, uint64_t uid, proc_id_t* proc)
{
    // Store event for later inspection
    memcpy(&evtBuf[evtBufNext++], event, sizeof(*event));
    if (evtBufNext >= BUFSIZE) fail();

    //__real_ctlSendEvent(ctl, event, uid, proc);
}

int __wrap_mtcSendMetric(mtc_t* mtc, event_t* metric)
{
    // Store metric for later inspection
    memcpy(&mtcBuf[mtcBufNext++], metric, sizeof(*metric));
    if (mtcBufNext >= BUFSIZE) fail();

    //return __real_mtcSendMetric(mtc, metric);
    return 0;
}

int
eventCalls(const char* str)
{
    int i, returnVal = 0;
    for (i=0; i<evtBufNext; i++) {
        if (!str || !strcmp(evtBuf[i].name, str)) returnVal++;
    }
    return returnVal;
}

int
metricCalls(const char* str)
{
    int i, returnVal = 0;
    for (i=0; i<mtcBufNext; i++) {
        if (!str || !strcmp(mtcBuf[i].name, str)) returnVal++;
    }
    return returnVal;
}

int
eventValues(const char* str)
{
    int i, returnVal = 0;
    for (i=0; i < evtBufNext; i++) {
        if (!str || !strcmp(evtBuf[i].name, str)) {
            returnVal += evtBuf[i].value.integer;
        }
    }
    return returnVal;
}

int
metricValues(const char* str)
{
    int i, returnVal = 0;
    for (i=0; i < mtcBufNext; i++) {
        if (!str || !strcmp(mtcBuf[i].name, str)) {
            returnVal += mtcBuf[i].value.integer;
        }
    }
    return returnVal;
}

void
clearTestData(void)
{
    memset(&evtBuf, 0, sizeof(evtBuf));
    evtBufNext = 0;
    memset(&mtcBuf, 0, sizeof(mtcBuf));
    mtcBufNext = 0;
}

static void
init_g_fn()
{
    // Currently this just initializes stuff that is used in os.c
    // if maintaining this becomes a pain, we could refactor part of the
    // constructor in wrap.c to be a separate function we could call here.
    g_fn.open = dlsym(RTLD_NEXT, "open");
    g_fn.close = dlsym(RTLD_NEXT, "close");
    g_fn.read = dlsym(RTLD_NEXT, "read");
    g_fn.socket = dlsym(RTLD_NEXT, "socket");
    g_fn.sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    g_fn.recvmsg = dlsym(RTLD_NEXT, "recvmsg");
    g_fn.sigaction = dlsym(RTLD_NEXT, "sigaction");
    g_fn.__xstat = dlsym(RTLD_NEXT, "__xstat");
}

static void
init_g_proc()
{
    g_proc.pid = 50;
    g_proc.ppid = 49;
    strcpy(g_proc.hostname, "hostname");
    strcpy(g_proc.procname, "procname");
    g_proc.cmd = strdup("command with args");
    strcpy(g_proc.id, "procid");
}

static int
countTestSetup(void** state)
{
    // init objects unique to this test that are external to count
    initTime();
    init_g_fn();

    // init objects that count has
    init_g_proc();
    g_log = logCreate();
    g_mtc = mtcCreate();
    g_ctl = ctlCreate();

    g_urls = 0;
    g_blockconn = 0;

    initState();

    // Call the general groupSetup() too.
    return groupSetup(state);
}

static int
countTestTeardown(void** state)
{
    logDestroy(&g_log);
    mtcDestroy(&g_mtc);
    ctlDestroy(&g_ctl);

    // Call the general groupTeardown() too.
    return groupTeardown(state);
}

static void
nothingCrashesBeforeAnyInit(void** state)
{
    resetState();
    setReportingInterval(10);
    sendProcessStartMetric();
    scopeLog("hey", 1, CFG_LOG_ERROR);
    setVerbosity(9);
    addSock(3, SOCK_SEQPACKET);
    doBlockConnection(4, NULL);
    doErrorMetric(NET_ERR_CONN, PERIODIC, "A", "B");
    doDNSMetricName(DNS, "something.com", 1234);
    doProcMetric(PROC_CPU, 2345);
    doStatMetric("statFunc", "/the/path/to/something");
    doFSMetric(FS_DURATION, 5, PERIODIC,
               "writeFunc", 5432, "/the/path/to/something/else");
    doTotal(TOT_READ);
    doTotalDuration(TOT_DNS_DURATION);
    doNetMetric(NETTX, 6, EVENT_BASED, 6543);
    doSetConnection(7, NULL, 3, LOCAL);
    doSetAddrs(8);
    doAddNewSock(9);
    getDNSName(10, NULL, 0);
    doURL(11, NULL, 0, NETRX);
    doRecv(12, 4312);
    doSend(13, 6682);
    doAccept(14, NULL, 0, "acceptFunc");
    reportFD(15, EVENT_BASED);
    reportAllFds(PERIODIC);
    doRead(16, 987, 1, 13, "readFunc");
    doWrite(17, 876, 1, NULL, 0, "writeFunc");
    doSeek(18, 1, "seekymcseekface");
    doStatPath("/pathy/path", 0, "statymcstatface");
    doStatFd(19, 0, "toomuchstat4u");
    doDupFile(20, 21, "dup");
    doDupSock(22, 23);
    doDup(24, 0, "dupFunc", 1);
    doDup2(25, 26, 0, "dup2Func");
    doClose(26, "closeFunc");
    doOpen(27, "/the/file/path", FD, "openFunc");
    doSendFile(28, 29, 23548, 0, "sendFileFunc");
    doCloseAndReportFailures(30, 1, "closeFunc");
    doCloseAllStreams();
    remotePortIsDNS(31);
    sockIsTCP(32);
    clearTestData();
}

static void
initStateDoesNotCrash(void** state)
{
    initState();
    clearTestData();
}

static void
doReadFileNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(9);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doRead(16, 987, 1, 0, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Without read/write summarization, every doRead is output
    clearTestData();
    doRead(16, 987, 1, 13, "readFunc");
    doRead(16, 987, 1, 13, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 2);
    assert_int_equal(metricValues("fs.read"), 2*13);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventValues("fs.read"), 2*13);

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal shouldn't output fs.read or fs.op.close.  It's already reported
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_READ);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doReadFileSummarizedOpenCloseNotSummarized(void** state)
{
    clearTestData();
    setVerbosity(6);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doRead(16, 987, 1, 0, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // With read/write summarization, no doRead is output at the time
    clearTestData();
    doRead(16, 987, 1, 13, "readFunc");
    doRead(16, 987, 1, 13, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventValues("fs.read"), 2*13);

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal should output fs.read activity from above
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_READ);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls("fs.read"), 1);
    assert_int_equal(metricValues("fs.read"), 2*13);
    assert_int_equal(metricCalls("fs.op.close"), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doReadFileFullSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 0);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doRead(16, 987, 1, 0, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // With read/write summarization, no doRead is output at the time
    clearTestData();
    doRead(16, 987, 1, 13, "readFunc");
    doRead(16, 987, 1, 13, "readFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventValues("fs.read"), 2*13);

    // With open/close summarization, doClose does not output either
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 0);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal should output fs.read
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_READ);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls("fs.open"), 1);
    assert_int_equal(metricCalls("fs.read"), 1);
    assert_int_equal(metricValues("fs.read"), 2*13);
    assert_int_equal(metricCalls("fs.close"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.op.close"), 0);
}

static void
doWriteFileNoSummarization(void** state)
{
    char* buf = "hey.\n";

    clearTestData();
    setVerbosity(9);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doWrite(16, 987, 1, buf, 0, "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Without read/write summarization, every doWrite is output
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 2);
    assert_int_equal(metricValues("fs.write"), 2*sizeof(buf));
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventValues("fs.write"), 2*sizeof(buf));

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal shouldn't output fs.write or fs.op.close.  It's already reported
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_WRITE);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doWriteFileSummarizedOpenCloseNotSummarized(void** state)
{
    char* buf = "hey.\n";

    clearTestData();
    setVerbosity(6);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doWrite(16, 987, 1, buf, 0, "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // With read/write summarization, no doWrite is output at the time
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventValues("fs.write"), 2*sizeof(buf));

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal should output fs.write activity from above
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_WRITE);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls("fs.write"), 1);
    assert_int_equal(metricValues("fs.write"), 2*sizeof(buf));
    assert_int_equal(metricCalls("fs.op.close"), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doWriteFileFullSummarization(void** state)
{
    char* buf = "hey.\n";

    clearTestData();
    setVerbosity(5);
    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 0);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Zeros should not be reported on any interface
    doWrite(16, 987, 1, buf, 0, "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // With read/write summarization, no doWrite is output at the time
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventValues("fs.write"), 2*sizeof(buf));

    // With open/close summarization, doClose does not output either
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 0);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal should output fs.write
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_WRITE);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls("fs.open"), 1);
    assert_int_equal(metricCalls("fs.write"), 1);
    assert_int_equal(metricValues("fs.write"), 2*sizeof(buf));
    assert_int_equal(metricCalls("fs.close"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.op.close"), 0);
}

static void
doRecvNoSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(9);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doRecv(16, 0);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Without rx/tx summarization, every doRecv is output
    clearTestData();
    doRecv(16, 13);
    doRecv(16, 13);
    assert_int_equal(metricCalls("net.rx"), 2);
    assert_int_equal(metricValues("net.rx"), 2*13);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventValues("net.rx"), 2*13);

    // Without open/close summarization, every doClose it output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal shouldn't output anything.  It's already reported
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_RX);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doRecvSummarizedOpenCloseNotSummarized(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(7);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doRecv(16, 0);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // With rx/tx summarization, no doRecv is output at the time
    clearTestData();
    doRecv(16, 13);
    doRecv(16, 13);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventValues("net.rx"), 2*13);

    // Without open/close summarization, every doClose it output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal should output net.rx activity from above.
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 1);
    assert_int_equal(metricValues("net.rx"), 2*13);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doRecvFullSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(6);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doRecv(16, 0);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // With rx/tx summarization, no doRecv is output at the time
    clearTestData();
    doRecv(16, 13);
    doRecv(16, 13);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventValues("net.rx"), 2*13);

    // With open/close summarization, doClose does not output either
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal should output net.rx activity from above.
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 1);
    assert_int_equal(metricValues("net.rx"), 2*13);
    assert_int_equal(metricCalls("net.tcp"), 0);     // Interesting...
    assert_int_equal(metricCalls("net.port"), 0);    // need high water mark?
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doSendNoSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(9);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doSend(16, 0);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Without rx/tx summarization, every doSend is output
    clearTestData();
    doSend(16, 13);
    doSend(16, 13);
    assert_int_equal(metricCalls("net.tx"), 2);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventValues("net.tx"), 2*13);

    // Without open/close summarization, every doClose it output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal shouldn't output anything.  It's already reported
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_TX);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doSendSummarizedOpenCloseNotSummarized(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(7);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doSend(16, 0);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // With rx/tx summarization, no doSend is output at the time
    clearTestData();
    doSend(16, 13);
    doSend(16, 13);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventValues("net.tx"), 2*13);

    // Without open/close summarization, every doClose it output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 1);
    assert_int_equal(metricCalls("net.port"), 1);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal should output net.tx activity from above.
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 1);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doSendFullSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "13456", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(6);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    // Well, unless it's a change to zero for a gauge
    doSend(16, 0);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // With rx/tx summarization, no doSend is output at the time
    clearTestData();
    doSend(16, 13);
    doSend(16, 13);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventValues("net.tx"), 2*13);

    // With open/close summarization, doClose does not output either
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal should output net.tx activity from above.
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 1);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(metricCalls("net.tcp"), 0);     // Interesting...
    assert_int_equal(metricCalls("net.port"), 0);    // need high water mark?
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doSeekNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(8);

    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Totals should not be reported if zero
    doTotal(TOT_SEEK);
    assert_int_equal(metricCalls("fs.op.seek"), 0);
    assert_int_equal(eventCalls("fs.op.seek"), 0);

    // Without seek summarization, every doSeek is output
    clearTestData();
    doSeek(16, 1, "readFunc");
    doSeek(16, 1, "readFunc");
    assert_int_equal(metricCalls("fs.op.seek"), 2);
    assert_int_equal(metricValues("fs.op.seek"), 2);
    assert_int_equal(eventCalls("fs.op.seek"), 2);
    assert_int_equal(eventValues("fs.op.seek"), 2);

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.op.seek"), 0);
    assert_int_equal(eventCalls("fs.op.seek"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal shouldn't output fs.op.seek or fs.op.close.  It's already reported
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_SEEK);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doSeekSummarization(void** state)
{
    clearTestData();
    setVerbosity(7);

    doOpen(16, "/the/file/path", FD, "openFunc");
    assert_int_equal(metricCalls("fs.op.open"), 1);
    assert_int_equal(eventCalls("fs.op.open"), 1);

    // Totals should not be reported if zero
    doTotal(TOT_SEEK);
    assert_int_equal(metricCalls("fs.op.seek"), 0);
    assert_int_equal(eventCalls("fs.op.seek"), 0);

    // With seek summarization, no doSeek is output
    clearTestData();
    doSeek(16, 1, "readFunc");
    doSeek(16, 1, "readFunc");
    assert_int_equal(metricCalls("fs.op.seek"), 0);
    assert_int_equal(metricValues("fs.op.seek"), 0);
    assert_int_equal(eventCalls("fs.op.seek"), 2);
    assert_int_equal(eventValues("fs.op.seek"), 2);

    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("fs.op.seek"), 0);
    assert_int_equal(eventCalls("fs.op.seek"), 0);
    assert_int_equal(metricCalls("fs.op.close"), 1);
    assert_int_equal(eventCalls("fs.op.close"), 1);

    // doTotal should output fs.op.seek per the activity above.
    clearTestData();
    doTotal(TOT_OPEN);
    doTotal(TOT_SEEK);
    doTotal(TOT_CLOSE);
    assert_int_equal(metricCalls("fs.seek"), 1);
    assert_int_equal(metricValues("fs.seek"), 2);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doStatPathNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(7);

    // Totals should not be reported if zero
    doTotal(TOT_STAT);
    assert_int_equal(metricCalls("fs.op.stat"), 0);
    assert_int_equal(eventCalls("fs.op.stat"), 0);

    // Without stat summarization, every doStat is output
    clearTestData();
    doStatPath("/the/path", 0, "statFunc");
    doStatPath("/the/path", 0, "statFunc");
    assert_int_equal(metricCalls("fs.op.stat"), 2);
    assert_int_equal(metricValues("fs.op.stat"), 2);
    assert_int_equal(eventCalls("fs.op.stat"), 2);
    assert_int_equal(eventValues("fs.op.stat"), 2);

    // doTotal shouldn't output fs.op.seek or fs.op.close.  It's already reported
    clearTestData();
    doTotal(TOT_STAT);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doStatPathSummarization(void** state)
{
    clearTestData();
    setVerbosity(6);

    // Totals should not be reported if zero
    doTotal(TOT_STAT);
    assert_int_equal(metricCalls("fs.op.stat"), 0);
    assert_int_equal(eventCalls("fs.op.stat"), 0);

    // Without stat summarization, every doStat is output
    clearTestData();
    doStatPath("/the/path", 0, "statFunc");
    doStatPath("/the/path", 0, "statFunc");
    assert_int_equal(metricCalls("fs.op.stat"), 0);
    assert_int_equal(metricValues("fs.op.stat"), 0);
    assert_int_equal(eventCalls("fs.op.stat"), 2);
    assert_int_equal(eventValues("fs.op.stat"), 2);

    // doTotal shouldn't output fs.op.seek or fs.op.close.  It's already reported
    clearTestData();
    doTotal(TOT_STAT);
    assert_int_equal(metricCalls("fs.stat"), 1);
    assert_int_equal(metricValues("fs.stat"), 2);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doStatFdNoSummarization(void** state)
{
     clearTestData();
     setVerbosity(7);
     doOpen(16, "/the/file/path", FD, "openFunc");
     assert_int_equal(metricCalls("fs.op.open"), 1);
     assert_int_equal(eventCalls("fs.op.open"), 1);

     // Totals should not be reported if zero
     doTotal(TOT_STAT);
     assert_int_equal(metricCalls("fs.op.stat"), 0);
     assert_int_equal(eventCalls("fs.op.stat"), 0);

     // Without stat summarization, every doStatFd is output
     clearTestData();
     doStatFd(16, 0, "statFunc");
     doStatFd(16, 0, "statFunc");
     assert_int_equal(metricCalls("fs.op.stat"), 2);
     assert_int_equal(metricValues("fs.op.stat"), 2);
     assert_int_equal(eventCalls("fs.op.stat"), 2);
     assert_int_equal(eventValues("fs.op.stat"), 2);

     // Without open/close summarization, every doClose is output
     clearTestData();
     doClose(16, "closeFunc");
     assert_int_equal(metricCalls("fs.op.stat"), 0);
     assert_int_equal(eventCalls("fs.op.stat"), 0);
     assert_int_equal(metricCalls("fs.op.close"), 1);
     assert_int_equal(eventCalls("fs.op.close"), 1);

     // doTotal shouldn't output fs.op.stat or fs.op.close.  It's already reported
     clearTestData();
     doTotal(TOT_OPEN);
     doTotal(TOT_STAT);
     doTotal(TOT_CLOSE);
     assert_int_equal(metricCalls(NULL), 0);
     assert_int_equal(eventCalls(NULL), 0);
}

static void
doStatFdSummarization(void** state)
{
     clearTestData();
     setVerbosity(6);
     doOpen(16, "/the/file/path", FD, "openFunc");
     assert_int_equal(metricCalls("fs.op.open"), 1);
     assert_int_equal(eventCalls("fs.op.open"), 1);

     // Totals should not be reported if zero
     doTotal(TOT_STAT);
     assert_int_equal(metricCalls("fs.op.stat"), 0);
     assert_int_equal(eventCalls("fs.op.stat"), 0);

     // With stat summarization, doStatFd is is not output
     clearTestData();
     doStatFd(16, 0, "statFunc");
     doStatFd(16, 0, "statFunc");
     assert_int_equal(metricCalls("fs.op.stat"), 0);
     assert_int_equal(metricValues("fs.op.stat"), 0);
     assert_int_equal(eventCalls("fs.op.stat"), 2);
     assert_int_equal(eventValues("fs.op.stat"), 2);

     // Without open/close summarization, every doClose is output
     clearTestData();
     doClose(16, "closeFunc");
     assert_int_equal(metricCalls("fs.op.stat"), 0);
     assert_int_equal(eventCalls("fs.op.stat"), 0);
     assert_int_equal(metricCalls("fs.op.close"), 1);
     assert_int_equal(eventCalls("fs.op.close"), 1);

     // doTotal shouldn't output fs.op.stat or fs.op.close.  It's already reported
     clearTestData();
     doTotal(TOT_OPEN);
     doTotal(TOT_STAT);
     doTotal(TOT_CLOSE);
     assert_int_equal(metricCalls("fs.stat"), 1);
     assert_int_equal(metricValues("fs.stat"), 2);
     assert_int_equal(eventCalls(NULL), 0);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    // Make sure that the functions can be hit before anything
    // is initialized (before constuctor).
    const struct CMUnitTest preInitTests[] = {
        cmocka_unit_test(nothingCrashesBeforeAnyInit),
    };
    int pre_init_errors = cmocka_run_group_tests(preInitTests, NULL, NULL);

    // Run tests
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(initStateDoesNotCrash),
        cmocka_unit_test(doReadFileNoSummarization),
        cmocka_unit_test(doReadFileSummarizedOpenCloseNotSummarized),
        cmocka_unit_test(doReadFileFullSummarization),
        cmocka_unit_test(doWriteFileNoSummarization),
        cmocka_unit_test(doWriteFileSummarizedOpenCloseNotSummarized),
        cmocka_unit_test(doWriteFileFullSummarization),
        cmocka_unit_test(doRecvNoSummarization),
        cmocka_unit_test(doRecvSummarizedOpenCloseNotSummarized),
        cmocka_unit_test(doRecvFullSummarization),
        cmocka_unit_test(doSendNoSummarization),
        cmocka_unit_test(doSendSummarizedOpenCloseNotSummarized),
        cmocka_unit_test(doSendFullSummarization),
        cmocka_unit_test(doSeekNoSummarization),
        cmocka_unit_test(doSeekSummarization),
        cmocka_unit_test(doStatPathNoSummarization),
        cmocka_unit_test(doStatPathSummarization),
        cmocka_unit_test(doStatFdNoSummarization),
        cmocka_unit_test(doStatFdSummarization),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    int test_errors = cmocka_run_group_tests(tests, countTestSetup, countTestTeardown);
    return pre_init_errors || test_errors;
}

