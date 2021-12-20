#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "dbg.h"
#include "fn.h"
#include "plattime.h"
#include "report.h"
#include "runtimecfg.h"
#include "state.h"
#include "com.h"
#include "test.h"

#define BUFSIZE 500

event_t evtBuf[BUFSIZE] = {{0}};
int evtBufNext = 0;
event_t mtcBuf[BUFSIZE] = {{0}};
int mtcBufNext = 0;
static config_t *test_config;

// These signatures satisfy --wrap=cmdSendEvent in the Makefile
#ifdef __linux__
int __real_cmdSendEvent(ctl_t*, event_t*, uint64_t, proc_id_t*);
int __wrap_cmdSendEvent(ctl_t* ctl, event_t* event, uint64_t uid, proc_id_t* proc)
#endif // __linux__
#ifdef __APPLE__
int cmdSendEvent(ctl_t* ctl, event_t* event, uint64_t uit, proc_id_t* proc)
#endif // __APPLE__
{
    // Store event for later inspection
    memcpy(&evtBuf[evtBufNext++], event, sizeof(*event));
    if (evtBufNext >= BUFSIZE) fail();

    return 0; //__real_cmdSendEvent(ctl, event, uid, proc);
}

// These signatures satisfy --wrap=cmdSendMetric in the Makefile
#ifdef __linux__
int __real_cmdSendMetric(mtc_t*, event_t*);
int __wrap_cmdSendMetric(mtc_t* mtc, event_t* metric)
#endif // __linux__
#ifdef __APPLE__
int cmdSendMetric(mtc_t* mtc, event_t* metric)
#endif // __APPLE__
{
    // Store metric for later inspection
    memcpy(&mtcBuf[mtcBufNext++], metric, sizeof(*metric));
    if (mtcBufNext >= BUFSIZE) fail();

    return 0; //__real_cmdSendMetric(mtc, metric);
}


int
eventCalls(const char* str)
{
    doEvent();
    int i, returnVal = 0;
    for (i=0; i<evtBufNext; i++) {
        if (!str || !strcmp(evtBuf[i].name, str)) returnVal++;
    }
    return returnVal;
}

int
metricCalls(const char* str)
{
    doEvent();
    int i, returnVal = 0;
    for (i=0; i<mtcBufNext; i++) {
        if (!str || !strcmp(mtcBuf[i].name, str)) returnVal++;
    }
    return returnVal;
}

// we no longer clear counters on rd/wr, only on close
// the last entry is the total
int
eventRdWrValues(const char *str)
{
    doEvent();
    int i, returnVal = 0;
    for (i=0; i < evtBufNext; i++) {
        if (!str || !strcmp(evtBuf[i].name, str)) {
            if (evtBuf[i].value.integer > 0) {
                returnVal = evtBuf[i].value.integer;
            }
        }
    }
    return returnVal;
}

int
eventValues(const char* str)
{
    doEvent();
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
    doEvent();
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
    doEvent();
    memset(&evtBuf, 0, sizeof(evtBuf));
    evtBufNext = 0;
    memset(&mtcBuf, 0, sizeof(mtcBuf));
    mtcBufNext = 0;
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
    initFn();

    // init objects that count has
    init_g_proc();
    test_config = cfgCreateDefault();

    g_log = logCreate();
    g_mtc = mtcCreate();
    g_ctl = ctlCreate(test_config);

    initState();

    // Turn on metric events
    evt_fmt_t * evt_fmt = evtFormatCreate();
    evtFormatSourceEnabledSet(evt_fmt, CFG_SRC_METRIC, TRUE);
    ctlEvtSet(g_ctl, evt_fmt);

    // Call the general groupSetup() too.
    return groupSetup(state);
}

static int
countTestTeardown(void** state)
{
    logDestroy(&g_log);
    mtcDestroy(&g_mtc);
    ctlDestroy(&g_ctl);
    cfgDestroy(&test_config);

    // Call the general groupTeardown() too.
    return groupTeardown(state);
}

static void
nothingCrashesBeforeAnyInit(void** state)
{
    resetState();

    // report.h
    setReportingInterval(10);
    sendProcessStartMetric();
    doErrorMetric(NET_ERR_CONN, PERIODIC, "A", "B", NULL);
    doProcMetric(PROC_CPU, 2345);
    doStatMetric("statFunc", "/the/path/to/something", NULL);
    doTotal(TOT_READ);
    doTotalDuration(TOT_DNS_DURATION);
    doEvent();

    // state.h
    setVerbosity(9);
    addSock(3, SOCK_SEQPACKET, 0);
    doBlockConnection(4, NULL);
    doSetConnection(7, NULL, 3, LOCAL);
    doSetAddrs(8);
    doAddNewSock(9);
    getDNSName(10, NULL, 0);
    doURL(11, NULL, 0, NETRX);
    doRecv(12, 4312, NULL, 0, BUF);
    doSend(13, 6682, NULL, 0, BUF);
    doAccept(14, NULL, 0, "acceptFunc");
    reportFD(15, EVENT_BASED);
    reportAllFds(PERIODIC);
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    doWrite(17, 876, 1, NULL, 0, "writeFunc", BUF, 0);
    doSeek(18, 1, "seekymcseekface");
#ifdef __linux__
    doStatPath("/pathy/path", 0, "statymcstatface");
    doStatFd(19, 0, "toomuchstat4u");
#endif // __linux__
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
    doUpdateState(OPEN_PORTS, 3, 4, "something", "/path/to/something");
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
    doRead(16, 987, 1, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Without read/write summarization, every doRead is output
    clearTestData();
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 2);
    assert_int_equal(metricValues("fs.read"), 2*13);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventRdWrValues("fs.read"), 2*13);

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
    doRead(16, 987, 1, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // With read/write summarization, no doRead is output at the time
    clearTestData();
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventRdWrValues("fs.read"), 2*13);

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
    doRead(16, 987, 1, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_READ);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 0);

    // With read/write summarization, no doRead is output at the time
    clearTestData();
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    doRead(16, 987, 1, NULL, 13, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.read"), 0);
    assert_int_equal(eventCalls("fs.read"), 2);
    assert_int_equal(eventRdWrValues("fs.read"), 2*13);

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
    doWrite(16, 987, 1, buf, 0, "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Without read/write summarization, every doWrite is output
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 2);
    assert_int_equal(metricValues("fs.write"), 2*sizeof(buf));
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventRdWrValues("fs.write"), 2*sizeof(buf));

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
    doWrite(16, 987, 1, buf, 0, "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // With read/write summarization, no doWrite is output at the time
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventRdWrValues("fs.write"), 2*sizeof(buf));

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
    doWrite(16, 987, 1, buf, 0, "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_WRITE);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 0);

    // With read/write summarization, no doWrite is output at the time
    clearTestData();
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    doWrite(16, 987, 1, buf, sizeof(buf), "writeFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.write"), 0);
    assert_int_equal(eventCalls("fs.write"), 2);
    assert_int_equal(eventRdWrValues("fs.write"), 2*sizeof(buf));

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
    doRecv(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Without rx/tx summarization, every doRecv is output
    clearTestData();
    doRecv(16, 13, NULL, 13, BUF);
    doRecv(16, 13, NULL, 13, BUF);
    assert_int_equal(metricCalls("net.rx"), 2);
    assert_int_equal(metricValues("net.rx"), 2*13);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventRdWrValues("net.rx"), 2*13);

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
    doRecv(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // With rx/tx summarization, no doRecv is output at the time
    clearTestData();
    doRecv(16, 13, NULL, 13, BUF);
    doRecv(16, 13, NULL, 13, BUF);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventRdWrValues("net.rx"), 2*13);

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
    doRecv(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_RX);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 0);

    // With rx/tx summarization, no doRecv is output at the time
    clearTestData();
    doRecv(16, 13, NULL, 13, BUF);
    doRecv(16, 13, NULL, 13, BUF);
    assert_int_equal(metricCalls("net.rx"), 0);
    assert_int_equal(eventCalls("net.rx"), 2);
    assert_int_equal(eventRdWrValues("net.rx"), 2*13);

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
    doSend(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Without rx/tx summarization, every doSend is output
    clearTestData();
    doSend(16, 13, NULL, 0, BUF);
    doSend(16, 13, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 2);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventRdWrValues("net.tx"), 2*13);

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
    doSend(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // With rx/tx summarization, no doSend is output at the time
    clearTestData();
    doSend(16, 13, NULL, 0, BUF);
    doSend(16, 13, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventRdWrValues("net.tx"), 2*13);

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
    doSend(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);

    // With rx/tx summarization, no doSend is output at the time
    clearTestData();
    doSend(16, 13, NULL, 0, BUF);
    doSend(16, 13, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventRdWrValues("net.tx"), 2*13);

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

#ifdef __linux__
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

     // doTotal should output fs.stat.
     clearTestData();
     doTotal(TOT_OPEN);
     doTotal(TOT_STAT);
     doTotal(TOT_CLOSE);
     assert_int_equal(metricCalls("fs.stat"), 1);
     assert_int_equal(metricValues("fs.stat"), 2);
     assert_int_equal(eventCalls(NULL), 0);
}
#endif // __linux__

static void
doDNSSendNoDNSSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "53", &hints, &addr_list) || !addr_list) {
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
    doSend(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_DNS);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);


    // Without DNS summarization, every net.dns is output
    clearTestData();

    // A query to look up www.google.com
    uint8_t pkt[] = {
	0xde, 0xaf, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x01, 0x03, 0x77, 0x77, 0x77,
	0x06, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x03,
	0x63, 0x6f, 0x6d, 0x00, 0x00, 0x01
    };
    getDNSName(16, pkt, sizeof(pkt));
    doSend(16, 13, NULL, 13, BUF);
    // Switch from www.google.com to www.reddit.com.
    // This is because we only report dns domain *changes*.
    memcpy(&pkt[17], "reddit", 6);
    getDNSName(16, pkt, sizeof(pkt));
    doSend(16, 13, NULL, 13, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventRdWrValues("net.tx"), 2*13);
    assert_int_equal(metricCalls("net.dns"), 2);
    assert_int_equal(metricValues("net.dns"), 2);
    // the way this is done, there is now a raw event and a DNS event
    // for each doEvent()
    assert_int_equal(eventCalls("net.dns.req"), 4);
    assert_int_equal(eventValues("net.dns.req"), 4);


    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal shouldn't output net.dns.  It's already reported
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_DNS);
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 1);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doDNSSendDNSSummarization(void** state)
{
    struct addrinfo* addr_list = NULL;
    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    if (getaddrinfo("localhost", "53", &hints, &addr_list) || !addr_list) {
        fail();
    }

    clearTestData();
    setVerbosity(5);
    doAccept(16, addr_list->ai_addr, &addr_list->ai_addrlen, "acceptFunc");
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // Zeros should not be reported on any interface
    doSend(16, 0, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);

    // Totals should not be reported if zero either
    doTotal(TOT_DNS);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);


    // With DNS summarization, net.dns is not output
    clearTestData();

    // A query to look up www.google.com
    uint8_t pkt[] = {
	0xde, 0xaf, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x01, 0x03, 0x77, 0x77, 0x77,
	0x06, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x03,
	0x63, 0x6f, 0x6d, 0x00, 0x00, 0x01
    };
    getDNSName(16, pkt, sizeof(pkt));
    doSend(16, 13, NULL, 0, BUF);
    // Switch from www.google.com to www.reddit.com.
    // This is because we only report dns domain *changes*.
    memcpy(&pkt[17], "reddit", 6);
    getDNSName(16, pkt, sizeof(pkt));
    doSend(16, 13, NULL, 0, BUF);
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 2);
    assert_int_equal(eventRdWrValues("net.tx"), 2*13);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(metricValues("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 4);
    assert_int_equal(eventValues("net.dns.req"), 4);


    // Without open/close summarization, every doClose is output
    clearTestData();
    doClose(16, "closeFunc");
    assert_int_equal(metricCalls("net.tx"), 0);
    assert_int_equal(eventCalls("net.tx"), 0);
    assert_int_equal(metricCalls("net.dns"), 0);
    assert_int_equal(eventCalls("net.dns.req"), 0);
    assert_int_equal(metricCalls("net.tcp"), 0);
    assert_int_equal(metricCalls("net.port"), 0);
    assert_int_equal(eventCalls("net.tcp"), 1);
    assert_int_equal(eventCalls("net.port"), 1);

    // doTotal should output net.dns and net.tx.  (not reported above)
    clearTestData();
    doTotal(TOT_PORTS);
    doTotal(TOT_TCP_CONN);
    doTotal(TOT_DNS);
    doTotal(TOT_TX);
    assert_int_equal(metricCalls("net.tx"), 1);
    assert_int_equal(metricValues("net.tx"), 2*13);
    assert_int_equal(metricCalls("net.dns"), 1);
    assert_int_equal(metricValues("net.dns"), 2);
    assert_int_equal(eventCalls(NULL), 0);

    if(addr_list) freeaddrinfo(addr_list);
}

static void
doFSConnectionErrorNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);
    doOpen(16, "/the/file/path", FD, "openFunc");

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "open/close" fs.error and report it immediately
    doDup(16, -1, "dupFunc", FALSE);
    doDup(16, -1, "dupFunc", FALSE);
    assert_int_equal(metricCalls("fs.error"), 2);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doFSConnectionErrorSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);
    doOpen(16, "/the/file/path", FD, "openFunc");

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "open/close" fs.error but not report it
    doDup(16, -1, "dupFunc", FALSE);
    doDup(16, -1, "dupFunc", FALSE);
    assert_int_equal(metricCalls("fs.error"), 0);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Ok, this should report the earlier fs.errors
    clearTestData();
    doErrorMetric(FS_ERR_OPEN_CLOSE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("fs.error"), 1);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doNetConnectionErrorNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);
    addSock(16, SOCK_STREAM, 0);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "connection" net.error and report it immediately
    doDup(16, -1, "dupFunc", TRUE);
    doDup(16, -1, "dupFunc", TRUE);
    assert_int_equal(metricCalls("net.error"), 2);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doNetConnectionErrorSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);
    addSock(16, SOCK_STREAM, 0);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "connection" net.error but not report it
    doDup(16, -1, "dupFunc", TRUE);
    doDup(16, -1, "dupFunc", TRUE);
    assert_int_equal(metricCalls("net.error"), 0);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Ok, this should report the earlier net.errors
    clearTestData();
    doErrorMetric(NET_ERR_CONN, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("net.error"), 1);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doFSReadWriteErrorNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);
    doOpen(16, "/the/file/path", FD, "openFunc");

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "read/write" fs.error and report it immediately
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.error"), 2);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doFSReadWriteErrorSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);
    doOpen(16, "/the/file/path", FD, "openFunc");

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "read/write" fs.error but not report it
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("fs.error"), 0);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Ok, this should report the earlier fs.errors
    clearTestData();
    doErrorMetric(FS_ERR_READ_WRITE, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("fs.error"), 1);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doNetRxTxErrorNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);
    addSock(16, SOCK_STREAM, 0);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "rx/tx" net.error and report it immediately
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("net.error"), 2);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

static void
doNetRxTxErrorSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);
    addSock(16, SOCK_STREAM, 0);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "rx/tx" net.error but not report it
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    doRead(16, 987, 0, NULL, 0, "readFunc", BUF, 0);
    assert_int_equal(metricCalls("net.error"), 0);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Ok, this should report the earlier net.errors
    clearTestData();
    doErrorMetric(NET_ERR_RX_TX, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("net.error"), 1);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);

    doClose(16, "closeFunc");
}

#ifdef __linux__
static void
doStatErrNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "stat" fs.error and report it immediately
    doStatPath("/pathy/path", -1, "statymcstatface");
    doStatPath("/pathy/path", -1, "statymcstatface");
    assert_int_equal(metricCalls("fs.error"), 2);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doStatErrSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "stat" fs.error but not report it
    doStatPath("/pathy/path", -1, "statymcstatface");
    doStatPath("/pathy/path", -1, "statymcstatface");
    assert_int_equal(metricCalls("fs.error"), 0);
    assert_int_equal(eventCalls("fs.error"), 2);
    assert_int_equal(eventValues("fs.error"), 2);

    // Ok, this should report the earlier fs.errors
    clearTestData();
    doErrorMetric(FS_ERR_STAT, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("fs.error"), 1);
    assert_int_equal(metricValues("fs.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);
}
#endif // __linux__

static void
doDNSErrNoSummarization(void** state)
{
    clearTestData();
    setVerbosity(5);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "dns" net.error and report it immediately
    doUpdateState(NET_ERR_DNS, -1, 0, "gethostbyname_r", "blah");
    doUpdateState(NET_ERR_DNS, -1, 0, "gethostbyname_r", "blah");
    assert_int_equal(metricCalls("net.error"), 2);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);
}

static void
doDNSErrSummarization(void** state)
{
    clearTestData();
    setVerbosity(4);

    // Zeros should not be reported on any interface
    clearTestData();
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls(NULL), 0);
    assert_int_equal(eventCalls(NULL), 0);

    // Should create "dns" net.error but not report it
    doUpdateState(NET_ERR_DNS, -1, 0, "gethostbyname_r", "blah");
    doUpdateState(NET_ERR_DNS, -1, 0, "gethostbyname_r", "blah");
    assert_int_equal(metricCalls("net.error"), 0);
    assert_int_equal(eventCalls("net.error"), 2);
    assert_int_equal(eventValues("net.error"), 2);

    // Nothing to see here, because it was reported earlier
    clearTestData();
    doErrorMetric(NET_ERR_DNS, PERIODIC, "summary", "summary", NULL);
    assert_int_equal(metricCalls("net.error"), 1);
    assert_int_equal(metricValues("net.error"), 2);
    assert_int_equal(eventCalls(NULL), 0);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    // Make sure that the functions can be hit before anything
    // is initialized (before constructor).
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
#ifdef __linux__
        cmocka_unit_test(doStatPathNoSummarization),
        cmocka_unit_test(doStatPathSummarization),
        cmocka_unit_test(doStatFdNoSummarization),
        cmocka_unit_test(doStatFdSummarization),
#endif // __linux__
        cmocka_unit_test(doDNSSendNoDNSSummarization),
        cmocka_unit_test(doDNSSendDNSSummarization),
        cmocka_unit_test(doFSConnectionErrorNoSummarization),
        cmocka_unit_test(doFSConnectionErrorSummarization),
        cmocka_unit_test(doNetConnectionErrorNoSummarization),
        cmocka_unit_test(doNetConnectionErrorSummarization),
        cmocka_unit_test(doFSReadWriteErrorNoSummarization),
        cmocka_unit_test(doFSReadWriteErrorSummarization),
        cmocka_unit_test(doNetRxTxErrorNoSummarization),
        cmocka_unit_test(doNetRxTxErrorSummarization),
#ifdef __linux__
        cmocka_unit_test(doStatErrNoSummarization),
        cmocka_unit_test(doStatErrSummarization),
#endif // __linux__
        cmocka_unit_test(doDNSErrNoSummarization),
        cmocka_unit_test(doDNSErrSummarization),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    int test_errors = cmocka_run_group_tests(tests, countTestSetup, countTestTeardown);
    return pre_init_errors || test_errors;
}

