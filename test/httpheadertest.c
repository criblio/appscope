#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>

#include "dbg.h"
#include "plattime.h"
#include "fn.h"
#include "ctl.h"
#include "evtformat.h"
#include "httpstate.h"
#include "test.h"

#define UNIX_SOCK_PATH "/tmp/headertestsock"

extern void doProtocolMetric(protocol_info *);
rtconfig g_cfg = {0};
char *header_event;
 
static int
needleTestSetup(void** state)
{
    initTime();
    initFn();
    initState();
    initReporting();
    initHttpState();

    strcpy(g_proc.hostname, "thisHostName");
    strcpy(g_proc.procname, "thisProcName");
    g_proc.pid = 77;

    // Call the general groupSetup() too.
    return groupSetup(state);
}

#ifdef __LINUX__
int __real_cmdSendHttp(ctl_t *, event_t *, uint64_t, proc_id_t *);
int __wrap_cmdSendHttp(ctl_t *ctl, event_t *event, uint64_t id, proc_id_t *proc)
#endif // __LINUX__
#ifdef __MACOS__
int cmdSendHttp(ctl_t *ctl, event_t *event, uint64_t id, proc_id_t *proc)
#endif // __MACOS__
{
    //printf("%s: %s\n", __FUNCTION__, event->name);
    if (!event || !proc) return -1;

    if (strncmp(event->name, "http-metrics", strlen("http-metrics")) == 0) return 0;

    cJSON *json  = fmtMetricJson(event, NULL, CFG_SRC_HTTP);
    header_event = cJSON_PrintUnformatted(json);

    //printf("HTTP Header event: %s\n", header_event);

    return 0;
}

#ifdef __LINUX__
int __real_cmdPostEvent(ctl_t *, char *);
int __wrap_cmdPostEvent(ctl_t *ctl, char *event)
#endif // __LINUX__
#ifdef __MACOS__
int cmdPostEvent(ctl_t *ctl, char *event)
#endif // __MACOS__
{
    //printf("%s: data at: %p\n", __FUNCTION__, event);
    doProtocolMetric((protocol_info *)event);
    return 0;
}

static net_info *
getUnix(int fd)
{
    addSock(fd, SOCK_STREAM, AF_UNIX);

    struct sockaddr_un sa;
    bzero((char *)&sa, sizeof(sa));
    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, UNIX_SOCK_PATH, sizeof(sa.sun_path)-1);

    doSetConnection(fd, (struct sockaddr *)&sa, sizeof(struct sockaddr_in), LOCAL);

    return getNetEntry(fd);
}

static net_info *
getNet(int fd)
{
    addSock(fd, SOCK_STREAM, AF_INET);

    struct sockaddr_in sa;
    inet_pton(AF_INET, "192.1.2.3", &sa.sin_addr);
    sa.sin_family = AF_INET;
    sa.sin_port = 9999;
    doSetConnection(fd, (struct sockaddr *)&sa, sizeof(struct sockaddr_in), LOCAL);

    inet_pton(AF_INET, "192.1.2.99", &sa.sin_addr);
    sa.sin_family = AF_INET;
    sa.sin_port = 7777;
    doSetConnection(fd, (struct sockaddr *)&sa, sizeof(struct sockaddr_in), REMOTE);

    return getNetEntry(fd);
}

static void
headerBasicRequest(void **state)
{
    char *request = "GET /hello HTTP/1.1\r\nHost: localhost:4430\r\nUser-Agent: curl/7.68.0\r\nAccept: */*\r\n\r\n";
    char *result = "{\"http.method\":\"GET\",\"http.target\":\"/hello\",\"http.flavor\":\"1.1\",\"http.scheme\":\"HTTPS\",\"http.host\":\"localhost:4430\",\"http.user_agent\":\"curl/7.68.0\",\"net.host.name\":\"thisHostName\"}";
    size_t buflen = strlen(request);

    net_info net = {0};
    net.fd = 0;
    net.type = SOCK_STREAM;

    assert_true(doHttp(0x12345, 0, &net, request, buflen, TLSRX, BUF));
    //printf("%s: %s\n", __FUNCTION__, header_event);
    assert_string_equal(header_event, result);
}

static void
headerBasicResponse(void **state)
{
    char *response = "HTTP/1.1 200 OK blah blah now is the time for a response\r\nContent-Type: text/plain\r\nDate: Tue, 29 Sep 2020 19:56:15 GMT\r\n\r\n";
    char *result =
"{\"http.flavor\":\"1.1\","
"\"http.status_code\":200,"
"\"http.status_text\":\"OK blah blah now is the time for a response\","
"\"http.server.duration\":0,"
"\"net.host.name\":\"thisHostName\"}";

    net_info net = {0};
    net.fd = 3;
    net.type = SOCK_STREAM;

    assert_true(doHttp(0x12345, 3, &net, response, strlen(response), TLSRX, BUF));
    //printf("%s: %s\n\n\n", __FUNCTION__, header_event);
    assert_string_equal(header_event, result);
}

static void
headerRequestIP(void **state)
{
    char *request = "GET /hello HTTP/1.1\r\nHost: localhost:4430\r\nUser-Agent: curl/7.68.0\r\nAccept: */*\r\nContent-Length: 12345\r\nX-Forwarded-For: 192.7.7.7\r\n\r\n";
    char *result =
"{\"http.method\":\"GET\","
"\"http.target\":\"/hello\","
"\"http.flavor\":\"1.1\","
"\"http.scheme\":\"HTTPS\","
"\"http.host\":\"localhost:4430\","
"\"http.user_agent\":\"curl/7.68.0\","
"\"http.client_ip\":\"192.7.7.7\","
"\"net.host.name\":\"thisHostName\","
"\"net.transport\":\"IP.TCP\","
"\"net.peer.ip\":\"192.1.2.99\","
"\"net.peer.port\":\"24862\","
"\"net.host.ip\":\"192.1.2.3\","
"\"net.host.port\":\"3879\","
"\"http.request_content_length\":12345}";

    net_info *net = getNet(3);
    assert_non_null(net);
    assert_true(doHttp(0x12345, 3, net, request, strlen(request), TLSRX, BUF));
    //printf("%s: %s\n\n\n", __FUNCTION__, header_event);
    assert_string_equal(header_event, result);
}

static void
headerResponseIP(void **state)
{
    char *response = "HTTP/1.1 777 Not OK\r\nContent-Type: text/plain\r\nDate: Tue, 29 Sep 2020 19:56:15 GMT\r\nContent-Length: 27\r\n\r\n";
    char *result =
"{\"http.flavor\":\"1.1\","
"\"http.status_code\":777,"
"\"http.status_text\":\"Not OK\","
"\"http.server.duration\":0,"
"\"net.host.name\":\"thisHostName\","
"\"net.transport\":\"IP.TCP\","
"\"net.peer.ip\":\"192.1.2.99\","
"\"net.peer.port\":\"24862\","
"\"net.host.ip\":\"192.1.2.3\","
"\"net.host.port\":\"3879\","
"\"http.response_content_length\":27}";

    net_info *net = getNet(3);
    assert_non_null(net);
    assert_true(doHttp(0x12345, 3, net, response, strlen(response), TLSRX, BUF));
    //printf("%s: %s\n\n\n", __FUNCTION__, header_event);
    assert_string_equal(header_event, result);
}

static void
headerRequestUnix(void **state)
{
    char *request = "GET /hello HTTP/1.1\r\nHost: localhost:4430\r\nUser-Agent: curl/7.68.0\r\nAccept: */*\r\nContent-Length: 12345\r\nX-Forwarded-For: 192.7.7.7\r\n\r\n";
    char *result =
"{\"http.method\":\"GET\","
"\"http.target\":\"/hello\","
"\"http.flavor\":\"1.1\","
"\"http.scheme\":\"HTTPS\","
"\"http.host\":\"localhost:4430\","
"\"http.user_agent\":\"curl/7.68.0\","
"\"http.client_ip\":\"192.7.7.7\","
"\"net.host.name\":\"thisHostName\","
"\"net.transport\":\"Unix.TCP\","
"\"http.request_content_length\":12345}";

    net_info *net = getUnix(3);
    assert_non_null(net);
    assert_true(doHttp(0x12345, 3, net, request, strlen(request), TLSRX, BUF));
    //printf("%s: %s\n\n\n", __FUNCTION__, header_event);
    assert_string_equal(header_event, result);
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(headerBasicRequest),
        cmocka_unit_test(headerBasicResponse),
        cmocka_unit_test(headerRequestIP),
        cmocka_unit_test(headerResponseIP),
        cmocka_unit_test(headerRequestUnix),
    };
    return cmocka_run_group_tests(tests, needleTestSetup, groupTeardown);
}

