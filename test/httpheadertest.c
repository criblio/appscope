#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "httpstate.h"
#include "test.h"

rtconfig g_cfg = {0};
 
#ifdef __LINUX__
int __real_cmdSendHttp(ctl_t *, event_t *, uint64_t, proc_id_t *);
int __wrap_cmdSendHttp(ctl_t *ctl, event_t *event, uint64_t time, proc_id_t *proc)
#endif // __LINUX__
#ifdef __MACOS__
int cmdSendHttp(ctl_t *ctl, event_t *event, uint64_t time, proc_id_t *proc)
#endif // __MACOS__
{
    return 0;
}

#ifdef __LINUX__
int __real_cmdPostEvent(ctl_t *, event_t *);
int __wrap_cmdPostEvent(ctl_t *ctl, event_t *event)
#endif // __LINUX__
#ifdef __MACOS__
int cmdPostEvent(ctl_t *ctl, event_t *event)
#endif // __MACOS__
{
    return 0;
}

static int
needleTestSetup(void** state)
{
    initTime();
    initFn();
    initHttpState();

    // Call the general groupSetup() too.
    return groupSetup(state);
}

static void
freeMsg()
{

}

static void
headerBasic(void **state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    size_t buflen = strlen(buffer);

    net_info net = {0};
    net.fd = 0;
    net.type = SOCK_STREAM;

    assert_true(doHttp(0x12345, 0, &net, buffer, buflen, TLSRX, BUF));
    //assert_non_null(g_msg);
    //struct http_post_t *post = (struct http_post_t*) g_msg->data;
    //assert_non_null(post);
    //char *header = post->hdr;
    //assert_non_null(header);
    //assert_string_equal(post->hdr, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
    //freeMsg(&g_msg);
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(headerBasic),
    };
    return cmocka_run_group_tests(tests, needleTestSetup, groupTeardown);
}

