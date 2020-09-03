#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "ctl.h"
#include "dbg.h"
#include "fn.h"
#include "httpstate.h"
#include "plattime.h"
#include "test.h"


ctl_t *g_ctl = NULL;
struct protocol_info_t* g_msg = NULL;


void
freeMsg(struct protocol_info_t** msg_ptr)
{
    if (!msg_ptr || !*msg_ptr) return;

    struct protocol_info_t *msg = *msg_ptr;
    struct http_post_t *post = msg ? (struct http_post_t*) msg->data : NULL;
    char *header = post ? post->hdr : NULL;

    if (header) free(header);
    if (post) free(post);
    if (msg) free(msg);
    *msg_ptr = NULL;
}

// This has almost nothing to do with this test.
// I'm defining it here to avoid more dependencies.
int
get_port_net(net_info *net, int type, control_type_t which) {
    return 0;
}

// This on the other hand is an important part of this test.
int
cmdPostEvent(ctl_t *ctl, char *event)
{
    if (g_msg) freeMsg(&g_msg); // Don't leak
    g_msg = (struct protocol_info_t*)event;
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
doHttpWithSingleBufferWithNet(void** state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    size_t buflen = strlen(buffer);

    net_info net = {0};
    net.type = SOCK_STREAM;

    assert_true(doHttp(13, 3, &net, buffer, buflen, NETRX, BUF));
    assert_non_null(g_msg);
    struct http_post_t *post = (struct http_post_t*) g_msg->data;
    assert_non_null(post);
    char *header = post->hdr;
    assert_non_null(header);
    assert_string_equal(post->hdr, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
    freeMsg(&g_msg);
}

static void
doHttpWithSingleBufferWithoutNet(void** state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    size_t buflen = strlen(buffer);

    assert_true(doHttp(13, 3, NULL, buffer, buflen, NETRX, BUF));
    assert_non_null(g_msg);
    struct http_post_t *post = (struct http_post_t*) g_msg->data;
    assert_non_null(post);
    char *header = post->hdr;
    assert_non_null(header);
    assert_string_equal(header, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
    freeMsg(&g_msg);
}

static void
doHttpWithPartialHeaderBeforeClose(void** state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n";
    size_t buflen = strlen(buffer);

    net_info net = {0};
    net.type = SOCK_STREAM;

    assert_false(doHttp(13, 3, &net, buffer, buflen, NETRX, BUF));
    assert_null(g_msg);

    assert_non_null(net.http.hdr);

    // This acts like a virtual doClose()
    resetHttp(&net.http);

    assert_null(net.http.hdr);
}

static void
doHttpWithConsecutiveHeaders(void** state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n"
        "Connection: close\r\n"
        "\r\n";
    size_t buflen = strlen(buffer);

    net_info net = {0};
    net.type = SOCK_STREAM;

    {
        assert_true(doHttp(13, 3, &net, buffer, buflen, NETRX, BUF));
        assert_non_null(g_msg);
        struct http_post_t *post = (struct http_post_t*) g_msg->data;
        assert_non_null(post);
        char *header = post->hdr;
        assert_non_null(header);
        assert_string_equal(header, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
        freeMsg(&g_msg);
    }

    {
        assert_true(doHttp(13, 3, &net, buffer, buflen, NETRX, BUF));
        assert_non_null(g_msg);
        struct http_post_t *post = (struct http_post_t*) g_msg->data;
        assert_non_null(post);
        char *header = post->hdr;
        assert_non_null(header);
        assert_string_equal(header, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
        freeMsg(&g_msg);
    }
}

static void
doHttpWithSplitHeader(void** state)
{
    char *buffers[] = {
        "GET / HTTP/1.0\r\n",
        "Host: www.google.com\r\n",
        "Connection: close\r\n",
        "\r\n",
        NULL };
    net_info net = {0};
    net.type = SOCK_STREAM;
    int i;

    for (i=0; buffers[i]; i++) {
        size_t buflen = strlen(buffers[i]);
        bool returnValue = doHttp(13, 3, &net, (void*)buffers[i], buflen, NETRX, BUF);
        if (i == 3) {
            assert_true(returnValue);
            assert_non_null(g_msg);
            struct http_post_t *post = (struct http_post_t*) g_msg->data;
            assert_non_null(post);
            char *header = post->hdr;
            assert_non_null(header);
            assert_string_equal(post->hdr, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
            freeMsg(&g_msg);
        } else {
            assert_false(returnValue);
            assert_null(g_msg);
        }
    }
}

static void
doHttpWithInterleavedEncryption(void** state)
{
    char *buffers[] = {
        "GET / HTTP/1.0\r\n",
        "\x17\x03\x03",
        "Host: www.google.com\r\n",
        "\x17\x03\x03",
        "Connection: close\r\n",
        "\x17\x03\x03",
        "\r\n",
        "\x17\x03\x03",
        NULL };
    net_info net = {0};
    net.type = SOCK_STREAM;
    int i;

    for (i=0; buffers[i]; i++) {
        size_t buflen = strlen(buffers[i]);
        int encrypted = buffers[i][0] == '\x17';
        bool returnValue = doHttp(13, 3, &net, (void*)buffers[i], buflen, (encrypted) ? TLSRX : NETRX, BUF);
        if (i == 6) {
            assert_true(returnValue);
            assert_non_null(g_msg);
            struct http_post_t *post = (struct http_post_t*) g_msg->data;
            assert_non_null(post);
            char *header = post->hdr;
            assert_non_null(header);
            assert_string_equal(post->hdr, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close\r\n");
            freeMsg(&g_msg);
        } else {
            assert_false(returnValue);
            assert_null(g_msg);
        }
    }
}

static void
doHttpWhichRequiresRealloc(void** state)
{
    char *buffers[] = {
        "GET / HTTP/1.0\r\n",
        "0123456789ABCD\r\n", // <-- repeat this one a bunch, 16 bytes at a time
        "X\r\n\r\n",
        NULL };
    net_info net = {0};
    net.type = SOCK_STREAM;
    int headersize = 0;

    while (headersize < 4096) {
        char *buffer = (!headersize) ? buffers[0] : buffers[1];

        size_t buflen = strlen(buffer);
        bool returnValue = doHttp(13, 3, &net, (void*)buffer, buflen, NETRX, BUF);
        assert_false(returnValue);
        headersize += buflen;
    }

    // buffers[2] takes us 3 bytes past the original 4096 limit.
    assert_true(doHttp(13, 3, &net, (void*)buffers[2], strlen(buffers[2]), NETRX, BUF));
}

static void
doHttpWhichExceedsReallocSize(void** state)
{
    char *buffers[] = {
        "GET / HTTP/1.0\r\n",
        "0123456789ABCD\r\n", // <-- repeat this one a bunch, 16 bytes at a time
        "X\r\n\r\n",
        NULL };
    net_info net = {0};
    net.type = SOCK_STREAM;
    int headersize = 0;

    while (headersize < 4*4096) {
        char *buffer = (!headersize) ? buffers[0] : buffers[1];

        size_t buflen = strlen(buffer);
        bool returnValue = doHttp(13, 3, &net, (void*)buffer, buflen, NETRX, BUF);
        assert_false(returnValue);
        headersize += buflen;
    }

    // buffers[2] takes us 3 bytes past the max (4*4096) limit.
    assert_false(doHttp(13, 3, &net, (void*)buffers[2], strlen(buffers[2]), NETRX, BUF));

    // exceeding the limit leaves a breadcrumb in dbg.
    assert_int_equal(dbgCountMatchingLines("src/httpstate.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests

}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(doHttpWithSingleBufferWithNet),
        cmocka_unit_test(doHttpWithSingleBufferWithoutNet),
        cmocka_unit_test(doHttpWithPartialHeaderBeforeClose),
        cmocka_unit_test(doHttpWithConsecutiveHeaders),
        cmocka_unit_test(doHttpWithSplitHeader),
        cmocka_unit_test(doHttpWithInterleavedEncryption),
        cmocka_unit_test(doHttpWhichRequiresRealloc),
        cmocka_unit_test(doHttpWhichExceedsReallocSize),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, needleTestSetup, groupTeardown);
}

