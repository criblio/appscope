#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/statvfs.h>
#ifdef __LINUX__
#include <sys/vfs.h>
#include <sys/epoll.h>
#endif // _LINUX_

#include "ctl.h"
#include "httpstate.h"
#include "plattime.h"
#include "wrap.h"
#include "test.h"


interposed_funcs g_fn = {0};
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


static int
needleTestSetup(void** state)
{
    initTime();
    init_g_fn();
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

    net_info net = {0};
    net.type = SOCK_STREAM;

    size_t buflen = strlen(buffer);
    assert_true(doHttp(13, 3, &net, buffer, buflen, NETRX, BUF));
    assert_non_null(g_msg);
    struct http_post_t *post = (struct http_post_t*) g_msg->data;
    assert_non_null(post);
    char *header = post->hdr;
    assert_non_null(header);
    assert_string_equal(header, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close");
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
    assert_string_equal(header, "GET / HTTP/1.0\r\nHost: www.google.com\r\nConnection: close");
    freeMsg(&g_msg);
}

static void
doHttpWithSplitBuffer(void** state)
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
        assert_false(doHttp(13, 3, &net, (void*)buffers[i], buflen, NETRX, BUF));
    }
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(doHttpWithSingleBufferWithNet),
        cmocka_unit_test(doHttpWithSingleBufferWithoutNet),
        cmocka_unit_test(doHttpWithSplitBuffer),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, needleTestSetup, groupTeardown);
}

