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
//    return ctlPostEvent(ctl, event);a
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
isHttpWithSplitBuffer(void** state)
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
        assert_false(isHttp(3, &net, (void**)&buffers[i], &buflen, NETRX, BUF));
    }
}

static void
isHttpWithSingleBuffer(void** state)
{
    char *buffer =
        "GET / HTTP/1.0\r\n"
        "Host: www.google.com\r\n"
        "Connection: close\r\n"
        "\r\n";

    net_info net = {0};
    net.type = SOCK_STREAM;

    size_t buflen = strlen(buffer);
    assert_true(isHttp(3, &net, (void**)&buffer, &buflen, NETRX, BUF));
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(isHttpWithSplitBuffer),
        cmocka_unit_test(isHttpWithSingleBuffer),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, needleTestSetup, groupTeardown);
}

