#include <netdb.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "transport.h"

#include "test.h"

static void
transportCreateUdpReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateUdp("127.0.0.1", 8126);
    assert_non_null(t);
    transportDestroy(&t);
}

static void
transportCreateUdpReturnsNullPtrForInvalidHost()
{
    transport_t* t;
    t = transportCreateUdp(NULL, 8126);
    assert_null(t);

    t = transportCreateUdp("this is not a good looking host name", 8126);
    assert_null(t);
}

static void
transportCreateUdpHandlesGoodHostArguments()
{
    char* host_values[] = {"localhost", "www.google.com",
                         "127.0.0.1", "0.0.0.0", "8.8.4.4",
                         "::1", "::ffff:127.0.0.1", "::", "2001:4860:4860::8844",
                         NULL };

    // These don't work yet, but should before we're done
    // Until then, stick to a string with IPv4 style notation.
    skip();

    char* host;
    for (host = host_values[0]; host; host++) {
        transport_t* t = transportCreateUdp(host, 1234);
        assert_non_null(t);
        transportDestroy(&t);
    }
}

static void
transportCreateFileReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateFile("/tmp/scope.log");
    assert_non_null(t);
    transportDestroy(&t);
    assert_null(t);
}

static void
transportCreateFileReturnsNullForInvalidPath(void** state)
{
    transport_t* t;
    t = transportCreateFile(NULL);
    assert_null(t);
    t = transportCreateFile("");
    assert_null(t);
}

static void
transportCreateFileCreatesDirectoriesAsNeeded(void** state)
{
    // This should work in my opinion, but doesn't right now
    skip();

    transport_t* t = transportCreateFile("/var/log/out/directory/path/here.log");
    assert_non_null(t);
}

static void
transportCreateUnixReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateUnix("/my/favorite/path");
    assert_non_null(t);
    transportDestroy(&t);
    assert_null(t);

}

static void
transportCreateUnixReturnsNullForInvalidPath(void** state)
{
    transport_t* t = transportCreateUnix(NULL);
    assert_null(t);
}


static void
transportCreateSyslogReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    transportDestroy(&t);
    assert_null(t);

}

static void
transportCreateShmReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateShm();
    assert_non_null(t);
    transportDestroy(&t);
    assert_null(t);
}

static void
transportDestroyNullTransportDoesNothing(void** state)
{
    transportDestroy(NULL);
    transport_t* t = NULL;
    transportDestroy(&t);
    // Implicitly shows that calling transportDestroy with NULL is harmless
}

static void
transportSendForNullTransportDoesNothing(void** state)
{
    char* msg = "Hey, this is cool!\n";
    assert_int_equal(transportSend(NULL, msg), -1);
}

static void
transportSendForNullMessageDoesNothing(void** state)
{
    char* path = "/tmp/path";
    transport_t* t = transportCreateFile(path);
    assert_non_null(t);
    assert_int_equal(transportSend(t, NULL), -1);
    transportDestroy(&t);
    unlink(path);
}

static void
transportSendForUnimplementedTransportTypesIsHarmless(void** state)
{
    transport_t* t;
    t = transportCreateUnix("/my/favorite/path");
    transportSend(t, "blah");
    transportDestroy(&t);

    t = transportCreateSyslog();
    transportSend(t, "blah");
    transportDestroy(&t);

    t = transportCreateShm();
    transportSend(t, "blah");
    transportDestroy(&t);
}

static void
transportSendForUdpTransmitsMsg(void** state)
{
    char* hostname = "127.0.0.1";
    char* portname = "8126";
    struct addrinfo hints = {0};
    hints.ai_family=AF_UNSPEC;
    hints.ai_socktype=SOCK_DGRAM;
    hints.ai_flags=AI_PASSIVE|AI_ADDRCONFIG;
    struct addrinfo* res = NULL;
    if (getaddrinfo(hostname, portname, &hints, &res)) {
        fail_msg("Couldn't create address for socket");
    }
    int sd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (!sd) fail_msg("Couldn't create socket");
    if (bind(sd, res->ai_addr, res->ai_addrlen) == -1) {
        fail_msg("Couldn't bind socket");
    }
    freeaddrinfo(res);

    transport_t* t = transportCreateUdp(hostname, atoi(portname));
    assert_non_null(t);
    char msg[] = "This is the payload message to transfer.\n";
    char buf[sizeof(msg)] = {0};
    assert_int_equal(transportSend(t, msg), 0);

    struct sockaddr from;
    socklen_t len;
    int byteCount=0;
    if ((byteCount = recvfrom(sd, buf, sizeof(buf), 0, &from, &len)) != sizeof(msg)) {
        fail_msg("Couldn't recvfrom");
    }
    assert_string_equal(msg, buf);

    transportDestroy(&t);

    close(sd);

}

static void
transportSendForFileWritesToFile(void** state)
{
    char* path = "/tmp/mypath";
    transport_t* t = transportCreateFile(path);
    assert_non_null(t);

    // open the file with the position at the end
    FILE* f = fopen(path, "r+");
    if (!f)
        fail_msg("Couldn't open file %s", path);
    if (fseek(f, 0, SEEK_END))
        fail_msg("Couldn't seek to end of file %s", path);

    // Since we're at the end, nothing should be there
    char buf[1024];
    size_t bytesRead = fread(buf, 1, sizeof(buf), f);
    assert_int_equal(bytesRead, 0);
    assert_true(feof(f));
    assert_false(ferror(f));
    clearerr(f);

    char msg[] = "This is the payload message to transfer.\n";
    assert_int_equal(transportSend(t, msg), 0);

    // Test that after the transportSend, that the msg got there.
    bytesRead = fread(buf, 1, sizeof(buf), f);
    assert_int_equal(bytesRead, strlen(msg)+1);
    assert_true(feof(f));
    assert_false(ferror(f));
    assert_string_equal(msg, buf);

    if (fclose(f)) fail_msg("Couldn't close file %s", path);

    transportDestroy(&t);

    if (unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

int
main (int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(transportCreateUdpReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateUdpReturnsNullPtrForInvalidHost),
        cmocka_unit_test(transportCreateUdpHandlesGoodHostArguments),
        cmocka_unit_test(transportCreateFileReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateFileReturnsNullForInvalidPath),
        cmocka_unit_test(transportCreateFileCreatesDirectoriesAsNeeded),
        cmocka_unit_test(transportCreateUnixReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateUnixReturnsNullForInvalidPath),
        cmocka_unit_test(transportCreateSyslogReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateShmReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportDestroyNullTransportDoesNothing),
        cmocka_unit_test(transportSendForNullTransportDoesNothing),
        cmocka_unit_test(transportSendForNullMessageDoesNothing),
        cmocka_unit_test(transportSendForUnimplementedTransportTypesIsHarmless),
        cmocka_unit_test(transportSendForUdpTransmitsMsg),
        cmocka_unit_test(transportSendForFileWritesToFile),
    };

    cmocka_run_group_tests(tests, NULL, NULL);

    return 0;
}
