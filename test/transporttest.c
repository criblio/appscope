#define _GNU_SOURCE
#include <netdb.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "fn.h"
#include "dbg.h"
#include "scopestdlib.h"
#include "transport.h"
#include "test.h"

static void
transportCreateTcpReturnsNullPtrForNullHostOrPath(void** state)
{
    transport_t* t;
    t = transportCreateTCP(NULL, "54321", FALSE, FALSE, NULL);
    assert_null(t);

    t = transportCreateTCP("127.0.0.1", NULL, FALSE, FALSE, NULL);
    assert_null(t);
}

static void
transportCreateTcpReturnsValidPtrInHappyPath(void** state)
{
    assert_false(transportNeedsConnection(NULL));
    transport_t* t = transportCreateTCP("127.0.0.1", "54321", FALSE, FALSE, NULL);
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
}

static void
transportCreateTcpReturnsValidPtrForUnresolvedHostPort(void** state)
{
    transport_t* t;
    // This is a valid test, but it hangs for as long as 30s.
    // (It depends on dns more than a unit test should.)
/*
    t = transportCreateTCP("-tota11y bogus hostname", "666");
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
*/
    t = transportCreateTCP("127.0.0.1", "mom's apple pie recipe", FALSE, FALSE, NULL);
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
}

static void
transportConnectEstablishesConnection(void** state)
{
    skip();
    // We need to verify that tcp connections can be established.
    // This is reminder to do this.
}

static void
transportCreateUdpReturnsNullPtrForNullHostOrPath(void** state)
{
    transport_t* t;
    t = transportCreateUdp(NULL, "8128");
    assert_null(t);

    t = transportCreateUdp("127.0.0.1", NULL);
    assert_null(t);
}

static void
transportCreateUdpReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateUdp("127.0.0.1", "8126");
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
    transportDestroy(&t);
}

static void
transportCreateUdpReturnsValidPtrForUnresolvedHostPort(void** state)
{
    transport_t* t;
    // This is a valid test, but it hangs for as long as 30s.
    // (It depends on dns more than a unit test should.)
/*
    t = transportCreateUdp("-tota11y bogus hostname", "666");
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
*/
    t = transportCreateUdp("127.0.0.1", "mom's apple pie recipe");
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
}

static void
transportCreateUdpHandlesGoodHostArguments(void** state)
{
    const char* host_values[] = {
           "localhost", "www.google.com",
           "127.0.0.1", "0.0.0.0", "8.8.4.4",
           "::1", "::ffff:127.0.0.1", "::", 
           // These will work only if machine supports ipv6
           //"2001:4860:4860::8844",
           //"ipv6.google.com",
    };

    int i;
    for (i=0; i<sizeof(host_values)/sizeof(host_values[0]); i++) {
        transport_t* t = transportCreateUdp(host_values[i], "1234");
        assert_non_null(t);
        transportDestroy(&t);
    }
}

static void
transportCreateFileReturnsValidPtrInHappyPath(void** state)
{
    const char* path = "/tmp/myscope.log";
    transport_t* t = transportCreateFile(path, CFG_BUFFER_LINE);
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
    transportDestroy(&t);
    assert_null(t);
    if (scope_unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

static void
transportCreateFileCreatesFileWithRWPermissionsForAll(void** state)
{
    const char* path = "/tmp/myscope.log";
    transport_t* t = transportCreateFile(path, CFG_BUFFER_LINE);
    assert_non_null(t);
    transportDestroy(&t);

    // test permissions are 0666
    struct stat buf;
    if (scope_stat(path, &buf))
        fail_msg("Couldn't test permissions for file %s", path);
    assert_true((buf.st_mode & 0777) == 0666);

    // Clean up
    if (scope_unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

static void
transportCreateFileReturnsUnconnectedForInvalidPath(void** state)
{
    transport_t* t;
    t = transportCreateFile(NULL, CFG_BUFFER_LINE);
    assert_null(t);

    assert_int_equal(dbgCountMatchingLines("src/transport.c"), 0);

    t = transportCreateFile("", CFG_BUFFER_LINE);
    assert_non_null(t);

    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);

    assert_int_equal(dbgCountMatchingLines("src/transport.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests
}

static void
transportCreateFileCreatesDirectoriesAsNeeded(void** state)
{
    // This should work in my opinion, but doesn't right now
    skip();

    transport_t* t = transportCreateFile("/var/log/out/directory/path/here.log", CFG_BUFFER_LINE);
    assert_non_null(t);
}

static void
transportCreateUnixReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateUnix("/my/favorite/path");
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
    assert_null(t);
}

static void
transportCreateUnixReturnsNullForInvalidPath(void** state)
{
    assert_int_equal(dbgCountMatchingLines("src/transport.c"), 0);

    transport_t* t = transportCreateUnix(NULL);
    assert_null(t);

    int size_limit = sizeof (((struct sockaddr_un*)NULL)->sun_path);
    char too_long_path[size_limit+1];
    int i;  // This loop just writes ascii '0' through '9' repeatedly.
    for (i=0; i < size_limit; i++) too_long_path[i]= '0' + i%10;
    too_long_path[i] = '\0';
    t = transportCreateUnix(too_long_path);
    assert_null(t);

    assert_int_equal(dbgCountMatchingLines("src/transport.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests
}

static void
transportCreateEdgeReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateEdge();
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);
    assert_null(t);
}

static void
transportCreateSyslogReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
    transportDestroy(&t);
    assert_null(t);

}

static void
transportCreateShmReturnsValidPtrInHappyPath(void** state)
{
    transport_t* t = transportCreateShm();
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
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
    const char* msg = "Hey, this is cool!\n";
    assert_int_equal(transportSend(NULL, msg, scope_strlen(msg)), -1);
}

static void
transportSendForNullMessageDoesNothing(void** state)
{
    const char* path = "/tmp/path";
    transport_t* t = transportCreateFile(path, CFG_BUFFER_LINE);
    assert_non_null(t);
    assert_int_equal(transportSend(t, NULL, 0), -1);
    transportDestroy(&t);
    if (scope_unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

static void
transportSendForUnimplementedTransportTypesIsHarmless(void** state)
{
    transport_t* t;

    t = transportCreateSyslog();
    transportSend(t, "blah", scope_strlen("blah"));
    transportDestroy(&t);

    t = transportCreateShm();
    transportSend(t, "blah", scope_strlen("blah"));
    transportDestroy(&t);
}

static void
transportSendForUdpTransmitsMsg(void** state)
{
    const char* hostname = "127.0.0.1";
    const char* portname = "8126";
    struct addrinfo hints = {0};
    hints.ai_family=AF_UNSPEC;
    hints.ai_socktype=SOCK_DGRAM;
    hints.ai_flags=AI_PASSIVE|AI_ADDRCONFIG;
    struct addrinfo* res = NULL;
    if (getaddrinfo(hostname, portname, &hints, &res)) {
        fail_msg("Couldn't create address for socket");
    }
    int sd = scope_socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sd == -1) {
        fail_msg("Couldn't create socket");
    }
    if (scope_bind(sd, (const struct sockaddr *)res->ai_addr, res->ai_addrlen) == -1) {
        fail_msg("Couldn't bind socket");
    }
    freeaddrinfo(res);

    transport_t* t = transportCreateUdp(hostname, portname);
    assert_non_null(t);
    const char msg[] = "This is the payload message to transfer.\n";
    char buf[sizeof(msg)] = {0};  // Has room for a null at the end
    assert_int_equal(transportSend(t, msg, scope_strlen(msg)), 0);

    struct sockaddr_storage from = {0};
    socklen_t len = sizeof(from);
    int byteCount=0;
    if ((byteCount = scope_recvfrom(sd, buf, sizeof(buf), 0, (struct sockaddr*)&from, &len)) != scope_strlen(msg)) {
        fail_msg("Couldn't recvfrom");
    }
    assert_string_equal(msg, buf);

    transportDestroy(&t);

    scope_close(sd);
}

static void
transportSendForAbstractUnixTransmitsMsg(void** state)
{
    const char* path = "@mytestsockname";

    // Create a unix address for a test socket
    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    scope_memset(addr.sun_path, 0, sizeof(addr.sun_path));
    scope_strncpy(addr.sun_path, path, scope_strlen(path));
    addr.sun_path[0] = 0;
    int addr_len = sizeof(sa_family_t) + scope_strlen(path);

    // Create a test socket that our transport can send to
    int sd = scope_socket(AF_UNIX, SOCK_STREAM, 0);
    if (sd == -1) {
        fail_msg("Couldn't create socket");
    }
    if (scope_bind(sd, (const struct sockaddr *)&addr, addr_len) == -1) {
        fail_msg("Couldn't bind socket");
    }
    if (scope_listen(sd, 10) == -1) {
        fail_msg("Couldn't listen on socket");
    }

    // Verify that the transport can be created, connected, and used for a tx
    transport_t* t = transportCreateUnix(path);
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
    const char msg[] = "This is the payload message to transfer.\n";
    char buf[sizeof(msg)] = {0};  // Has room for a null at the end
    assert_int_equal(transportSend(t, msg, scope_strlen(msg)), 0);

    // Verify that our test sockets can rx the stuff sent with the transport
    struct sockaddr_storage from = {0};
    socklen_t from_len = sizeof(from);
    int rx_sock = scope_accept(sd, (struct sockaddr *)&from, &from_len);
    assert_true(rx_sock != -1);
    int byteCount = scope_recv(rx_sock, buf, sizeof(buf), 0);
    if (byteCount != scope_strlen(msg)) {
        fail_msg("Couldn't recv properly");
    }
    assert_string_equal(msg, buf);

    transportDestroy(&t);

    scope_close(rx_sock);
    scope_close(sd);
}

static void
transportSendForFilepathUnixTransmitsMsg(void** state)
{
    const char* path = "/tmp/mysocket_file";

    // Create a unix address for a test socket
    struct sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    scope_memset(addr.sun_path, 0, sizeof(addr.sun_path));
    scope_strncpy(addr.sun_path, path, scope_strlen(path));
    int addr_len = sizeof(sa_family_t) + scope_strlen(path) + 1;

    // Create a test socket that our transport can send to
    int sd = scope_socket(AF_UNIX, SOCK_STREAM, 0);
    if (sd == -1) {
        fail_msg("Couldn't create socket");
    }
    if (scope_bind(sd, (const struct sockaddr *)&addr, addr_len) == -1) {
        fail_msg("Couldn't bind socket");
    }
    if (scope_listen(sd, 10) == -1) {
        fail_msg("Couldn't listen on socket");
    }

    // Verify that the transport can be created, connected, and used for a tx
    transport_t* t = transportCreateUnix(path);
    assert_non_null(t);
    assert_false(transportNeedsConnection(t));
    const char msg[] = "This is the payload message to transfer.\n";
    char buf[sizeof(msg)] = {0};  // Has room for a null at the end
    assert_int_equal(transportSend(t, msg, scope_strlen(msg)), 0);

    // Verify that our test sockets can rx the stuff sent with the transport
    struct sockaddr_storage from = {0};
    socklen_t from_len = sizeof(from);
    int rx_sock = scope_accept(sd, (struct sockaddr *)&from, &from_len);
    assert_true(rx_sock != -1);
    int byteCount = scope_recv(rx_sock, buf, sizeof(buf), 0);
    if (byteCount != scope_strlen(msg)) {
        fail_msg("Couldn't recv properly");
    }
    assert_string_equal(msg, buf);

    transportDestroy(&t);

    scope_close(rx_sock);
    scope_close(sd);
    scope_unlink(path);
}

static void
transportSendForFilepathUnixFailedTransmitsMsg(void** state)
{
    const char* path = "/tmp/mysocket_file";

    // Create a unix address for a test socket
    struct sockaddr_un addr;
    struct stat socket_stat;
    addr.sun_family = AF_UNIX;
    scope_memset(addr.sun_path, 0, sizeof(addr.sun_path));
    scope_strncpy(addr.sun_path, path, scope_strlen(path));
    int addr_len = sizeof(sa_family_t) + scope_strlen(path) + 1;

    // Create a test socket that our transport can send to
    int sd = scope_socket(AF_UNIX, SOCK_STREAM, 0);
    if (sd == -1) {
        fail_msg("Couldn't create socket");
    }
    if (scope_bind(sd, (const struct sockaddr *)&addr, addr_len) == -1) {
        fail_msg("Couldn't bind socket");
    }
    if (scope_listen(sd, 10) == -1) {
        fail_msg("Couldn't listen on socket");
    }

    if (scope_stat(path, &socket_stat) != 0) {
        fail_msg("Couldn't stat socket");
    }

    // Take the write permission 
    if (scope_chmod(path, socket_stat.st_mode & ~(S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ){
        fail_msg("Couldn't take the write permission");
    }

    // Verify that the transport can be created, but is not connected
    transport_t* t = transportCreateUnix(path);
    assert_non_null(t);
    assert_true(transportNeedsConnection(t));
    transportDestroy(&t);

    scope_close(sd);
    scope_unlink(path);
}

static void
transportSendForFileWritesToFileAfterFlushWhenFullyBuffered(void** state)
{
    const char* path = "/tmp/mypath";
    transport_t* t = transportCreateFile(path, CFG_BUFFER_FULLY);
    assert_non_null(t);


    long file_pos_before = fileEndPosition(path);
    const char msg[] = "This is the payload message to transfer.\n";
    assert_int_equal(transportSend(t, msg, strlen(msg)), 0);
    long file_pos_after = fileEndPosition(path);

    // With CFG_BUFFER_FULLY, this output only happens with the flush
    assert_int_equal(file_pos_before, file_pos_after);

    assert_int_equal(transportFlush(t), 0);
    file_pos_after = fileEndPosition(path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // open the file
    FILE* f = fopen(path, "r");
    if (!f)
        fail_msg("Couldn't open file %s", path);
    char buf[1024];
    const int maxReadSize = sizeof(buf)-1;
    size_t bytesRead = fread(buf, 1, maxReadSize, f);
    // Provide the null ourselves.  Safe because of maxReadSize
    buf[bytesRead] = '\0';

    assert_int_equal(bytesRead, strlen(msg));
    assert_true(feof(f));
    assert_false(ferror(f));
    assert_string_equal(msg, buf);

    if (fclose(f)) fail_msg("Couldn't close file %s", path);

    transportDestroy(&t);

    if (scope_unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

static void
transportSendForFileWritesToFileImmediatelyWhenLineBuffered(void** state)
{
    const char* path = "/tmp/mypath";
    transport_t* t = transportCreateFile(path, CFG_BUFFER_LINE);
    assert_non_null(t);

    // open the file with the position at the end
    FILE* f = fopen(path, "r+");
    if (!f)
        fail_msg("Couldn't open file %s", path);
    if (fseek(f, 0, SEEK_END))
        fail_msg("Couldn't seek to end of file %s", path);

    // Since we're at the end, nothing should be there
    char buf[1024];
    const int maxReadSize = sizeof(buf)-1;
    size_t bytesRead = fread(buf, 1, maxReadSize, f);
    assert_int_equal(bytesRead, 0);
    assert_true(feof(f));
    assert_false(ferror(f));
    clearerr(f);

    const char msg[] = "This is the payload message to transfer.\n";
    assert_int_equal(transportSend(t, msg, strlen(msg)), 0);

    // Test that after the transportSend, that the msg got there.
    bytesRead = fread(buf, 1, maxReadSize, f);
    // Provide the null ourselves.  Safe because of maxReadSize
    buf[bytesRead] = '\0';

    assert_int_equal(bytesRead, strlen(msg));
    assert_true(feof(f));
    assert_false(ferror(f));
    assert_string_equal(msg, buf);

    if (fclose(f)) fail_msg("Couldn't close file %s", path);

    transportDestroy(&t);

    if (scope_unlink(path))
        fail_msg("Couldn't delete test file %s", path);
}

static void
transportTcpReconnect(void** state)
{
    const int port_test = 7890;
    const int max_try = 100;
    char port_test_str[5];
    int server_fd, res, i;
    struct sockaddr_in server;
    res = scope_snprintf(port_test_str, sizeof(port_test_str), "%d", port_test);
    assert_in_range(res, 1, sizeof(port_test_str)-1);
    for (i = 0; i < max_try; ++i) {
        server_fd = scope_socket(AF_INET, SOCK_STREAM, 0);
        assert_int_not_equal(-1, server_fd);
        server.sin_family = AF_INET;
        server.sin_addr.s_addr = INADDR_ANY;
        server.sin_port = scope_htons(port_test);

        res = scope_bind(server_fd,(struct sockaddr *)&server , sizeof(server));
        assert_int_equal(0, res);
        res = scope_listen(server_fd, 10);
        assert_int_equal(0, res);

        transport_t* t = transportCreateTCP("127.0.0.1", port_test_str, FALSE, FALSE, NULL);
        assert_non_null(t);
        assert_false(transportNeedsConnection(t));
        transportReconnect(t);
        transportDestroy(&t);
        scope_close(server_fd);
    }
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(transportCreateTcpReturnsNullPtrForNullHostOrPath),
        cmocka_unit_test(transportCreateTcpReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateTcpReturnsValidPtrForUnresolvedHostPort),
        cmocka_unit_test(transportConnectEstablishesConnection),
        cmocka_unit_test(transportCreateUdpReturnsNullPtrForNullHostOrPath),
        cmocka_unit_test(transportCreateUdpReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateUdpReturnsValidPtrForUnresolvedHostPort),
        cmocka_unit_test(transportCreateUdpHandlesGoodHostArguments),
        cmocka_unit_test(transportCreateFileReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateFileCreatesFileWithRWPermissionsForAll),
        cmocka_unit_test(transportCreateFileReturnsUnconnectedForInvalidPath),
        cmocka_unit_test(transportCreateFileCreatesDirectoriesAsNeeded),
        cmocka_unit_test(transportCreateUnixReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateUnixReturnsNullForInvalidPath),
        cmocka_unit_test(transportCreateEdgeReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateSyslogReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportCreateShmReturnsValidPtrInHappyPath),
        cmocka_unit_test(transportDestroyNullTransportDoesNothing),
        cmocka_unit_test(transportSendForNullTransportDoesNothing),
        cmocka_unit_test(transportSendForNullMessageDoesNothing),
        cmocka_unit_test(transportSendForUnimplementedTransportTypesIsHarmless),
        cmocka_unit_test(transportSendForUdpTransmitsMsg),
        cmocka_unit_test(transportSendForAbstractUnixTransmitsMsg),
        cmocka_unit_test(transportSendForFilepathUnixTransmitsMsg),
        cmocka_unit_test(transportSendForFilepathUnixFailedTransmitsMsg),
        cmocka_unit_test(transportSendForFileWritesToFileAfterFlushWhenFullyBuffered),
        cmocka_unit_test(transportSendForFileWritesToFileImmediatelyWhenLineBuffered),
        cmocka_unit_test(transportTcpReconnect),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
