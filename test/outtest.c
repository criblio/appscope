#include <stdio.h>
#include <unistd.h>
#include "out.h"

#include "test.h"

static void
outCreateReturnsValidPtr(void** state)
{
    out_t* out = outCreate();
    assert_non_null(out);
    outDestroy(&out);
    assert_null(out);
}

static void
outDestroyNullOutDoesntCrash(void** state)
{
    outDestroy(NULL);
    out_t* out = NULL;
    outDestroy(&out);
    // Implicitly shows that calling outDestroy with NULL is harmless
}

static void
outSendForNullOutDoesntCrash(void** state)
{
    char* msg = "Hey, this is cool!\n";
    assert_int_equal(outSend(NULL, msg), -1);
}

static void
outSendForNullMessageDoesntCrash(void** state)
{
    out_t* out = outCreate();
    assert_non_null(out);
    transport_t* t = transportCreateSyslog();
    assert_non_null(t);
    outTransportSet(out, t);
    assert_int_equal(outSend(out, NULL), -1);
    outDestroy(&out);
}

static void
outStatsDPrefixVerifyDefault(void** state)
{
    out_t* out = outCreate();
    assert_null(outStatsDPrefix(out));
    outDestroy(&out);
}

static void
outStatsDPrefixSetAndGet(void** state)
{
    out_t* out = outCreate();
    outStatsDPrefixSet(out, "cribl.io");
    assert_string_equal(outStatsDPrefix(out), "cribl.io");
    outStatsDPrefixSet(out, "huh");
    assert_string_equal(outStatsDPrefix(out), "huh");
    outStatsDPrefixSet(out, NULL);
    assert_null(outStatsDPrefix(out));
}

static fpos_t
fileEndPosition(char* path)
{
    FILE* f;
    if ((f = fopen(path, "r"))) {
        fseek(f, 0, SEEK_END);
        fpos_t pos;
        fgetpos(f, &pos);
        fclose(f);
        return pos;
    }
    return -1;
}

static void
outTranportSetAndOutSend(void** state)
{
    char* file_path = "/tmp/my.path";
    out_t* out = outCreate();
    assert_non_null(out);
    transport_t* t1 = transportCreateUdp("127.0.0.1", 12345);
    transport_t* t2 = transportCreateUnix("/var/run/scope.sock");
    transport_t* t3 = transportCreateSyslog();
    transport_t* t4 = transportCreateShm();
    transport_t* t5 = transportCreateFile(file_path);
    outTransportSet(out, t1);
    outTransportSet(out, t2);
    outTransportSet(out, t3);
    outTransportSet(out, t4);
    outTransportSet(out, t5);

    // Test that transport is set by testing side effects of outSend
    // affecting the file at file_path when connected to a file transport.
    long file_pos_before = fileEndPosition(file_path);
    assert_int_equal(outSend(out, "Something to send\n"), 0);
    long file_pos_after = fileEndPosition(file_path);
    assert_int_not_equal(file_pos_before, file_pos_after);

    // Test that transport is cleared by seeing no side effects.
    outTransportSet(out, NULL);
    file_pos_before = fileEndPosition(file_path);
    assert_int_equal(outSend(out, "Something to send\n"), -1);
    file_pos_after = fileEndPosition(file_path);
    assert_int_equal(file_pos_before, file_pos_after);

    if (unlink(file_path))
        fail_msg("Couldn't delete file %s", file_path);

    outDestroy(&out);
}


int
main (int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(outCreateReturnsValidPtr),
        cmocka_unit_test(outDestroyNullOutDoesntCrash),
        cmocka_unit_test(outSendForNullOutDoesntCrash),
        cmocka_unit_test(outSendForNullMessageDoesntCrash),
        cmocka_unit_test(outStatsDPrefixVerifyDefault),
        cmocka_unit_test(outStatsDPrefixSetAndGet),
        cmocka_unit_test(outTranportSetAndOutSend),
    };

    cmocka_run_group_tests(tests, NULL, NULL);

    return 0;
}
