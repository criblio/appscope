#include <limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "format.h"

#include "test.h"


static void
fmtCreateReturnsValidPtrForGoodFormat(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    assert_non_null(fmt);
    fmtDestroy(&fmt);
    assert_null(fmt);

    fmt = fmtCreate(CFG_NEWLINE_DELIMITED);
    assert_non_null(fmt);
    fmtDestroy(&fmt);
    assert_null(fmt);
}

static void
fmtCreateReturnsNullPtrForBadFormat(void** state)
{
    format_t* fmt = fmtCreate(2);
    assert_null(fmt);
}

void
verifyDefaults(format_t* fmt)
{
    assert_string_equal(fmtStatsDPrefix(fmt), DEFAULT_STATSD_PREFIX);
    assert_int_equal(fmtStatsDMaxLen(fmt), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal(fmtOutVerbosity(fmt), DEFAULT_OUT_VERBOSITY);
}

static void
fmtCreateHasExpectedDefaults(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    verifyDefaults(fmt);
    fmtDestroy(&fmt);

    // Test that accessors work with null fmt too
    verifyDefaults(NULL);
}

static void
fmtDestroyNullDoesntCrash(void** state)
{
    fmtDestroy(NULL);
    format_t* fmt = NULL;
    fmtDestroy(&fmt);
    // Implicitly shows that calling fmtDestroy with NULL is harmless
}

static void
fmtStatsDPrefixSetAndGet(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    fmtStatsDPrefixSet(fmt, "cribl.io");
    assert_string_equal(fmtStatsDPrefix(fmt), "cribl.io");
    fmtStatsDPrefixSet(fmt, "");
    assert_string_equal(fmtStatsDPrefix(fmt), "");
    fmtStatsDPrefixSet(fmt, "huh");
    assert_string_equal(fmtStatsDPrefix(fmt), "huh");
    fmtStatsDPrefixSet(fmt, NULL);
    assert_string_equal(fmtStatsDPrefix(fmt), DEFAULT_STATSD_PREFIX);
    fmtDestroy(&fmt);
}

static void
fmtStatsDMaxLenSetAndGet(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    fmtStatsDMaxLenSet(fmt, 0);
    assert_int_equal(fmtStatsDMaxLen(fmt), 0);
    fmtStatsDMaxLenSet(fmt, UINT_MAX);
    assert_int_equal(fmtStatsDMaxLen(fmt), UINT_MAX);
    fmtDestroy(&fmt);
}

static void
fmtOutVerbositySetAndGet(void** state)
{
    format_t* fmt = fmtCreate(CFG_NEWLINE_DELIMITED);
    fmtOutVerbositySet(fmt, 0);
    assert_int_equal(fmtOutVerbosity(fmt), 0);
    fmtOutVerbositySet(fmt, UINT_MAX);
    assert_int_equal(fmtOutVerbosity(fmt), CFG_MAX_VERBOSITY);
    fmtDestroy(&fmt);
}

static void
fmtStringStatsDNullEventDoesntCrash(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    assert_null(fmtString(fmt, NULL));
    fmtDestroy(&fmt);
}

static void
fmtStringStatsDNullEventFieldsDoesntCrash(void** state)
{
    event_t e = {"useful.apps", 1, CURRENT, NULL};

    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    char* msg = fmtString(fmt, &e);
    assert_string_equal(msg, "useful.apps:1|g\n");
    if (msg) free(msg);
    fmtDestroy(&fmt);
}

static void
fmtStringNullFmtDoesntCrash(void** state)
{
    event_field_t fields[] = {
        STRFIELD("proc",             "redis",            2),
        FIELDEND
    };
    event_t e = {"useful.apps", 1, CURRENT, fields};

    assert_null(fmtString(NULL, &e));
}

static void
fmtStringStatsDHappyPath(void** state)
{
    char* g_hostname = "myhost";
    char* g_procname = "testapp";
    int g_openPorts = 2;
    pid_t pid = 666;
    int fd = 3;
    char* proto = "TCP";
    in_port_t localPort = 8125;

    event_field_t fields[] = {
        STRFIELD("proc",             g_procname,            2),
        NUMFIELD("pid",              pid,                   7),
        NUMFIELD("fd",               fd,                    7),
        STRFIELD("host",             g_hostname,            2),
        STRFIELD("proto",            proto,                 1),
        NUMFIELD("port",             localPort,             4),
        FIELDEND
    };
    event_t e = {"net.port", g_openPorts, CURRENT, fields};

    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    assert_non_null(fmt);

    char* msg = fmtString(fmt, &e);
    assert_non_null(msg);

    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "net.port:%d|g|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n",
         g_openPorts, g_procname, pid, fd, g_hostname, proto, localPort);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);
    free(msg);

    fmtDestroy(&fmt);
    assert_null(fmt);
}

static void
fmtStringStatsDReturnsNullIfSpaceIsInsufficient(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);
    fmtStatsDMaxLenSet(fmt, 28);
    fmtStatsDPrefixSet(fmt, "98");

    // Test one that just fits
    event_t e1 = {"A", -1234567890123456789, DELTA_MS, NULL};
    char* msg = fmtString(fmt, &e1);
    assert_string_equal(msg, "98A:-1234567890123456789|ms\n");
    assert_true(strlen(msg) == 28);
    free(msg);

    // One character too much (longer name)
    event_t e2 = {"AB", e1.value, e1.type, e1.fields};
    msg = fmtString(fmt, &e2);
    assert_null(msg);
    fmtDestroy(&fmt);
}

static void
fmtStringStatsDVerifyEachStatsDType(void** state)
{
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);

    data_type_t t;
    for (t=DELTA; t<=SET; t++) {
        event_t e = {"A", 1, t, NULL};
        char* msg = fmtString(fmt, &e);
        switch (t) {
            case DELTA:
                assert_string_equal(msg, "A:1|c\n");
                break;
            case CURRENT:
                assert_string_equal(msg, "A:1|g\n");
                break;
            case DELTA_MS:
                assert_string_equal(msg, "A:1|ms\n");
                break;
            case HISTOGRAM:
                assert_string_equal(msg, "A:1|h\n");
                break;
            case SET:
                assert_string_equal(msg, "A:1|s\n");
                break;
        }
        free(msg);
    }

    // In undefined case, just don't crash...
    event_t e = {"A", 1, SET+1, NULL};
    char* msg = fmtString(fmt, &e);
    if (msg) free(msg);

    fmtDestroy(&fmt);
}

static void
fmtStringStatsDOmitsFieldsIfSpaceIsInsufficient(void** state)
{
    event_field_t fields[] = {
        NUMFIELD("J",               222,                    9),
        STRFIELD("I",               "V",                    8),
        NUMFIELD("H",               111,                    7),
        STRFIELD("G",               "W",                    6),
        NUMFIELD("F",               321,                    5),
        STRFIELD("E",               "X",                    4),
        NUMFIELD("D",               654,                    3),
        STRFIELD("C",               "Y",                    2),
        NUMFIELD("B",               987,                    1),
        STRFIELD("A",               "Z",                    0),
        FIELDEND
    };
    event_t e = {"metric", 1, DELTA, fields};
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);

    // Note that this test documents that we don't prioritize
    // the lowest cardinality fields when space is scarce.  We
    // just "fit what we can", while ensuring that we obey
    // fmtStatsDMaxLen.

    fmtStatsDMaxLenSet(fmt, 31);
    char* msg = fmtString(fmt, &e);
    assert_non_null(msg);
    //                                 1111111111222222222233
    //                        1234567890123456789012345678901
    //                       "metric:1|c|#J:222,I:V,H:111,G:W,F:321,E:X,D:654"
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111\n");
    free(msg);

    fmtStatsDMaxLenSet(fmt, 32);
    msg = fmtString(fmt, &e);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111,G:W\n");
    free(msg);

    fmtDestroy(&fmt);
}

static void
fmtStringStatsDHonorsCardinality(void** state)
{
    event_field_t fields[] = {
        STRFIELD("A",               "Z",                    0),
        NUMFIELD("B",               987,                    1),
        STRFIELD("C",               "Y",                    2),
        NUMFIELD("D",               654,                    3),
        STRFIELD("E",               "X",                    4),
        NUMFIELD("F",               321,                    5),
        STRFIELD("G",               "W",                    6),
        NUMFIELD("H",               111,                    7),
        STRFIELD("I",               "V",                    8),
        NUMFIELD("J",               222,                    9),
        FIELDEND
    };
    event_t e = {"metric", 1, DELTA, fields};
    format_t* fmt = fmtCreate(CFG_EXPANDED_STATSD);

    fmtOutVerbositySet(fmt, 0);
    char* msg = fmtString(fmt, &e);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z\n");
    free(msg);

    fmtOutVerbositySet(fmt, 5);
    msg = fmtString(fmt, &e);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z,B:987,C:Y,D:654,E:X,F:321\n");
    free(msg);

    fmtOutVerbositySet(fmt, 9);
    msg = fmtString(fmt, &e);
    assert_non_null(msg);
    // Verify every name is there, 10 in total
    int count =0;
    event_field_t* f;
    for (f = fields; f->value_type != FMT_END; f++) {
        assert_non_null(strstr(msg, f->name));
        count++;
    }
    assert_true(count == 10);
    free(msg);

    fmtDestroy(&fmt);
}

static void
fmtStringNewlineDelimitedReturnsNull(void** state)
{
    // Just because it's not implemented yet...
    format_t* fmt = fmtCreate(CFG_NEWLINE_DELIMITED);
    event_t e = {"A", 1, SET, NULL};
    assert_null(fmtString(fmt, &e));
    fmtDestroy(&fmt);
}


int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fmtCreateReturnsValidPtrForGoodFormat),
        cmocka_unit_test(fmtCreateReturnsNullPtrForBadFormat),
        cmocka_unit_test(fmtCreateHasExpectedDefaults),
        cmocka_unit_test(fmtDestroyNullDoesntCrash),
        cmocka_unit_test(fmtStatsDPrefixSetAndGet),
        cmocka_unit_test(fmtStatsDMaxLenSetAndGet),
        cmocka_unit_test(fmtOutVerbositySetAndGet),
        cmocka_unit_test(fmtStringStatsDNullEventDoesntCrash),
        cmocka_unit_test(fmtStringStatsDNullEventFieldsDoesntCrash),
        cmocka_unit_test(fmtStringNullFmtDoesntCrash),
        cmocka_unit_test(fmtStringStatsDHappyPath),
        cmocka_unit_test(fmtStringStatsDReturnsNullIfSpaceIsInsufficient),
        cmocka_unit_test(fmtStringStatsDVerifyEachStatsDType),
        cmocka_unit_test(fmtStringStatsDOmitsFieldsIfSpaceIsInsufficient),
        cmocka_unit_test(fmtStringStatsDHonorsCardinality),
        cmocka_unit_test(fmtStringNewlineDelimitedReturnsNull),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
