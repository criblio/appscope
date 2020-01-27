#include <errno.h>
#include <float.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dbg.h"
#include "format.h"

#include "test.h"


static void
fmtCreateReturnsValidPtrForGoodFormat(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_non_null(fmt);
    fmtDestroy(&fmt);
    assert_null(fmt);

    fmt = fmtCreate(CFG_METRIC_JSON);
    assert_non_null(fmt);
    fmtDestroy(&fmt);
    assert_null(fmt);
}

static void
fmtCreateReturnsNullPtrForBadFormat(void** state)
{
    format_t* fmt = fmtCreate(CFG_FORMAT_MAX);
    assert_null(fmt);
}

void
verifyDefaults(format_t* fmt)
{
    assert_string_equal(fmtStatsDPrefix(fmt), DEFAULT_STATSD_PREFIX);
    assert_int_equal(fmtStatsDMaxLen(fmt), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal(fmtOutVerbosity(fmt), DEFAULT_OUT_VERBOSITY);
    assert_int_equal(fmtCustomTags(fmt), DEFAULT_CUSTOM_TAGS);
}

static void
fmtCreateHasExpectedDefaults(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
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
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
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
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    fmtStatsDMaxLenSet(fmt, 0);
    assert_int_equal(fmtStatsDMaxLen(fmt), 0);
    fmtStatsDMaxLenSet(fmt, UINT_MAX);
    assert_int_equal(fmtStatsDMaxLen(fmt), UINT_MAX);
    fmtDestroy(&fmt);
}

static void
fmtOutVerbositySetAndGet(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_JSON);
    fmtOutVerbositySet(fmt, 0);
    assert_int_equal(fmtOutVerbosity(fmt), 0);
    fmtOutVerbositySet(fmt, UINT_MAX);
    assert_int_equal(fmtOutVerbosity(fmt), CFG_MAX_VERBOSITY);
    fmtDestroy(&fmt);
}

static void
fmtCustomTagsSetAndGet(void ** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    {
        custom_tag_t t1 = {"name1", "value1"};
        custom_tag_t t2 = {"name2", "value2"};
        custom_tag_t* tags[] = { &t1, &t2, NULL };
        fmtCustomTagsSet(fmt, tags);
        assert_non_null(fmtCustomTags(fmt));
        assert_string_equal(fmtCustomTags(fmt)[0]->name, "name1");
        assert_string_equal(fmtCustomTags(fmt)[0]->value, "value1");
        assert_string_equal(fmtCustomTags(fmt)[1]->name, "name2");
        assert_string_equal(fmtCustomTags(fmt)[1]->value, "value2");
        assert_null(fmtCustomTags(fmt)[2]);
    }

    custom_tag_t* tags[] = { NULL };
    fmtCustomTagsSet(fmt, tags);
    assert_null(fmtCustomTags(fmt));

    fmtCustomTagsSet(fmt, NULL);
    assert_null(fmtCustomTags(fmt));

    fmtDestroy(&fmt);
}

static void
fmtStatsDStringNullEventDoesntCrash(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_null(fmtStatsDString(fmt, NULL, NULL));
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringNullEventFieldsDoesntCrash(void** state)
{
    event_t e = INT_EVENT("useful.apps", 1, CURRENT, NULL);

    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    char* msg = fmtStatsDString(fmt, &e, NULL);
    assert_string_equal(msg, "useful.apps:1|g\n");
    if (msg) free(msg);
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringNullFmtDoesntCrash(void** state)
{
    event_field_t fields[] = {
        STRFIELD("proc",             "redis",            2),
        FIELDEND
    };
    event_t e = INT_EVENT("useful.apps", 1, CURRENT, fields);

    assert_null(fmtStatsDString(NULL, &e, NULL));
}

static void
fmtStatsDStringHappyPath(void** state)
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
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_non_null(fmt);
    fmtOutVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char* msg = fmtStatsDString(fmt, &e, NULL);
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
fmtStatsDStringHappyPathFilteredFields(void** state)
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
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_non_null(fmt);
    fmtOutVerbositySet(fmt, CFG_MAX_VERBOSITY);

    regex_t re;
    assert_int_equal(regcomp(&re, "^[p]", REG_EXTENDED), 0);

    char* msg = fmtStatsDString(fmt, &e, &re);
    assert_non_null(msg);


    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "net.port:%d|g|#proc:%s,pid:%d,proto:%s,port:%d\n",
         g_openPorts, g_procname, pid, proto, localPort);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);
    free(msg);

    regfree(&re);
    fmtDestroy(&fmt);
    assert_null(fmt);
}

static void
fmtStatsDStringWithCustomFields(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_non_null(fmt);

    custom_tag_t t1 = {"name1", "value1"};
    custom_tag_t t2 = {"name2", "value2"};
    custom_tag_t* tags[] = { &t1, &t2, NULL };
    fmtCustomTagsSet(fmt, tags);

    event_t e = INT_EVENT("statsd.metric", 3, CURRENT, NULL);

    char* msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);

    assert_string_equal("statsd.metric:3|g|#name1:value1,name2:value2\n", msg);
    free(msg);
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringWithCustomAndStatsdFields(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    assert_non_null(fmt);

    custom_tag_t t1 = {"tag", "value"};
    custom_tag_t* tags[] = { &t1, NULL };
    fmtCustomTagsSet(fmt, tags);

    event_field_t fields[] = {
        STRFIELD("proc",             "test",                2),
        FIELDEND
    };
    event_t e = INT_EVENT("fs.read", 3, CURRENT, fields);

    char* msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);

    assert_string_equal("fs.read:3|g|#tag:value,proc:test\n", msg);
    free(msg);
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringReturnsNullIfSpaceIsInsufficient(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    fmtStatsDMaxLenSet(fmt, 28);
    fmtStatsDPrefixSet(fmt, "98");

    // Test one that just fits
    event_t e1 = INT_EVENT("A", -1234567890123456789, DELTA_MS, NULL);
    char* msg = fmtStatsDString(fmt, &e1, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "98A:-1234567890123456789|ms\n");
    assert_true(strlen(msg) == 28);
    free(msg);

    // One character too much (longer name)
    event_t e2 = INT_EVENT("AB", e1.value.integer, e1.type, e1.fields);
    msg = fmtStatsDString(fmt, &e2, NULL);
    assert_null(msg);
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringReturnsNullIfSpaceIsInsufficientMax(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    fmtStatsDMaxLenSet(fmt, 2+1+315+3);
    fmtStatsDPrefixSet(fmt, "98");

    // Test one that just fits
    event_t e1 = FLT_EVENT("A", -DBL_MAX, DELTA_MS, NULL);
    char* msg = fmtStatsDString(fmt, &e1, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "98A:-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00|ms\n");
    assert_true(strlen(msg) == 2+1+315+3);
    free(msg);

    // One character too much (longer name)
    event_t e2 = FLT_EVENT("AB", e1.value.floating, e1.type, e1.fields);
    msg = fmtStatsDString(fmt, &e2, NULL);
    assert_null(msg);
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringVerifyEachStatsDType(void** state)
{
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);

    data_type_t t;
    for (t=DELTA; t<=SET; t++) {
        event_t e = INT_EVENT("A", 1, t, NULL);
        char* msg = fmtStatsDString(fmt, &e, NULL);
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

    assert_int_equal(dbgCountMatchingLines("src/format.c"), 0);

    // In undefined case, just don't crash...
    event_t e = INT_EVENT("A", 1, SET+1, NULL);
    char* msg = fmtStatsDString(fmt, &e, NULL);
    if (msg) free(msg);

    assert_int_equal(dbgCountMatchingLines("src/format.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests
    fmtDestroy(&fmt);
}

static void
fmtStatsDStringOmitsFieldsIfSpaceIsInsufficient(void** state)
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
    event_t e = INT_EVENT("metric", 1, DELTA, fields);
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);
    fmtOutVerbositySet(fmt, CFG_MAX_VERBOSITY);

    // Note that this test documents that we don't prioritize
    // the lowest cardinality fields when space is scarce.  We
    // just "fit what we can", while ensuring that we obey
    // fmtStatsDMaxLen.

    fmtStatsDMaxLenSet(fmt, 31);
    char* msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);
    //                                 1111111111222222222233
    //                        1234567890123456789012345678901
    //                       "metric:1|c|#J:222,I:V,H:111,G:W,F:321,E:X,D:654"
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111\n");
    free(msg);

    fmtStatsDMaxLenSet(fmt, 32);
    msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111,G:W\n");
    free(msg);

    fmtDestroy(&fmt);
}

static void
fmtStatsDStringHonorsCardinality(void** state)
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
    event_t e = INT_EVENT("metric", 1, DELTA, fields);
    format_t* fmt = fmtCreate(CFG_METRIC_STATSD);

    fmtOutVerbositySet(fmt, 0);
    char* msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z\n");
    free(msg);

    fmtOutVerbositySet(fmt, 5);
    msg = fmtStatsDString(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z,B:987,C:Y,D:654,E:X,F:321\n");
    free(msg);

    fmtOutVerbositySet(fmt, 9);
    msg = fmtStatsDString(fmt, &e, NULL);
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
fmtEventJsonValue(void** state)
{
    proc_id_t proc = {.pid = 1234,
                      .ppid = 1233,
                      .hostname = "earl",
                      .procname = "formattest",
                      .cmd = "cmd",
                      .id = "earl-formattest-cmd"};
    event_format_t event_format;
    event_format.timestamp = 1573058085.991;
    event_format.src = "stdin";
    event_format.proc = &proc;
    event_format.uid = 0xCAFEBABEDEADBEEF;
    event_format.data = cJSON_CreateString("поспехаў");
    event_format.sourcetype = CFG_SRC_SYSLOG;

    assert_null(fmtEventJson(NULL));

    cJSON* json = fmtEventJson(&event_format);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);

    //printf("%s:%d %s\n", __FUNCTION__, __LINE__, str);
    assert_string_equal(str, "{\"sourcetype\":\"syslog\","
                              "\"id\":\"earl-formattest-cmd\","
                              "\"_time\":1573058085.991,"
                              "\"source\":\"stdin\","
                              "\"data\":\"поспехаў\","
                              "\"host\":\"earl\","
                              "\"_channel\":\"14627333968688430831\"}");
    free(str);
    cJSON_Delete(json);
}

static void
fmtEventJsonWithEmbeddedNulls(void** state)
{
    proc_id_t proc = {.pid = 1234,
                      .ppid = 1233,
                      .hostname = "earl",
                      .procname = "",
                      .cmd = "",
                      .id = "earl--"};
    event_format_t event_format;
    event_format.timestamp = 1573058085.001;
    event_format.src = "stdout";
    event_format.proc = &proc;
    event_format.uid = 0xCAFEBABEDEADBEEF;
    char buf[] = "Unë mund të ha qelq dhe nuk më gjen gjë";
    int datasize = strlen(buf);
    buf[9] = '\0';                  //  <-- Null in middle of buf
    buf[29] = '\0';                 //  <-- Null in middle of buf
    event_format.data = cJSON_CreateStringFromBuffer(buf, datasize);
    event_format.sourcetype = CFG_SRC_CONSOLE;
    
    // test that data has the nulls properly escaped
    cJSON* json = fmtEventJson(&event_format);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str, "{\"sourcetype\":\"console\","
                              "\"id\":\"earl--\","
                              "\"_time\":1573058085.001,"
                              "\"source\":\"stdout\","
                              "\"data\":\"Unë mund\\u0000të ha qelq dhe nuk\\u0000më gjen gjë\","
                              "\"host\":\"earl\","
                              "\"_channel\":\"14627333968688430831\"}");
    free(str);
    cJSON_Delete(json);

    // test that null data omits a data field.
    event_format.data=NULL;
    json = fmtEventJson(&event_format);
    assert_non_null(json);
    str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str, "{\"sourcetype\":\"console\","
                              "\"id\":\"earl--\","
                              "\"_time\":1573058085.001,"
                              "\"source\":\"stdout\","
                              "\"host\":\"earl\","
                              "\"_channel\":\"14627333968688430831\"}");
    free(str);
    cJSON_Delete(json);
}

static void
fmtMetricJsonNoFields(void** state)
{
    const char* map[] =
        //DELTA     CURRENT  DELTA_MS  HISTOGRAM    SET
        {"counter", "gauge", "timer", "histogram", "set", "unknown"};

    // test each value of _metric_type
    data_type_t type;
    for (type=DELTA; type<=SET+1; type++) {
        event_t e = INT_EVENT("A", 1, type, NULL);
        cJSON* json = fmtMetricJson(&e, NULL);
        cJSON* json_type = cJSON_GetObjectItem(json, "_metric_type");
        assert_string_equal(map[type], cJSON_GetStringValue(json_type));
        if (json) cJSON_Delete(json);
    }
}

static void
fmtMetricJsonWFields(void** state)
{
    event_field_t fields[] = {
        STRFIELD("A",               "Z",                    0),
        NUMFIELD("B",               987,                    1),
        STRFIELD("C",               "Y",                    2),
        NUMFIELD("D",               654,                    3),
        FIELDEND
    };
    event_t e = INT_EVENT("hey", 2, HISTOGRAM, fields);
    cJSON* json = fmtMetricJson(&e, NULL);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str,
                 "{\"_metric\":\"hey\","
                 "\"_metric_type\":\"histogram\","
                 "\"_value\":2,"
                 "\"A\":\"Z\",\"B\":987,\"C\":\"Y\",\"D\":654}");
    if (str) free(str);
    cJSON_Delete(json);
}

static void
fmtMetricJsonWFilteredFields(void** state)
{
    event_field_t fields[] = {
        STRFIELD("A",               "Z",                    0),
        NUMFIELD("B",               987,                    1),
        STRFIELD("C",               "Y",                    2),
        NUMFIELD("D",               654,                    3),
        FIELDEND
    };
    event_t e = INT_EVENT("hey", 2, HISTOGRAM, fields);
    regex_t re;
    assert_int_equal(regcomp(&re, "[AD]", REG_EXTENDED), 0);
    cJSON* json = fmtMetricJson(&e, &re);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str,
                 "{\"_metric\":\"hey\","
                 "\"_metric_type\":\"histogram\","
                 "\"_value\":2,"
                 "\"A\":\"Z\",\"D\":654}");
    if (str) free(str);
    regfree(&re);
    cJSON_Delete(json);
}

static void
fmtMetricJsonEscapedValues(void** state)
{
    {
        event_t e = INT_EVENT("Paç \"fat!", 3, SET, NULL);    // embedded double quote
        cJSON* json = fmtMetricJson(&e, NULL);
        assert_non_null(json);
        char* str = cJSON_PrintUnformatted(json);
        assert_non_null(str);
        assert_string_equal(str,
                 "{\"_metric\":\"Paç \\\"fat!\","
                 "\"_metric_type\":\"set\","
                 "\"_value\":3}");
        free(str);
        cJSON_Delete(json);
    }

    {
        event_field_t fields[] = {
            STRFIELD("A",         "행운을	빕니다",    0),   // embedded tab
            NUMFIELD("Viel\\ Glück",     123,      1),   // embedded backslash
            FIELDEND
        };
        event_t e = INT_EVENT("you", 4, DELTA, fields);
        cJSON* json = fmtMetricJson(&e, NULL);
        assert_non_null(json);
        char* str = cJSON_PrintUnformatted(json);
        assert_non_null(str);
        assert_string_equal(str,
                 "{\"_metric\":\"you\","
                 "\"_metric_type\":\"counter\","
                 "\"_value\":4,"
                 "\"A\":\"행운을\\t빕니다\","
                 "\"Viel\\\\ Glück\":123}");
        free(str);
        cJSON_Delete(json);
    }
}

typedef struct {
    char* decoded;
    char* encoded;
} test_case_t;

static void
fmtUrlEncodeDecodeRoundTrip(void** state)
{
    // Verify that NULL in returns NULL
    assert_null(fmtUrlEncode(NULL));
    assert_null(fmtUrlDecode(NULL));

    // Test things that should round trip without issues
    test_case_t round_trip_tests[] = {
        // Contains all the characters we wish to avoid in statsd
        {.decoded="bar|pound#colon:ampersand@comma,end",
         .encoded="bar%7Cpound%23colon%3Aampersand%40comma%2Cend"},
        // Other misc tests
        {.decoded="",
         .encoded=""},
        {.decoded="a b",
         .encoded="a%20b"},
        {.decoded="행운을",
         .encoded="%ED%96%89%EC%9A%B4%EC%9D%84"},
    };

    int i;
    for (i=0; i<sizeof(round_trip_tests)/sizeof(round_trip_tests[0]); i++) {
        test_case_t* test = &round_trip_tests[i];

        char* encoded = fmtUrlEncode(test->decoded);
        assert_non_null(encoded);
        assert_string_equal(encoded, test->encoded);

        char* decoded = fmtUrlDecode(encoded);
        assert_non_null(decoded);
        assert_string_equal(decoded, test->decoded);

        free(encoded);
        free(decoded);
    }

    // Verify lower case hex nibbles are tolerated
    char* decoded = fmtUrlDecode("bar%7cpound%23colon%3aampersand%40comma%2cend");
    assert_non_null(decoded);
    assert_string_equal(decoded, "bar|pound#colon:ampersand@comma,end");
    free(decoded);
}

static void
fmtUrlDecodeToleratesBadData(void** state)
{
    // Verify that malformed input is tolerated on decode
    test_case_t bad_input_tests[] ={
        {.decoded="",
         .encoded="%"},
        {.decoded="",
         .encoded="%0"},
        {.decoded="",
         .encoded="%G0"},
        {.decoded="",
         .encoded="%0g"},
        {.decoded="hey",
         .encoded="hey%0gyou"},
        {.decoded="hey",
         .encoded="hey%%0you"},
        {.decoded="hey",
         .encoded="hey%1gyou"},
        {.decoded="hey",
         .encoded="hey%%1you"},
    };

    int i;
    for (i=0; i<sizeof(bad_input_tests)/sizeof(bad_input_tests[0]); i++) {
        test_case_t* test = &bad_input_tests[i];

        assert_int_equal(dbgCountMatchingLines("src/format.c"), 0);

        char* decoded = fmtUrlDecode(test->encoded);
        assert_non_null(decoded);
        assert_string_equal(decoded, test->decoded);

        assert_int_equal(dbgCountMatchingLines("src/format.c"), 1);
        dbgInit(); // reset dbg for the rest of the tests

        free(decoded);
    }
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
        cmocka_unit_test(fmtCustomTagsSetAndGet),
        cmocka_unit_test(fmtStatsDStringNullEventDoesntCrash),
        cmocka_unit_test(fmtStatsDStringNullEventFieldsDoesntCrash),
        cmocka_unit_test(fmtStatsDStringNullFmtDoesntCrash),
        cmocka_unit_test(fmtStatsDStringHappyPath),
        cmocka_unit_test(fmtStatsDStringHappyPathFilteredFields),
        cmocka_unit_test(fmtStatsDStringWithCustomFields),
        cmocka_unit_test(fmtStatsDStringWithCustomAndStatsdFields),
        cmocka_unit_test(fmtStatsDStringReturnsNullIfSpaceIsInsufficient),
        cmocka_unit_test(fmtStatsDStringReturnsNullIfSpaceIsInsufficientMax),
        cmocka_unit_test(fmtStatsDStringVerifyEachStatsDType),
        cmocka_unit_test(fmtStatsDStringOmitsFieldsIfSpaceIsInsufficient),
        cmocka_unit_test(fmtStatsDStringHonorsCardinality),
        cmocka_unit_test(fmtEventJsonValue),
        cmocka_unit_test(fmtEventJsonWithEmbeddedNulls),
        cmocka_unit_test(fmtMetricJsonNoFields),
        cmocka_unit_test(fmtMetricJsonWFields),
        cmocka_unit_test(fmtMetricJsonWFilteredFields),
        cmocka_unit_test(fmtMetricJsonEscapedValues),
        cmocka_unit_test(fmtUrlEncodeDecodeRoundTrip),
        cmocka_unit_test(fmtUrlDecodeToleratesBadData),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
