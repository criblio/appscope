#include <errno.h>
#include <float.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dbg.h"
#include "mtcformat.h"
#include "scopestdlib.h"

#include "test.h"

static void
mtcFormatCreateReturnsValidPtrForGoodFormat(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_non_null(fmt);
    mtcFormatDestroy(&fmt);
    assert_null(fmt);

    fmt = mtcFormatCreate(CFG_FMT_NDJSON);
    assert_non_null(fmt);
    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatCreateReturnsNullPtrForBadFormat(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FORMAT_MAX);
    assert_null(fmt);
}

void
verifyDefaults(mtc_fmt_t *fmt)
{
    assert_string_equal(mtcFormatStatsDPrefix(fmt), DEFAULT_STATSD_PREFIX);
    assert_int_equal(mtcFormatStatsDMaxLen(fmt), DEFAULT_STATSD_MAX_LEN);
    assert_int_equal(mtcFormatVerbosity(fmt), DEFAULT_MTC_VERBOSITY);
    assert_int_equal(mtcFormatCustomTags(fmt), DEFAULT_CUSTOM_TAGS);
}

static void
mtcFormatCreateHasExpectedDefaults(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    verifyDefaults(fmt);
    mtcFormatDestroy(&fmt);

    // Test that accessors work with null fmt too
    verifyDefaults(NULL);
}

static void
mtcFormatDestroyNullDoesntCrash(void **state)
{
    mtcFormatDestroy(NULL);
    mtc_fmt_t *fmt = NULL;
    mtcFormatDestroy(&fmt);
    // Implicitly shows that calling mtcFormatDestroy with NULL is harmless
}

static void
mtcFormatStatsDPrefixSetAndGet(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatStatsDPrefixSet(fmt, "cribl.io");
    assert_string_equal(mtcFormatStatsDPrefix(fmt), "cribl.io");
    mtcFormatStatsDPrefixSet(fmt, "");
    assert_string_equal(mtcFormatStatsDPrefix(fmt), "");
    mtcFormatStatsDPrefixSet(fmt, "huh");
    assert_string_equal(mtcFormatStatsDPrefix(fmt), "huh");
    mtcFormatStatsDPrefixSet(fmt, NULL);
    assert_string_equal(mtcFormatStatsDPrefix(fmt), DEFAULT_STATSD_PREFIX);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatStatsDMaxLenSetAndGet(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatStatsDMaxLenSet(fmt, 0);
    assert_int_equal(mtcFormatStatsDMaxLen(fmt), 0);
    mtcFormatStatsDMaxLenSet(fmt, UINT_MAX);
    assert_int_equal(mtcFormatStatsDMaxLen(fmt), UINT_MAX);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatVerbositySetAndGet(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_NDJSON);
    mtcFormatVerbositySet(fmt, 0);
    assert_int_equal(mtcFormatVerbosity(fmt), 0);
    mtcFormatVerbositySet(fmt, UINT_MAX);
    assert_int_equal(mtcFormatVerbosity(fmt), CFG_MAX_VERBOSITY);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatCustomTagsSetAndGet(void ** state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    {
        custom_tag_t t1 = {"name1", "value1"};
        custom_tag_t t2 = {"name2", "value2"};
        custom_tag_t *tags[] = { &t1, &t2, NULL };
        mtcFormatCustomTagsSet(fmt, tags);
        assert_non_null(mtcFormatCustomTags(fmt));
        assert_string_equal(mtcFormatCustomTags(fmt)[0]->name, "name1");
        assert_string_equal(mtcFormatCustomTags(fmt)[0]->value, "value1");
        assert_string_equal(mtcFormatCustomTags(fmt)[1]->name, "name2");
        assert_string_equal(mtcFormatCustomTags(fmt)[1]->value, "value2");
        assert_null(mtcFormatCustomTags(fmt)[2]);
    }

    custom_tag_t *tags[] = { NULL };
    mtcFormatCustomTagsSet(fmt, tags);
    assert_null(mtcFormatCustomTags(fmt));

    mtcFormatCustomTagsSet(fmt, NULL);
    assert_null(mtcFormatCustomTags(fmt));

    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputNullEventDoesntCrash(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_null(mtcFormatEventForOutput(fmt, NULL, NULL));
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputNullEventFieldsDoesntCrash(void **state)
{
    event_t e = INT_EVENT("useful.apps", 1, CURRENT, NULL);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_string_equal(msg, "useful.apps:1|g\n");
    if (msg) scope_free(msg);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputNullFmtDoesntCrash(void **state)
{
    event_field_t fields[] = {
        STRFIELD("proc",  "redis", 2,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("useful.apps", 1, CURRENT, fields);

    assert_null(mtcFormatEventForOutput(NULL, &e, NULL));
}

static void
mtcFormatEventForOutputHappyPathStatsd(void **state)
{
    char *g_hostname = "myhost";
    char *g_procname = "testapp";
    int g_openPorts = 2;
    pid_t pid = 666;
    int fd = 3;
    char *proto = "TCP";
    in_port_t localPort = 8125;

    event_field_t fields[] = {
        STRFIELD("proc",    g_procname,   2,  TRUE),
        NUMFIELD("pid",     pid,          7,  TRUE),
        NUMFIELD("fd",      fd,           7,  TRUE),
        STRFIELD("host",    g_hostname,   2,  TRUE),
        STRFIELD("proto",   proto,        1,  TRUE),
        NUMFIELD("port",    localPort,    4,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_non_null(fmt);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "net.port:%d|g|#proc:%s,pid:%d,fd:%d,host:%s,proto:%s,port:%d\n",
         g_openPorts, g_procname, pid, fd, g_hostname, proto, localPort);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);
    scope_free(msg);

    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputHappyPathJson(void **state)
{
    char *g_hostname = "myhost";
    char *g_procname = "testapp";
    int g_openPorts = 2;
    pid_t pid = 666;
    int fd = 3;
    char *proto = "TCP";
    in_port_t localPort = 8125;

    event_field_t fields[] = {
        STRFIELD("proc",    g_procname,   2,  TRUE),
        NUMFIELD("pid",     pid,          7,  TRUE),
        NUMFIELD("fd",      fd,           7,  TRUE),
        STRFIELD("host",    g_hostname,   2,  TRUE),
        STRFIELD("proto",   proto,        1,  TRUE),
        NUMFIELD("port",    localPort,    4,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_NDJSON);
    assert_non_null(fmt);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    // For usability, the json string needs to be newline terminated
    assert_int_equal(msg[strlen(msg)-1], '\n');

    cJSON *json = cJSON_Parse(msg);
    assert_non_null(json);

    assert_true(cJSON_HasObjectItem(json, "type"));

    cJSON *body = cJSON_GetObjectItem(json, "body");

    // Fields expected for all json metrics
    assert_true(cJSON_HasObjectItem(body, "_metric"));
    assert_true(cJSON_HasObjectItem(body, "_metric_type"));
    assert_true(cJSON_HasObjectItem(body, "_value"));
    // Fields expected for our specific metric
    assert_true(cJSON_HasObjectItem(body, "proc"));
    assert_true(cJSON_HasObjectItem(body, "pid"));
    assert_true(cJSON_HasObjectItem(body, "fd"));
    assert_true(cJSON_HasObjectItem(body, "host"));
    assert_true(cJSON_HasObjectItem(body, "proto"));
    assert_true(cJSON_HasObjectItem(body, "port"));
    // And... _time field is required too
    assert_true(cJSON_HasObjectItem(body, "_time"));

    cJSON_Delete(json);
    scope_free(msg);

    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputHappyPathPrometheus(void **state)
{
    char *g_hostname = "myhost";
    char *g_procname = "testapp";
    int g_openPorts = 2;
    pid_t pid = 666;
    int fd = 3;
    char *proto = "TCP";
    in_port_t localPort = 8125;

    event_field_t fields[] = {
        STRFIELD("proc",    g_procname,   2,  TRUE),
        NUMFIELD("pid",     pid,          7,  TRUE),
        NUMFIELD("fd",      fd,           7,  TRUE),
        STRFIELD("host",    g_hostname,   2,  TRUE),
        STRFIELD("proto",   proto,        1,  TRUE),
        NUMFIELD("port",    localPort,    4,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_PROMETHEUS);
    assert_non_null(fmt);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "# TYPE net_port gauge\n"
        "net_port{proc=\"%s\",pid=\"%d\",fd=\"%d\",host=\"%s\",proto=\"%s\",port=\"%d\"} %d\n",
         g_procname, pid, fd, g_hostname, proto, localPort, g_openPorts);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);
    scope_free(msg);

    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputJsonWithCustomFields(void **state)
{
    char *g_procname = "testapp";
    int g_chickenTeeth = 2;

    event_field_t fields[] = {
        STRFIELD("proc",    g_procname,   2,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("chicken.teeth", g_chickenTeeth, CURRENT, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_NDJSON);
    assert_non_null(fmt);

    custom_tag_t t1 = {"name1", "value1"};
    custom_tag_t t2 = {"name2", "value2"};
    custom_tag_t *tags[] = { &t1, &t2, NULL };
    mtcFormatCustomTagsSet(fmt, tags);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    // For usability, the json string needs to be newline terminated
    assert_int_equal(msg[strlen(msg)-1], '\n');

    cJSON *json = cJSON_Parse(msg);
    assert_non_null(json);

    cJSON *type = cJSON_GetObjectItem(json, "type");
    assert_non_null(type);
    assert_string_equal(cJSON_GetStringValue(type), "metric");

    cJSON *body = cJSON_GetObjectItem(json, "body");

    // Fields expected for all json metrics
    assert_true(cJSON_HasObjectItem(body, "_metric"));
    assert_true(cJSON_HasObjectItem(body, "_metric_type"));
    assert_true(cJSON_HasObjectItem(body, "_value"));
    // Fields expected for our specific metric
    assert_true(cJSON_HasObjectItem(body, "proc"));
    // Fields expected for custom fields (tags)
    cJSON *tag1 = cJSON_GetObjectItem(body, "name1");
    assert_non_null(tag1);
    assert_string_equal(cJSON_GetStringValue(tag1), "value1");
    cJSON *tag2 = cJSON_GetObjectItem(body, "name2");
    assert_non_null(tag2);
    assert_string_equal(cJSON_GetStringValue(tag2), "value2");

    // And... _time field is required too
    assert_true(cJSON_HasObjectItem(body, "_time"));

    cJSON_Delete(json);
    scope_free(msg);

    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputPrometheusWithCustomFields(void **state)
{
    char *g_procname = "testapp";
    float g_chickenTeeth = 2.3 / 4.3;

    event_field_t fields[] = {
        STRFIELD("proc",    g_procname,   2,  TRUE),
        FIELDEND
    };
    event_t e = FLT_EVENT("chicken.teeth", g_chickenTeeth, DELTA, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_PROMETHEUS);
    assert_non_null(fmt);

    custom_tag_t t1 = {"name1", "value1"};
    custom_tag_t t2 = {"name2", "value2"};
    custom_tag_t *tags[] = { &t1, &t2, NULL };
    mtcFormatCustomTagsSet(fmt, tags);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "# TYPE chicken_teeth counter\n"
        "chicken_teeth{name1=\"%s\",name2=\"%s\",proc=\"%s\"} %.2f\n",
         "value1", "value2", g_procname, g_chickenTeeth);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);

    scope_free(msg);
    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputHappyPathFilteredFields(void **state)
{
    char *g_hostname = "myhost";
    char *g_procname = "testapp";
    int g_openPorts = 2;
    pid_t pid = 666;
    int fd = 3;
    char *proto = "TCP";
    in_port_t localPort = 8125;

    event_field_t fields[] = {
        STRFIELD("proc",     g_procname,   2,  TRUE),
        NUMFIELD("pid",      pid,          7,  TRUE),
        NUMFIELD("fd",       fd,           7,  TRUE),
        STRFIELD("host",     g_hostname,   2,  TRUE),
        STRFIELD("proto",    proto,        1,  TRUE),
        NUMFIELD("port",     localPort,    4,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("net.port", g_openPorts, CURRENT, fields);

    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_non_null(fmt);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    regex_t re;
    assert_int_equal(regcomp(&re, "^[p]", REG_EXTENDED), 0);

    char *msg = mtcFormatEventForOutput(fmt, &e, &re);
    assert_non_null(msg);


    char expected[1024];
    int rv = snprintf(expected, sizeof(expected),
        "net.port:%d|g|#proc:%s,pid:%d,proto:%s,port:%d\n",
         g_openPorts, g_procname, pid, proto, localPort);
    assert_true(rv > 0 && rv < 1024);
    assert_string_equal(expected, msg);
    scope_free(msg);

    regfree(&re);
    mtcFormatDestroy(&fmt);
    assert_null(fmt);
}

static void
mtcFormatEventForOutputWithCustomFields(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_non_null(fmt);

    custom_tag_t t1 = {"name1", "value1"};
    custom_tag_t t2 = {"name2", "value2"};
    custom_tag_t *tags[] = { &t1, &t2, NULL };
    mtcFormatCustomTagsSet(fmt, tags);

    event_t e = INT_EVENT("statsd.metric", 3, CURRENT, NULL);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    assert_string_equal("statsd.metric:3|g|#name1:value1,name2:value2\n", msg);
    scope_free(msg);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputWithCustomAndStatsdFields(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    assert_non_null(fmt);

    custom_tag_t t1 = {"tag", "value"};
    custom_tag_t *tags[] = { &t1, NULL };
    mtcFormatCustomTagsSet(fmt, tags);

    event_field_t fields[] = {
        STRFIELD("proc",   "test",  2,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("fs.read", 3, CURRENT, fields);

    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);

    assert_string_equal("fs.read:3|g|#tag:value,proc:test\n", msg);
    scope_free(msg);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputReturnsNullIfSpaceIsInsufficient(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatStatsDMaxLenSet(fmt, 28);
    mtcFormatStatsDPrefixSet(fmt, "98");

    // Test one that just fits
    event_t e1 = INT_EVENT("A", -1234567890123456789, DELTA_MS, NULL);
    char *msg = mtcFormatEventForOutput(fmt, &e1, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "98A:-1234567890123456789|ms\n");
    assert_true(strlen(msg) == 28);
    scope_free(msg);

    // One character too much (longer name)
    event_t e2 = INT_EVENT("AB", e1.value.integer, e1.type, e1.fields);
    msg = mtcFormatEventForOutput(fmt, &e2, NULL);
    assert_null(msg);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputReturnsNullIfSpaceIsInsufficientMax(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatStatsDMaxLenSet(fmt, 2+1+315+3);
    mtcFormatStatsDPrefixSet(fmt, "98");

    // Test one that just fits
    event_t e1 = FLT_EVENT("A", -DBL_MAX, DELTA_MS, NULL);
    char *msg = mtcFormatEventForOutput(fmt, &e1, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "98A:-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00|ms\n");
    assert_true(strlen(msg) == 2+1+315+3);
    scope_free(msg);

    // One character too much (longer name)
    event_t e2 = FLT_EVENT("AB", e1.value.floating, e1.type, e1.fields);
    msg = mtcFormatEventForOutput(fmt, &e2, NULL);
    assert_null(msg);
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputVerifyEachStatsDType(void **state)
{
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);

    data_type_t t;
    for (t=DELTA; t<=SET; t++) {
        event_t e = INT_EVENT("A", 1, t, NULL);
        char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
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
        scope_free(msg);
    }

    assert_int_equal(dbgCountMatchingLines("src/mtcformat.c"), 0);

    // In undefined case, just don't crash...
    event_t e = INT_EVENT("A", 1, SET+1, NULL);
    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    if (msg) scope_free(msg);

    assert_int_equal(dbgCountMatchingLines("src/mtcformat.c"), 1);
    dbgInit(); // reset dbg for the rest of the tests
    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputOmitsFieldsIfSpaceIsInsufficient(void **state)
{
    event_field_t fields[] = {
        NUMFIELD("J",  222,   9,  TRUE),
        STRFIELD("I",  "V",   8,  TRUE),
        NUMFIELD("H",  111,   7,  TRUE),
        STRFIELD("G",  "W",   6,  TRUE),
        NUMFIELD("F",  321,   5,  TRUE),
        STRFIELD("E",  "X",   4,  TRUE),
        NUMFIELD("D",  654,   3,  TRUE),
        STRFIELD("C",  "Y",   2,  TRUE),
        NUMFIELD("B",  987,   1,  TRUE),
        STRFIELD("A",  "Z",   0,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("metric", 1, DELTA, fields);
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);
    mtcFormatVerbositySet(fmt, CFG_MAX_VERBOSITY);

    // Note that this test documents that we don't prioritize
    // the lowest cardinality fields when space is scarce.  We
    // just "fit what we can", while ensuring that we obey
    // mtcFormatStatsDMaxLen.

    mtcFormatStatsDMaxLenSet(fmt, 31);
    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);
    //                                 1111111111222222222233
    //                        1234567890123456789012345678901
    //                       "metric:1|c|#J:222,I:V,H:111,G:W,F:321,E:X,D:654"
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111\n");
    scope_free(msg);

    mtcFormatStatsDMaxLenSet(fmt, 32);
    msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#J:222,I:V,H:111,G:W\n");
    scope_free(msg);

    mtcFormatDestroy(&fmt);
}

static void
mtcFormatEventForOutputHonorsCardinality(void **state)
{
    event_field_t fields[] = {
        STRFIELD("A",   "Z",   0,  TRUE),
        NUMFIELD("B",   987,   1,  TRUE),
        STRFIELD("C",   "Y",   2,  TRUE),
        NUMFIELD("D",   654,   3,  TRUE),
        STRFIELD("E",   "X",   4,  TRUE),
        NUMFIELD("F",   321,   5,  TRUE),
        STRFIELD("G",   "W",   6,  TRUE),
        NUMFIELD("H",   111,   7,  TRUE),
        STRFIELD("I",   "V",   8,  TRUE),
        NUMFIELD("J",   222,   9,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("metric", 1, DELTA, fields);
    mtc_fmt_t *fmt = mtcFormatCreate(CFG_FMT_STATSD);

    mtcFormatVerbositySet(fmt, 0);
    char *msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z\n");
    scope_free(msg);

    mtcFormatVerbositySet(fmt, 5);
    msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);
    assert_string_equal(msg, "metric:1|c|#A:Z,B:987,C:Y,D:654,E:X,F:321\n");
    scope_free(msg);

    mtcFormatVerbositySet(fmt, 9);
    msg = mtcFormatEventForOutput(fmt, &e, NULL);
    assert_non_null(msg);
    // Verify every name is there, 10 in total
    int count =0;
    event_field_t *f;
    for (f = fields; f->value_type != FMT_END; f++) {
        assert_non_null(strstr(msg, f->name));
        count++;
    }
    assert_true(count == 10);
    scope_free(msg);

    mtcFormatDestroy(&fmt);
}


typedef struct {
    char *decoded;
    char *encoded;
} test_case_t;

static void
fmtUrlEncodeDecodeRoundTrip(void **state)
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
        test_case_t *test = &round_trip_tests[i];

        char *encoded = fmtUrlEncode(test->decoded);
        assert_non_null(encoded);
        assert_string_equal(encoded, test->encoded);

        char *decoded = fmtUrlDecode(encoded);
        assert_non_null(decoded);
        assert_string_equal(decoded, test->decoded);

        scope_free(encoded);
        scope_free(decoded);
    }

    // Verify lower case hex nibbles are tolerated
    char *decoded = fmtUrlDecode("bar%7cpound%23colon%3aampersand%40comma%2cend");
    assert_non_null(decoded);
    assert_string_equal(decoded, "bar|pound#colon:ampersand@comma,end");
    scope_free(decoded);
}

static void
fmtUrlDecodeToleratesBadData(void **state)
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
        test_case_t *test = &bad_input_tests[i];

        assert_int_equal(dbgCountMatchingLines("src/mtcformat.c"), 0);

        char *decoded = fmtUrlDecode(test->encoded);
        assert_non_null(decoded);
        assert_string_equal(decoded, test->decoded);

        assert_int_equal(dbgCountMatchingLines("src/mtcformat.c"), 1);
        dbgInit(); // reset dbg for the rest of the tests

        scope_free(decoded);
    }
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(mtcFormatCreateReturnsValidPtrForGoodFormat),
        cmocka_unit_test(mtcFormatCreateReturnsNullPtrForBadFormat),
        cmocka_unit_test(mtcFormatCreateHasExpectedDefaults),
        cmocka_unit_test(mtcFormatDestroyNullDoesntCrash),
        cmocka_unit_test(mtcFormatStatsDPrefixSetAndGet),
        cmocka_unit_test(mtcFormatStatsDMaxLenSetAndGet),
        cmocka_unit_test(mtcFormatVerbositySetAndGet),
        cmocka_unit_test(mtcFormatCustomTagsSetAndGet),
        cmocka_unit_test(mtcFormatEventForOutputNullEventDoesntCrash),
        cmocka_unit_test(mtcFormatEventForOutputNullEventFieldsDoesntCrash),
        cmocka_unit_test(mtcFormatEventForOutputNullFmtDoesntCrash),
        cmocka_unit_test(mtcFormatEventForOutputHappyPathStatsd),
        cmocka_unit_test(mtcFormatEventForOutputHappyPathJson),
        cmocka_unit_test(mtcFormatEventForOutputHappyPathPrometheus),
        cmocka_unit_test(mtcFormatEventForOutputJsonWithCustomFields),
        cmocka_unit_test(mtcFormatEventForOutputPrometheusWithCustomFields),
        cmocka_unit_test(mtcFormatEventForOutputHappyPathFilteredFields),
        cmocka_unit_test(mtcFormatEventForOutputWithCustomFields),
        cmocka_unit_test(mtcFormatEventForOutputWithCustomAndStatsdFields),
        cmocka_unit_test(mtcFormatEventForOutputReturnsNullIfSpaceIsInsufficient),
        cmocka_unit_test(mtcFormatEventForOutputReturnsNullIfSpaceIsInsufficientMax),
        cmocka_unit_test(mtcFormatEventForOutputVerifyEachStatsDType),
        cmocka_unit_test(mtcFormatEventForOutputOmitsFieldsIfSpaceIsInsufficient),
        cmocka_unit_test(mtcFormatEventForOutputHonorsCardinality),
        cmocka_unit_test(fmtUrlEncodeDecodeRoundTrip),
        cmocka_unit_test(fmtUrlDecodeToleratesBadData),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
