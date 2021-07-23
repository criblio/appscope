#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "dbg.h"
#include "evtformat.h"

#include "test.h"

static void
evtFormatCreateReturnsValidPtr(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatDestroy(&evt);
    assert_null(evt);
}

static void
evtFormatDestroyNullMtcDoesntCrash(void** state)
{
    evtFormatDestroy(NULL);
    evt_fmt_t* evt = NULL;
    evtFormatDestroy(&evt);
    // Implicitly shows that calling evtFormatDestroy with NULL is harmless
}

static void
evtFormatMetricHappyPath(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};

    // when enabled, we should get a non-null json
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
    cJSON* json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);

    // grab the time value from json to use in our expected output.
    // the specific value isn't of interest.
    cJSON* time = cJSON_GetObjectItem(json, "_time");
    char *timestr = cJSON_Print(time);

    char* expected = NULL;
    asprintf(&expected, "{\"sourcetype\":\"metric\","
                           "\"_time\":%s,"
                           "\"source\":\"A\","
                           "\"host\":\"host\","
                           "\"proc\":\"evttest\","
                           "\"cmd\":\"cmd-4\","
                           "\"pid\":4848,"
                           "\"data\":{\"_metric\":\"A\",\"_metric_type\":\"counter\",\"_value\":1}}", timestr);

    assert_non_null(expected);
    char* actual = cJSON_PrintUnformatted(json);
    assert_non_null(actual);
    assert_string_equal(expected, actual);
    cJSON_Delete(json);
    free(actual);
    free(expected);
    free(timestr);

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricWithSourceDisabledReturnsNull(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};

    // default is disabled
    cJSON* json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_null(json);

    // when enabled, we should get a non-null json
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Set it back to disabled, just to be sure.
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 0);
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_null(json);
    cJSON_Delete(json);

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricWithAndWithoutMatchingNameFilter(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};
    cJSON* json;

    // Default name filter allows everything
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Changing the name filter to "^B" shouldn't match.
    evtFormatNameFilterSet(evt, CFG_SRC_METRIC, "^B");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_null(json);

    // Changing the name filter to "^A" should match.
    evtFormatNameFilterSet(evt, CFG_SRC_METRIC, "^A");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricWithAndWithoutMatchingFieldFilter(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);

    event_field_t fields[] = {
        STRFIELD("proc",   "ps",  3, TRUE),
        NUMFIELD("pid",     2,    3, TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("A", 1, DELTA, fields);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};
    cJSON* json, *data;

    // Default field filter allows both fields
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    data = cJSON_GetObjectItem(json, "data");
    assert_non_null(data);
    assert_non_null(cJSON_GetObjectItem(data, "proc"));
    assert_non_null(cJSON_GetObjectItem(data, "pid"));
    cJSON_Delete(json);

    // Changing the field filter to ".*oc" should match proc but not pid
    evtFormatFieldFilterSet(evt, CFG_SRC_METRIC, ".*oc");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    data = cJSON_GetObjectItem(json, "data");
    assert_non_null(data);
    assert_non_null(cJSON_GetObjectItem(data, "proc"));
    assert_null(cJSON_GetObjectItem(data, "pid"));
    cJSON_Delete(json);

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricWithAndWithoutMatchingValueFilter(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);

    event_t e = INT_EVENT("A", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};
    cJSON* json;

    // Default value filter allows everything
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Changing the value filter to "^2" shouldn't match.
    evtFormatValueFilterSet(evt, CFG_SRC_METRIC, "^2");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_null(json);

    // Adding a field with value 2 should match.
    event_field_t fields[] = {
        STRFIELD("proc", "ps", 3, TRUE),
        NUMFIELD("pid",  2,    3, TRUE),
        FIELDEND
    };
    e.fields = fields;
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Changing the value filter to "^1" should match.
    evtFormatValueFilterSet(evt, CFG_SRC_METRIC, "^1");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Changing the value filter to "ps" should match too.
    evtFormatValueFilterSet(evt, CFG_SRC_METRIC, "ps");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_non_null(json);
    cJSON_Delete(json);

    // Changing the value filter to "blah" should not match.
    evtFormatValueFilterSet(evt, CFG_SRC_METRIC, "blah");
    json = evtFormatMetric(evt, &e, 12345, &proc);
    assert_null(json);

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricRateLimitReturnsNotice(void** state)
{
    const unsigned ratelimit = 5; // 5 events per second for test

    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
    evtFormatRateLimitSet(evt, ratelimit);

    event_t e = INT_EVENT("Hey", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};
    cJSON* json, *data;

    time_t initial, current;
    time(&initial);

    int i;
    for (i=0; i<=ratelimit; i++) {
        json = evtFormatMetric(evt, &e, 12345, &proc);
        assert_non_null(json);

        time(&current);
        if (initial != current) {
            // This test depends on running all iterations in the same second.
            // If we find this isn't true, start the loop over.
            initial = current;
            i=-1;
            cJSON_Delete(json);

            // reset evt state
            evtFormatDestroy(&evt);
            evt = evtFormatCreate();
            evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
            continue;
        }

        //printf("i=%d %s\n", i, msg);
        data = cJSON_GetObjectItem(json, "data");
        assert_non_null(data);

        if (i<ratelimit) {
            // Verify that data contains _metric, and not "Truncated"
            assert_true(cJSON_HasObjectItem(data, "_metric"));
            assert_false(cJSON_IsString(data));
        } else {
            // Verify that data contains "Truncated", and not _metric
            assert_true(cJSON_IsString(data));
            assert_non_null(strstr(data->valuestring, "Truncated"));
            assert_false(cJSON_HasObjectItem(data, "_metric"));
        }
        cJSON_Delete(json);
    }

    evtFormatDestroy(&evt);
}

static void
evtFormatMetricRateLimitCanBeTurnedOff(void** state)
{
    const unsigned ratelimit = 0; // 0 means "no limit"

    evt_fmt_t* evt = evtFormatCreate();
    assert_non_null(evt);
    evtFormatSourceEnabledSet(evt, CFG_SRC_METRIC, 1);
    evtFormatRateLimitSet(evt, ratelimit);

    event_t e = INT_EVENT("Hey", 1, DELTA, NULL);
    proc_id_t proc = {.pid = 4848,
                      .ppid = 4847,
                      .hostname = "host",
                      .procname = "evttest",
                      .cmd = "cmd-4",
                      .id = "host-evttest-cmd-4"};
    cJSON* json, *data;

    int i;
    for (i=0; i<=500000; i++) {  // 1/2 million is arbitrary
        json = evtFormatMetric(evt, &e, 12345, &proc);
        assert_non_null(json);

        //printf("i=%d %s\n", i, msg);
        data = cJSON_GetObjectItem(json, "data");
        assert_non_null(data);

        // Verify that data contains _metric, and not "Truncated"
        assert_true(cJSON_HasObjectItem(data, "_metric"));
        assert_false(cJSON_IsString(data));
        cJSON_Delete(json);
    }

    evtFormatDestroy(&evt);
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

    assert_null(fmtEventJson(NULL, NULL));

    cJSON* json = fmtEventJson(NULL, &event_format);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);

    //printf("%s:%d %s\n", __FUNCTION__, __LINE__, str);
    assert_string_equal(str, "{\"sourcetype\":\"syslog\","
                              "\"_time\":1573058085.991,"
                              "\"source\":\"stdin\","
                              "\"host\":\"earl\","
                              "\"proc\":\"formattest\","
                              "\"cmd\":\"cmd\",\"pid\":1234,"
                              "\"data\":\"поспехаў\"}");

    free(str);
    cJSON_Delete(json);
}

static void
fmtEventJsonWithCustomTags(void **state)
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

    evt_fmt_t *efmt = evtFormatCreate();
    custom_tag_t tag1 = {.name = "hey", .value = "you"};
    custom_tag_t tag2 = {.name = "this", .value = "rocks"};
    custom_tag_t *tags[] = { &tag1, &tag2, NULL };
    evtFormatCustomTagsSet(efmt, (custom_tag_t **)&tags);

    cJSON* json = fmtEventJson(efmt, &event_format);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);

    evtFormatDestroy(&efmt);

    //printf("%s:%d %s\n", __FUNCTION__, __LINE__, str);
    assert_string_equal(str, "{\"sourcetype\":\"syslog\","
                              "\"_time\":1573058085.991,"
                              "\"source\":\"stdin\","
                              "\"host\":\"earl\","
                              "\"proc\":\"formattest\","
                              "\"cmd\":\"cmd\",\"pid\":1234,"
                              "\"hey\":\"you\","
                              "\"this\":\"rocks\","
                              "\"data\":\"поспехаў\"}");

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
    cJSON* json = fmtEventJson(NULL, &event_format);
    assert_non_null(json);
    char* str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str, "{\"sourcetype\":\"console\","
                              "\"_time\":1573058085.001,"
                              "\"source\":\"stdout\","
                              "\"host\":\"earl\","
                              "\"proc\":\"\","
                              "\"cmd\":\"\","
                              "\"pid\":1234,"
                              "\"data\":\"Unë mund\\u0000të ha qelq dhe nuk\\u0000më gjen gjë\"}");

    free(str);
    cJSON_Delete(json);

    // test that null data omits a data field.
    event_format.data=NULL;
    json = fmtEventJson(NULL, &event_format);
    assert_non_null(json);
    str = cJSON_PrintUnformatted(json);
    assert_non_null(str);
    assert_string_equal(str, "{\"sourcetype\":\"console\","
                              "\"_time\":1573058085.001,"
                              "\"source\":\"stdout\","
                              "\"host\":\"earl\","
                              "\"proc\":\"\","
                              "\"cmd\":\"\","
                              "\"pid\":1234}");
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
        cJSON* json = fmtMetricJson(&e, NULL, CFG_SRC_METRIC);
        cJSON* json_type = cJSON_GetObjectItem(json, "_metric_type");
        assert_string_equal(map[type], cJSON_GetStringValue(json_type));
        if (json) cJSON_Delete(json);
    }
}

static void
fmtMetricJsonWFields(void** state)
{
    event_field_t fields[] = {
        STRFIELD("A",     "Z",  0,  TRUE),
        NUMFIELD("B",     987,  1,  TRUE),
        STRFIELD("C",     "Y",  2,  TRUE),
        NUMFIELD("D",     654,  3,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("hey", 2, HISTOGRAM, fields);
    cJSON* json = fmtMetricJson(&e, NULL, CFG_SRC_METRIC);
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
        STRFIELD("A",  "Z",  0,  TRUE),
        NUMFIELD("B",  987,  1,  TRUE),
        STRFIELD("C",  "Y",  2,  TRUE),
        NUMFIELD("D",  654,  3,  TRUE),
        FIELDEND
    };
    event_t e = INT_EVENT("hey", 2, HISTOGRAM, fields);
    regex_t re;
    assert_int_equal(regcomp(&re, "[AD]", REG_EXTENDED), 0);
    cJSON* json = fmtMetricJson(&e, &re, CFG_SRC_METRIC);
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
        cJSON* json = fmtMetricJson(&e, NULL, CFG_SRC_METRIC);
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
            STRFIELD("A",         "행운을	빕니다",    0,  TRUE),   // embedded tab
            NUMFIELD("Viel\\ Glück",     123,      1,  TRUE),   // embedded backslash
            FIELDEND
        };
        event_t e = INT_EVENT("you", 4, DELTA, fields);
        cJSON* json = fmtMetricJson(&e, NULL, CFG_SRC_METRIC);
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

static void
evtFormatValueFilterSetAndGet(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();

    /*
     * WARNING: This is hardcoded!! 
     * The default is ".*"
     * When the default changes this needs to change
    */
    regex_t* default_re = evtFormatValueFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "anythingmatches", 0, NULL, 0), 0);

    // Make sure it can be changed
    evtFormatValueFilterSet(evt, CFG_SRC_FILE, "myvalue.*");
    regex_t* new_re = evtFormatValueFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "whatever", 0, NULL, 0), REG_NOMATCH);
    assert_int_equal(regexec(new_re, "myvalue.value", 0, NULL, 0), 0);

    // Make sure default is returned for null strings
    evtFormatValueFilterSet(evt, CFG_SRC_FILE, "");
    new_re = evtFormatValueFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "anythingmatches", 0, NULL, 0), 0);

    // Make sure default is returned for bad regex
    evtFormatValueFilterSet(evt, CFG_SRC_FILE, "W![T^F?");
    new_re = evtFormatValueFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "anything", 0, NULL, 0), 0);

    evtFormatDestroy(&evt);

    // Get a default filter, even if evt is NULL
    default_re = evtFormatValueFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "whatever", 0, NULL, 0), 0);
}

static void
evtFormatFieldFilterSetAndGet(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();

    /*
     * WARNING: This is hardcoded!! 
     * The default is ".*host.*"
     * When the default changes this needs to change
    */
    regex_t* default_re = evtFormatFieldFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "host:", 0, NULL, 0), 0);

    // Make sure it can be changed
    evtFormatFieldFilterSet(evt, CFG_SRC_FILE, "myfield.*");
    regex_t* new_re = evtFormatFieldFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "whatever", 0, NULL, 0), REG_NOMATCH);
    assert_int_equal(regexec(new_re, "myfield.value", 0, NULL, 0), 0);

    // Make sure default is returned for null strings
    evtFormatFieldFilterSet(evt, CFG_SRC_FILE, "");
    new_re = evtFormatFieldFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "host.myhost", 0, NULL, 0), 0);

    // Make sure default is returned for bad regex
    evtFormatFieldFilterSet(evt, CFG_SRC_FILE, "W![T^F?");
    new_re = evtFormatFieldFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "thishost", 0, NULL, 0), 0);

    evtFormatDestroy(&evt);

    // Get a default filter, even if evt is NULL
    default_re = evtFormatFieldFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, "dohost", 0, NULL, 0), 0);
}

typedef struct {
    const char *filename;
    int matches;
} result_entry_t;

static void
evtFormatNameFilterSetAndGet(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();

    result_entry_t expected[] = {
		{"/var/log/messages", TRUE},
		{"/app/logs/stdout", TRUE},
		{"/app/logs/stderr", TRUE},
		{"/opt/cribl/log/foo.txt", TRUE},
		{"/opt/cribl/log/cribl.log", TRUE},
		{"/some/container/path.log", TRUE},
		{"/some/container/path.log1", TRUE},
		{"/some/container/path.log42", TRUE},
		{"/some/container/path.log.1", TRUE},
		{"/some/container/path.log.2", TRUE},
		{"/some/container/path.log.fancy", TRUE},
		// negative tests
		{"/opt/cribl/blog/foo.txt", FALSE},
		{"/opt/cribl/log420/foo.txt", FALSE},
		{"/opt/cribl/local/logger.yml", FALSE},
		{"/opt/cribl/local/file.logger", FALSE},
		{"/opt/cribl/local/file.420", FALSE},
        // delimiter for this array
        {NULL, FALSE}
    };
    result_entry_t* test;
    /*
     * WARNING: This is hardcoded!! 
     * The default is "(\/logs?\/)|(\.log$)|(\.log[.\d])"
     * When the default changes this needs to change
    */
    regex_t* default_re = evtFormatNameFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    for (test = expected; test->filename; test++) {
        assert_int_equal(regexec(default_re, test->filename, 0, NULL, 0),
            (test->matches) ? 0 : REG_NOMATCH);
    }
    test = expected;

    // Make sure it can be changed
    evtFormatNameFilterSet(evt, CFG_SRC_FILE, "net.*");
    regex_t* new_re = evtFormatNameFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "whatever", 0, NULL, 0), REG_NOMATCH);
    assert_int_equal(regexec(new_re, "net.tx", 0, NULL, 0), 0);

    // Make sure default is returned for null strings
    evtFormatNameFilterSet(evt, CFG_SRC_FILE, "");
    new_re = evtFormatNameFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, "anythingwithlogmatches", 0, NULL, 0), 0);

    // Make sure default is returned for bad regex
    evtFormatNameFilterSet(evt, CFG_SRC_FILE, "W![T^F?");
    new_re = evtFormatNameFilter(evt, CFG_SRC_FILE);
    assert_non_null(new_re);
    assert_int_equal(regexec(new_re, test->filename, 0, NULL, 0),
        (test->matches) ? 0 : REG_NOMATCH);

    evtFormatDestroy(&evt);

    // Get a default filter, even if evt is NULL
    default_re = evtFormatNameFilter(evt, CFG_SRC_FILE);
    assert_non_null(default_re);
    assert_int_equal(regexec(default_re, test->filename, 0, NULL, 0),
        (test->matches) ? 0 : REG_NOMATCH);
}

static void
evtFormatSourceEnabledSetAndGet(void** state)
{
    evt_fmt_t* evt = evtFormatCreate();

    // Set everything to 1
    int i, j;
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX+1; i++) {
        evtFormatSourceEnabledSet(evt, i, 1);
        if (i >= CFG_SRC_MAX) {
            assert_int_equal(evtFormatSourceEnabled(evt, i), DEFAULT_SRC_FILE);
             assert_int_equal(dbgCountMatchingLines("src/evtformat.c"), 1);
             dbgInit(); // reset dbg for the rest of the tests
        } else {
             assert_int_equal(dbgCountMatchingLines("src/evtformat.c"), 0);
             assert_int_equal(evtFormatSourceEnabled(evt, i), 1);
        }
    }

    // Clear one at a time to see there aren't side effects
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX; i++) {
        evtFormatSourceEnabledSet(evt, i, 0); // Clear it
        for (j=CFG_SRC_FILE; j<CFG_SRC_MAX; j++) {
            if (i==j)
                 assert_int_equal(evtFormatSourceEnabled(evt, j), 0);
            else
                 assert_int_equal(evtFormatSourceEnabled(evt, j), 1);
        }
        evtFormatSourceEnabledSet(evt, i, 1); // Set it back
    }

    evtFormatDestroy(&evt);

    // Test get with NULL evt
    for (i=CFG_SRC_FILE; i<CFG_SRC_MAX; i++) {
        unsigned expected;
        switch (i) {
            case CFG_SRC_FILE:
                expected = DEFAULT_SRC_FILE;
                break;
            case CFG_SRC_CONSOLE:
                expected = DEFAULT_SRC_CONSOLE;
                break;
            case CFG_SRC_SYSLOG:
                expected = DEFAULT_SRC_SYSLOG;
                break;
            case CFG_SRC_METRIC:
                expected = DEFAULT_SRC_METRIC;
                break;
            case CFG_SRC_HTTP:
                expected = DEFAULT_SRC_HTTP;
                break;
            case CFG_SRC_NET:
                expected = DEFAULT_SRC_NET;
                break;
            case CFG_SRC_FS:
                expected = DEFAULT_SRC_FS;
                break;
            case CFG_SRC_DNS:
                expected = DEFAULT_SRC_DNS;
                break;
        }

        assert_int_equal(evtFormatSourceEnabled(evt, i), expected);
    }
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(evtFormatCreateReturnsValidPtr),
        cmocka_unit_test(evtFormatDestroyNullMtcDoesntCrash),
        cmocka_unit_test(evtFormatMetricHappyPath),
        cmocka_unit_test(evtFormatMetricWithSourceDisabledReturnsNull),
        cmocka_unit_test(evtFormatMetricWithAndWithoutMatchingNameFilter),
        cmocka_unit_test(evtFormatMetricWithAndWithoutMatchingFieldFilter),
        cmocka_unit_test(evtFormatMetricWithAndWithoutMatchingValueFilter),
        cmocka_unit_test(evtFormatMetricRateLimitReturnsNotice),
        cmocka_unit_test(evtFormatMetricRateLimitCanBeTurnedOff),
        cmocka_unit_test(fmtEventJsonValue),
        cmocka_unit_test(fmtEventJsonWithCustomTags),
        cmocka_unit_test(fmtEventJsonWithEmbeddedNulls),
        cmocka_unit_test(fmtMetricJsonNoFields),
        cmocka_unit_test(fmtMetricJsonWFields),
        cmocka_unit_test(fmtMetricJsonWFilteredFields),
        cmocka_unit_test(fmtMetricJsonEscapedValues),
        cmocka_unit_test(evtFormatSourceEnabledSetAndGet),
        cmocka_unit_test(evtFormatValueFilterSetAndGet),
        cmocka_unit_test(evtFormatFieldFilterSetAndGet),
        cmocka_unit_test(evtFormatNameFilterSetAndGet),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
