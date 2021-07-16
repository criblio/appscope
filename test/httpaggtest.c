#define _GNU_SOURCE
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "httpagg.h"
#include "test.h"

// We have our own implementation of cmdSendMetric, so the actual
// value of this isn't really used, but it needs to be non-null.
mtc_t *bogus_mtc_addr = (mtc_t*)0xDEADBEEF;
int g_send_metric_count = 0;

// Needed for httpAggSendReport
int cmdSendMetric(mtc_t *mtc, event_t *evt)
{
    g_send_metric_count++;
    return 0;
}

static void
httpAggCreateReturnsNonNull(void **state)
{
    http_agg_t *http_agg = httpAggCreate();
    assert_non_null(http_agg);
    httpAggDestroy(&http_agg);
    assert_null(http_agg);
}

static void
httpAggDestroyForNullDoesNotCrash(void **state)
{
    http_agg_t *http_agg = NULL;
    httpAggDestroy(&http_agg);
    httpAggDestroy(NULL);
}

static void
httpAggAddMetricForNullDoesNotCrash(void **state)
{
    httpAggAddMetric(NULL, NULL, 0, 0);
}

static void
httpAggAddMetricHappyPath(void **state)
{
    http_agg_t *http_agg = httpAggCreate();

    event_field_t fields[] = {
        STRFIELD("http_target", "/", 4, FALSE),
        NUMFIELD("http_status_code", 200, 1, FALSE),
        FIELDEND
    };
    event_t event = INT_EVENT("http_server_duration", 2, DELTA_MS, fields);

    // any report before we've received events should be empty 
    assert_int_equal(g_send_metric_count, 0);
    httpAggSendReport(http_agg, bogus_mtc_addr);
    assert_int_equal(g_send_metric_count, 0);

    // Now we add one event
    httpAggAddMetric(http_agg, &event, 3, 4);

    // a report after we've received an event should have some results
    httpAggSendReport(http_agg, bogus_mtc_addr);
    int count_after_one_event = g_send_metric_count;
    assert_int_not_equal(count_after_one_event, 0);

    // until it's cleared, the report should have the same number of results
    g_send_metric_count = 0;
    httpAggSendReport(http_agg, bogus_mtc_addr);
    assert_int_equal(g_send_metric_count, count_after_one_event);

    // When we send another event with the same target and status_code
    // We'll get the same number of results because they're just aggregated together
    g_send_metric_count = 0;
    httpAggAddMetric(http_agg, &event, 3, 4);
    httpAggSendReport(http_agg, bogus_mtc_addr);
    assert_int_equal(g_send_metric_count, count_after_one_event);

    // But if we reset, then the report shouldn't send any events.
    g_send_metric_count = 0;
    httpAggReset(http_agg);
    httpAggSendReport(http_agg, bogus_mtc_addr);
    assert_int_equal(g_send_metric_count, 0);

    httpAggDestroy(&http_agg);
}

static void
httpAggAddMetricWithQueryStringsAreAggregatedTogether(void **state)
{
    http_agg_t *http_agg = httpAggCreate();

    int previous_metric_count = 0;

    char *target[] = {"/hey/dude",
                      "/hey/dude?like=what",
                      "/hey/dude?like=where&did=my&car=go",
                      "/hey/dude?leave_me=alone"};
    int i;
    for (i=0; i<(sizeof(target)/sizeof(target[0])); i++) {
        event_field_t fields[] = {
            STRFIELD("http_target", target[i], 4, FALSE),
            NUMFIELD("http_status_code", 200, 1, FALSE),
            FIELDEND
        };
        event_t event = INT_EVENT("http_client_duration", 42, DELTA_MS, fields);
        httpAggAddMetric(http_agg, &event, -1, -1);

        // We expect all of these targets to be aggregated together.
        // If this is true, than the number of received metrics should
        // should be the same each time.
        g_send_metric_count = 0;
        httpAggSendReport(http_agg, bogus_mtc_addr);
        if (previous_metric_count) {
            assert_int_equal(previous_metric_count, g_send_metric_count);
        }
        previous_metric_count = g_send_metric_count;
    }

    httpAggDestroy(&http_agg);
}

static void
httpAggAddMetricWithManyStatusCodesDoesNotCrash(void **state)
{
    http_agg_t *http_agg = httpAggCreate();

    // When this was written, MAX_CODE_ENTRIES was set to 64 in src/httpreport.c
    // 100 here is chosen to make sure we tolerate more than this.
    int i;
    for (i=1; i<=100; i++) {
        event_field_t fields[] = {
            STRFIELD("http_target", "/", 4, FALSE),
            NUMFIELD("http_status_code", i, 1, FALSE),
            FIELDEND
        };
        event_t event = INT_EVENT("http_client_duration", 2, DELTA_MS, fields);
        httpAggAddMetric(http_agg, &event, -1, -1);
    }
    httpAggSendReport(http_agg, bogus_mtc_addr);

    httpAggDestroy(&http_agg);
}

static void
httpAggAddMetricWithManyHttpTargetsDoesNotCrash(void **state)
{
    http_agg_t *http_agg = httpAggCreate();

    // When this was written, DEFAULT_TARGET_LEN was set to 128 in 
    // src/httpreport.c.  250 is used here to exercise a realloc case.
    int i;
    for (i=0; i<250; i++) {
        char http_target[128];
        snprintf(http_target, sizeof(http_target), "/%d", i);
        event_field_t fields[] = {
            STRFIELD("http_target", http_target, 4, FALSE),
            NUMFIELD("http_status_code", 200, 1, FALSE),
            FIELDEND
        };
        event_t event = INT_EVENT("http_client_duration", 2, DELTA_MS, fields);
        httpAggAddMetric(http_agg, &event, -1, -1);
    }
    httpAggSendReport(http_agg, bogus_mtc_addr);
    
    httpAggDestroy(&http_agg);
}

static void
httpAggSendReportForNullDoesNotCrash(void **state)
{
    httpAggSendReport(NULL, NULL);
}

static void
httpAggResetForNullDoesNotCrash(void **state)
{
    httpAggReset(NULL);
}

int
main(int argc, char *argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(httpAggCreateReturnsNonNull),
        cmocka_unit_test(httpAggDestroyForNullDoesNotCrash),
        cmocka_unit_test(httpAggAddMetricForNullDoesNotCrash),
        cmocka_unit_test(httpAggAddMetricHappyPath),
        cmocka_unit_test(httpAggAddMetricWithQueryStringsAreAggregatedTogether),
        cmocka_unit_test(httpAggAddMetricWithManyStatusCodesDoesNotCrash),
        cmocka_unit_test(httpAggAddMetricWithManyHttpTargetsDoesNotCrash),
        cmocka_unit_test(httpAggSendReportForNullDoesNotCrash),
        cmocka_unit_test(httpAggResetForNullDoesNotCrash)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}

