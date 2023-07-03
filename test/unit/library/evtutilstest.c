#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "evtutils.h"
#include "test.h"
#include "scopestdlib.h"

static void
evtProtoAllocHttp1AndFree(void **state) {
    protocol_info *proto = evtProtoAllocHttp1(TRUE);
    assert_non_null(proto);

    // alloc'd sub-fields better be non-null too
    assert_non_null((http_post *)proto->data);

    evtProtoFree(proto);
}

static void
evtProtoAllocHttp2FrameAndFree(void **state) {
    protocol_info *proto = evtProtoAllocHttp2Frame(1);
    assert_non_null(proto);

    // alloc'd sub-fields better be non-null too
    http_post *post = (http_post *)proto->data;
    assert_non_null(post);
    assert_non_null((char *)post->hdr);

    evtProtoFree(proto);
}

static void
evtProtoAllocDetectAndFree(void **state) {
    protocol_info *proto = evtProtoAllocDetect(NULL);
    assert_null(proto);

    proto = evtProtoAllocDetect("a string goes here");
    assert_non_null(proto);

    // alloc'd fields better be non-null too
    assert_non_null((char *)proto->data);

    evtProtoFree(proto);
}

static void
evtProtoFreeDoesNotCrash(void **state) {
    // Shouldn't crash
    evtProtoFree(NULL);
}

static void
evtFreeDoesNotCrash(void **state) {

    // Shouldn't crash
    evtFree(NULL);

    protocol_info *proto[3];
    int i = 0;
    proto[i++] = evtProtoAllocHttp1(FALSE);
    proto[i++] = evtProtoAllocHttp2Frame(12345);
    proto[i++] = evtProtoAllocDetect("This is the protocol name");

    for (i--; i>=0; i--) {
        assert_non_null(proto[i]);
        evtFree((evt_type *)proto[i]);
    }
}


int
main(int argc, char* argv[])
{
    scope_printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(evtProtoAllocHttp1AndFree),
        cmocka_unit_test(evtProtoAllocHttp2FrameAndFree),
        cmocka_unit_test(evtProtoAllocDetectAndFree),
        cmocka_unit_test(evtProtoFreeDoesNotCrash),
        cmocka_unit_test(evtFreeDoesNotCrash),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
