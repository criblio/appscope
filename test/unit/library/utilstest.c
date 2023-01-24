#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "utils.h"

#include "fn.h"
#include "test.h"

static void
testSetUUID(void **state)
{
    char uuid[37];
    setUUID(uuid);

    assert_int_equal(strlen(uuid), 36);

    // UUID version is 4
    assert_int_equal(uuid[14] - '0', 4);

    assert_true(uuid[8] == '-');
    assert_true(uuid[13] == '-');
    assert_true(uuid[18] == '-');
    assert_true(uuid[23] == '-');

    // UUID is unique across calls
    char uuid_2[37];
    setUUID(uuid_2);

    assert_true(strcmp(uuid, uuid_2));
}

static void
testSetMachineID(void **state)
{
    char mach_id[33];
    setMachineID(mach_id);

    assert_int_equal(strlen(mach_id), 32);

    char mach_id_2[33];
    setMachineID(mach_id_2);

    // Machine ID is consistent across calls
    assert_true(!strcmp(mach_id, mach_id_2));
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    initFn();

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testSetUUID),
        cmocka_unit_test(testSetMachineID),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
