#include "test.h"
#include "dbg.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int
groupSetup(void **state)
{
    dbgInit();
    return 0;
}

int
groupTeardown(void **state)
{
    // as a policy, we're saying all tests that call groupSetup and
    // groupTeardown should be aware of things that would cause dbg
    // failures, and cleanup after themselves (call dbgInit()) before
    // execution is complete.
    unsigned long long failures = dbgCountAllLines();
    if (failures) {
        dbgDumpAll(stdout);
    }
    dbgDestroy();
    return failures;
}

void
dbgHasNoUnexpectedFailures(void **state)
{
    unsigned long long failures = dbgCountAllLines();
    if (failures) {
        dbgDumpAll(stdout);
    }
    assert_false(failures);
}

void
dbgDumpAllToBuffer(char *buf, int size)
{
    FILE *f = fmemopen(buf, size, "a+");
    assert_non_null(f);
    dbgDumpAll(f);
    if (ftell(f) >= size) {
        fail_msg("size of %d was inadequate for dbgDumpAllToBuffer, "
                 "%ld was needed",
                 size, ftell(f));
    }
    if (fclose(f))
        fail_msg("Couldn't close fmemopen'd file");
}

int
writeFile(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        fail_msg("Couldn't open file");

    if (!fwrite(text, strlen(text), 1, f))
        fail_msg("Couldn't write file");

    if (fclose(f))
        fail_msg("Couldn't close file");

    return 0;
}

int
deleteFile(const char *path)
{
    return unlink(path);
}

long
fileEndPosition(const char *path)
{
    FILE *f;
    if ((f = fopen(path, "r"))) {
        fseek(f, 0, SEEK_END);
        long pos = ftell(f);
        fclose(f);
        return pos;
    }
    return -1;
}
