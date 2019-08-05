#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.h"

// memcpy is using GLIBC_2.14, which is way newer than anything else we
// depend on.  This dependency on a relatively new GLIBC can be avoided
// by using memmove instead which only needs GLIBC_2.2.5
//
// At the moment, GLIBC_2.4 is the newest thing we need.  This test was
// written to be the canary in the coal mine if this changes...  We'd
// expect it to fail if someone writes code that uses memcpy instead of
// memmove, or adds some other new dependency on a new version of glibc.
//
// If you're reading this, please consider whether this new dependency
// can be avoided rather than just bumping up the LATEST_LIBC_VER_NEEDED
// It's about the usability of libwrap.so.
//
// Dependencies on GLIBC versions can be observed by:
//     nm lib/linux/libwrap.so | grep GLIBC_ | fgrep -v "2.2.5"
//
// Which yields output in this format:
//     "                 U memmove@@GLIBC_2.2.5\n",
//     "                 U realpath@@GLIBC_2.3\n",
//     "                 U __sprintf_chk@@GLIBC_2.3.4\n",
//     "                 U __stack_chk_fail@@GLIBC_2.4\n",
//

#define LATEST_LIBC_VER_NEEDED "2.4"
#define LIBC "GLIBC_"

static const char*
getNextLine(FILE* input, char* line, int len)
{
    if (!input || !line) return NULL;
    return fgets(line, len, input);
}

static const char*
removeNewline(char* line)
{
    if (!line) return NULL;
    char* newline = strstr(line, "\n");
    if (newline) *newline = '\0';
    return line;
}

static const char*
verFromLine(const char* line)
{
    if (!line) return NULL;

    // find where LIBC starts
    char* ptr=strstr(line, LIBC);

    //  move past LIBC to get to the version
    if (ptr) {
        ptr += strlen(LIBC);
        return ptr;
    }
    return NULL;
}

typedef struct {
    unsigned long lines_tested;
    unsigned long lines_glibc;
    unsigned long lines_failed;
} results_t;

static void
testEachLineInStream(FILE* input, const char* libcVerThreshold, results_t* result, FILE* output)
{
    char line[1024];
    while (getNextLine(input, line, sizeof(line))) {
        removeNewline(line);
        result->lines_tested++;
        const char* lineVer = verFromLine(line);
        if (!lineVer) continue;

        // If we get here, there is a glibc version on this line
        result->lines_glibc++;
        if (strverscmp(lineVer, libcVerThreshold) <= 0) continue;

        // If we get here, the glibc version on this line is too new.
        fprintf(output, "glibc symbol:`%s` depends on newer version than agreed upon:'%s'\n", line, libcVerThreshold);
        result->lines_failed++;
    }

    if (result->lines_failed) {
        fprintf(output, "test failed with %lu glibc symbols that are too new.\n", result->lines_failed);
    } else {
        fprintf(output, "test passed; all %lu glibc symbols have acceptible versions\n", result->lines_glibc);
    }
}

static void
testEachLineInStreamWorksWithCannedData(void** state)
{
    const char* path = "/tmp/nmStyleOutput.txt";

    const char* sample_nm_output[] = {
        "                 U time@@GLIBC_2.2.5\n",
        "                 U realpath@@GLIBC_2.3\n",
        "000000000000eed5 T cfgRead\n",
        "                 U __sprintf_chk@@GLIBC_2.3.4\n",
        "                 U __stack_chk_fail@@GLIBC_2.4\n",
    };

    // Put canned data in file
    FILE* f = fopen(path, "a");
    if (!f) fail_msg("Couldn't create file");
    int i;
    for (i=0; i<sizeof(sample_nm_output)/sizeof(sample_nm_output[0]); i++) {
        if (!fwrite(sample_nm_output[i], strlen(sample_nm_output[i]), 1, f))
            fail_msg("Couldn't write file");
    }
    if (fclose(f)) fail_msg("Couldn't close file");

    // Run the test with canned data
    results_t result = {0};
    FILE* f_in = fopen(path, "r");
    FILE* f_out = fopen("/dev/null", "a");
    testEachLineInStream(f_in, "2.3", &result, f_out);
    assert_int_equal(result.lines_tested, 5);
    assert_int_equal(result.lines_glibc,  4);
    assert_int_equal(result.lines_failed, 2); // __sprintf_chk, __stack_chk_fail
    fclose(f_in);
    fclose(f_out);

    // Delete the canned data file
    if (unlink(path))
        fail_msg("Couldn't delete file %s", path);
}

static void
testEachLineInStreamWithActualLibraryData(void** state)
{
    FILE* f_in = popen("nm ./lib/linux/libwrap.so", "r");
    results_t result = {0};
    testEachLineInStream(f_in, LATEST_LIBC_VER_NEEDED, &result, stdout);
    assert_true(result.lines_tested > 350);         // 383 when written
    assert_true(result.lines_tested > 40);          // 54 when written
    assert_int_equal(result.lines_failed, 0);
    pclose(f_in);
}

int
main (int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testEachLineInStreamWorksWithCannedData),
        cmocka_unit_test(testEachLineInStreamWithActualLibraryData),
    };
    cmocka_run_group_tests(tests, NULL, NULL);
    return 0;
}
