#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.h"

// secure_getenv (used by libcrypto.a) is using GLIBC_2.17.  This is the
// newest glibc symbol that we depend on.  This means that we don't expect
// our library to run on anything older than 2012-12-25, which was the
// release date for 2.17.
//
// LATEST_LIBC_VER_NEEDED defines the newest glibc version we need.
// This test was written to be the canary in the coal mine if this changes...
// We'd expect it to fail if someone writes code that adds any dependency
// on a newer version of glibc.
//
// If you're reading this, please consider whether this new dependency
// can be avoided rather than just bumping up the LATEST_LIBC_VER_NEEDED
// It's about the portability/usability of libscope.so.
//
// Dependencies on GLIBC versions can be observed by:
//     nm lib/linux/libscope.so | grep GLIBC_ | fgrep -v "2.2.5"
//
// Which yields output in this format:
//     "                 U memmove@@GLIBC_2.2.5\n",
//     "                 U realpath@@GLIBC_2.3\n",
//     "                 U __sprintf_chk@@GLIBC_2.3.4\n",
//     "                 U __stack_chk_fail@@GLIBC_2.4\n",
//

#define LATEST_LIBC_VER_NEEDED "2.17"
#define LIBC "GLIBC_"

#define PRIVATE "PRIVATE"
static const char* ok_private_funcs[] = {
    "_dl_sym",
    "_dl_argv",
    "__ctype_init"
};

static const char* funcs_to_avoid[] = {
    "setenv",
};

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
    unsigned long lines_glibc_private;
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

        // If we get here, GLIBC_ appears, but it might be GLIBC_PRIVATE.
        // If so, make sure the GLIBC_PRIVATE function is one we've seen before.
        result->lines_glibc++;
        if (!strcmp(lineVer, PRIVATE)) {
            result->lines_glibc_private++;
            int allowed = 0;
            int i;
            for (i=0; i<sizeof(ok_private_funcs)/sizeof(ok_private_funcs[0]); i++) {
                if (strstr(line, ok_private_funcs[i])) {
                    allowed = 1;
                    break;
                }
            }
            if (!allowed) {
                fprintf(output, "glibc symbol: '%s' is a private function we haven't seen before\n", line);
                result->lines_failed++;
            }
            continue;
        }

        // If we get here, there is a glibc version number on this line
        if (strverscmp(lineVer, libcVerThreshold) <= 0) continue;

        // If we get here, the glibc version on this line is too new.
        fprintf(output, "glibc symbol:`%s` depends on newer version than agreed upon:'%s'\n", line, libcVerThreshold);
        result->lines_failed++;
    }

    if (result->lines_failed) {
        fprintf(output, "test failed with %lu glibc symbols that are too new.\n", result->lines_failed);
    } else {
        fprintf(output, "test passed; all %lu glibc symbols have acceptable versions\n", result->lines_glibc);
    }
}

static void
testEachLineInStreamWorksWithCannedData(void** state)
{
    const char* path = "/tmp/nmStyleOutput.txt";

    const char* sample_nm_output[] = {
        "                 U __libc_dlsym@@GLIBC_PRIVATE\n",
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
    assert_int_equal(result.lines_tested, 6);
    assert_int_equal(result.lines_glibc,  5);
    assert_int_equal(result.lines_glibc_private,  1); // __libc_dlsym
    assert_int_equal(result.lines_failed, 3); // __sprintf_chk, __stack_chk_fail, __libc_dlsym
    fclose(f_in);
    fclose(f_out);

    // Delete the canned data file
    if (unlink(path))
        fail_msg("Couldn't delete file %s", path);
}

static void
testEachLineInStreamWithActualLibraryData(void** state)
{
    FILE* f_in = popen("nm ./lib/linux/libscope.so", "r");
    results_t result = {0};
    //FILE* f_out = fopen("/dev/null", "a");
    //                              replace f_out with stdout for debugging...
    testEachLineInStream(f_in, LATEST_LIBC_VER_NEEDED, &result, stdout);
    assert_true(result.lines_tested > 350);            // 383 when written
    assert_true(result.lines_glibc > 40);              // 54 when written
    // moving out of the library
    //assert_int_equal(result.lines_glibc_private,  3);  // because of _dl_sym, _dl_argv
    assert_int_equal(result.lines_failed, 0);
    pclose(f_in);
    //fclose(f_out);
}

// This test was created after we found that setenv was overridden in bash.
// When our library called setenv within a bash process, it did not
// work as expected, because the glibc version was not getting called.
// (in this case not even indirectly through bash!)
//
// We can work around this by ensuring we call glibc functions directly.
// See g_fn.setenv for an example of how to do this.
//
// Rather than let this kind of problem happen again silently, let's
// detect at build-time if we're using any (glibc) functions we've had
// troubles with.  Most of glibc's exported symbols are WEAK, and therefore
// could theoretically be in this category, but it seems particularly
// strange that bash didn't ever call glibc's setenv.  Until it becomes
// necessary to stop using glibc funcs altogether, this can be the list
// of those we want to avoid for now.
static void
testLibraryUsesFunctionsWeShouldNot(void **state)
{
    FILE* f_in = popen("nm ./lib/linux/libscope.so", "r");
    unsigned int lines_failed = 0;

    char line[1024];
    while (getNextLine(f_in, line, sizeof(line))) {
        removeNewline(line);

        int i, found = 0;
        for (i=0; i<sizeof(funcs_to_avoid)/sizeof(funcs_to_avoid[0]); i++) {
            char pattern[1024];
            snprintf(pattern, sizeof(pattern), " U %s@", funcs_to_avoid[i]);
            if (strstr(line, pattern)) {
                found = 1;
                break;
            }
        }
        if (found) {
            fprintf(stdout, "glibc symbol: '%s' is a function that we don't want to use\n", line);
            lines_failed++;
        }
    }

    if (lines_failed) {
        fail_msg("test failed with %u glibc functions we don't want to use.\n", lines_failed);
    }

    pclose(f_in);
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testEachLineInStreamWorksWithCannedData),
        cmocka_unit_test(testEachLineInStreamWithActualLibraryData),
        cmocka_unit_test(testLibraryUsesFunctionsWeShouldNot),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
