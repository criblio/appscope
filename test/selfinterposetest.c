#include <errno.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test.h"

typedef struct _fn_list_t fn_list_t;
struct _fn_list_t
{
    char* name;
    fn_list_t* next;
};

static int
contains(fn_list_t* list, const char* entry)
{
    fn_list_t* c = list;
    while (c) {
        if (!strcmp(c->name, entry)) {
            return 1;
        }
        c = c->next;
    }
    return 0;
}

static void
addToList(fn_list_t** head, const char* name)
{
    if (!head) return;

    // Don't add duplicates
    if (*head && contains(*head, name)) return;

    // It's new!  Add it!
    fn_list_t* new_head = calloc(1, sizeof(fn_list_t));
    if (!new_head) return;
    new_head->name = strdup(name);
    new_head->next = *head;

    *head = new_head;
}

static void
deleteList(fn_list_t** head)
{
    if (!head) return;
    fn_list_t* c = *head;
    while (c) {
        fn_list_t* next = c->next;
        if (c->name) free(c->name);
        free(c);
        c = next;
    }
    *head = NULL;
}

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

static fn_list_t*
symbolList(FILE* input, char* filter)
{
    fn_list_t* head=NULL;

    char line[1024];
    while (getNextLine(input, line, sizeof(line))) {
        removeNewline(line);
        if (strstr(line, filter)) addToList(&head, &line[19]);
    }

    return head;
}

static void
testSymbolListFiltersCorrectlyWithCannedData(void** state)
{
    const char* path = "/tmp/nmStyleOutput.txt";

    const char* sample_nm_output[] = {
        "00000000000160b0 T accept\n",
        "0000000000016182 T accept4\n",
        "000000000000f19f T access\n",
        "                 U __assert_fail\n",
        "                 U bcopy\n",
        "0000000000016258 T bind\n",
        "0000000000252258 B __bss_start\n",
        "                 U calloc\n",
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

    FILE* f_in = fopen(path, "r");
    fn_list_t* fn_list = symbolList(f_in, " U ");
    assert_false(contains(fn_list, "accept"));
    assert_true(contains(fn_list, "__assert_fail"));
    assert_true(contains(fn_list, "bcopy"));
    assert_true(contains(fn_list, "calloc"));
    fclose(f_in);
    deleteList(&fn_list);

    // Delete the canned data file
    if (unlink(path))
        fail_msg("Couldn't delete file %s", path);
}

typedef struct {
    char* failMsgs;
    int failMsgPos;
    int errCount;
} test_state_t;

static void
checkObjectFile(fn_list_t* interpose_list, const char* file, test_state_t* s)
{
    char cmdbuf[1024];
    snprintf(cmdbuf, sizeof(cmdbuf), "nm %s", file);

    FILE* f_in = popen(cmdbuf, "r");
    fn_list_t* used_list = symbolList(f_in, " U ");
    pclose(f_in);

    fn_list_t* interpose;
    for (interpose = interpose_list; interpose; interpose=interpose->next) {
        if (contains(used_list, interpose->name)) {
            s->errCount++;
            int i = sprintf(&s->failMsgs[s->failMsgPos],
                            "  %s contains %s\n", file, interpose->name);
            if (i > 0) s->failMsgPos += i;
        }
    }
    deleteList(&used_list);
}

static void
testNoInterposedSymbolIsUsed(void** state)
{
    test_state_t s = {0};
    s.failMsgs = calloc(1, 8 * 4096);

    // Get a list of all interposed functions
    char cmdbuf[1024];
    const char* os = "linux";
#ifdef __MACOS__
    os = "macOS";
#endif // __MACOS__
    snprintf(cmdbuf, sizeof(cmdbuf), "nm ./lib/%s/libscope.so", os);
    FILE* f_in = popen(cmdbuf, "r");
    fn_list_t* interpose_list = symbolList(f_in, " T ");
    pclose(f_in);

    // Loop through each object file to see if it uses an interposed function
    glob_t glob_obj;
    glob("./test/selfinterpose/*.o", GLOB_ERR | GLOB_NOSORT, NULL, &glob_obj);
    if (glob_obj.gl_pathc < 17)
        fail_msg("expected at least 17 files in ./test/selfinterpose/*.o");
    int i;
    for (i=0; i<glob_obj.gl_pathc; i++) {
        checkObjectFile(interpose_list, glob_obj.gl_pathv[i], &s);
    }
    globfree(&glob_obj);

    // Cleanup
    deleteList(&interpose_list);

    if (s.errCount) printf("%s", s.failMsgs);
    if (s.failMsgs) free(s.failMsgs);
    if (s.errCount) fail_msg("Found use of functions we interpose.");
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(testSymbolListFiltersCorrectlyWithCannedData),
        cmocka_unit_test(testNoInterposedSymbolIsUsed),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}

