#define _GNU_SOURCE

#include "coredump.h"

#include "scopetypes.h"
#include "scopestdlib.h"
#include "scopeelf.h"

#include "test.h"

static char *
generateLongFileName(size_t max){
    char *ptr = scope_calloc(1, sizeof(char) * (max + 1));
    ptr[0] = '/';
    for (int i = 1; i < max ;++i) {
        ptr[i] = 'a';
    }

    return ptr;
}

// Basic check if file specified in path is a core file
static bool
checkIfElfCoreFile(const char* path) {
    bool res = FALSE;
    int fd = scope_open(path, O_RDONLY);
    if (fd == -1) {
        goto end;
    }

    Elf64_Ehdr *header = scope_malloc(sizeof(Elf64_Ehdr));
    if (!header) {
        goto close_file;
    }

    if (scope_read(fd, header, sizeof(Elf64_Ehdr)) == -1) {
        goto free_header_buf;
    }

    // Verify ELF header
    if (header->e_ident[EI_MAG0] != ELFMAG0
        || header->e_ident[EI_MAG1] != ELFMAG1
        || header->e_ident[EI_MAG2] != ELFMAG2
        || header->e_ident[EI_MAG3] != ELFMAG3) {
        goto free_header_buf;
    }

    // Check for class
    if (header->e_ident[EI_CLASS] != ELFCLASS64) {
        goto free_header_buf;
    }
    
    // Check for type must point to core
    if (header->e_type != ET_CORE) {
        goto free_header_buf;
    }

    res = TRUE;

free_header_buf:
	scope_free(header);

close_file:
    scope_close(fd);

end:
    return res;
}

static void
coreDumpFailureTooLongPath(void** state) {
    char *path = generateLongFileName(PATH_MAX);
    pid_t pid = scope_getpid();

    bool coreRes = coreDumpGenerate(path, scope_strlen(path), pid);
    assert_false(coreRes);
    scope_free(path);
}

static void
coreDumpSuccess(void** state) {
    char *prefix = "/tmp/scope_core.";
    bool res;
    int unlinkRes;
    char pathBuf[1024] = {0};

    pid_t pid = scope_getpid();
    scope_snprintf(pathBuf, sizeof(pathBuf), "%s%d", prefix, pid);

    res = coreDumpGenerate(prefix, scope_strlen(prefix), pid);
    assert_true(res);

    res = checkIfElfCoreFile(pathBuf);
    assert_true(res);

    unlinkRes = scope_unlink(pathBuf);
    assert_int_equal(unlinkRes, 0);
}

int
main(int argc, char* argv[]) {
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(coreDumpFailureTooLongPath),
        cmocka_unit_test(coreDumpSuccess),
        cmocka_unit_test(dbgHasNoUnexpectedFailures),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
