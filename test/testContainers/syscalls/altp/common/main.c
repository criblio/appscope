#include "print_result.h"
#include "test_utils.h"

int
main(int __attribute__((unused)) argc, char **argv)
{
    int ret = do_test();

    char *test_name = strrchr(argv[0], '/');
    printf("%-30s \t", test_name ? ++test_name : argv[0]);

    ret == EXIT_SUCCESS ? print_passed() : print_failure();

    return ret;
}