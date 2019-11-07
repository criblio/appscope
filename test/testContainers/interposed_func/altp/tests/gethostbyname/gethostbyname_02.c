#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "test_utils.h"

int do_test() {
    int test_result = EXIT_FAILURE;

    char *host = "localhost";
    struct hostent hstnm, *hp;
    char buf[8192];
    int h_errnop = 0;

    if (gethostbyname_r(host, &hstnm, buf, sizeof(buf), &hp, &h_errnop) != 0) {
        TEST_ERROR();
    } else {
        int i = 0;
        while(hstnm.h_addr_list[i] != NULL) {
            if(strcmp(inet_ntoa((struct in_addr)*((struct in_addr *)hstnm.h_addr_list[i])), "127.0.0.1") == 0) {
                test_result = EXIT_SUCCESS;
            }
            i++;
        }
    }

    return test_result;
}