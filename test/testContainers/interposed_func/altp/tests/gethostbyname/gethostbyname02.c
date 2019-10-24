#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "test_utils.h"

int do_test() {
    int test_result = EXIT_FAILURE;
    struct hostent *hstnm = 0;

    hstnm = gethostbyname2("localhost", AF_INET);

    if(hstnm) {
        int i = 0;
        while(hstnm->h_addr_list[i] != NULL) {
            if(strcmp(inet_ntoa((struct in_addr)*((struct in_addr *)hstnm->h_addr_list[i])), "127.0.0.1") == 0) {
                test_result = EXIT_SUCCESS;
            }
            i++;
        }
    } else {
        TEST_ERROR();
    }

    return test_result;
}