#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "test_utils.h"

int
do_test()
{
    int test_result = EXIT_SUCCESS;

    struct addrinfo hints, *res;
    char addr[INET_ADDRSTRLEN];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo("localhost", NULL, &hints, &res) != 0) {
        TEST_ERROR();
    }

    while (res) {
        if (res->ai_family == AF_INET) {
            inet_ntop(res->ai_family, &((struct sockaddr_in *)res->ai_addr)->sin_addr, addr, INET_ADDRSTRLEN);

            if (strcmp("127.0.0.1", addr) != 0) {
                TEST_ERROR();
            }
        }

        res = res->ai_next;
    }

    return test_result;
}