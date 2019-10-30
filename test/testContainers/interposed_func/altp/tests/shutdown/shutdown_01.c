#include <sys/socket.h>
#include <arpa/inet.h>

#include "test_utils.h"

#define TEST_PORT 5555

int create_socket() {
    int yes = 1;
    int s = socket(AF_INET, SOCK_STREAM, 0);

    if (s == -1 || setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
        return -1;
    }

    return s;
}

int do_test() {
    int test_result = EXIT_SUCCESS;
    int pid, i = 0, j = 0;
    char msg[1024];   

    pid = fork();

    if (pid == 0) {
        for(i = 0; i < SEND_MSG_COUNT; i++) {
            struct sockaddr_in ip_addr;
            memset(&ip_addr, 0, sizeof(struct sockaddr_in));
            int s = create_socket();

            ip_addr.sin_family = AF_INET;
            ip_addr.sin_port = htons(TEST_PORT);
            ip_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

            if(connect(s, (struct sockaddr *) &ip_addr , sizeof(ip_addr)) != -1) {
                if(write(s, TEST_MSG, strlen(TEST_MSG)) != strlen(TEST_MSG)) {
                    TEST_ERROR();
                    break;
                }
            }

            if(shutdown(s, SHUT_RDWR) != 0) {
                TEST_ERROR();
            }
        }
    } else {
        struct sockaddr_in ip_addr;
        struct sockaddr_in client_addr;
        unsigned int addrlen = sizeof(client_addr);
        int client_s = 0;

        memset(&ip_addr, 0, sizeof(struct sockaddr_in));
        memset(&client_addr, 0, sizeof(struct sockaddr_in));

        int s = create_socket();

        ip_addr.sin_family = AF_INET;
        ip_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        ip_addr.sin_port=htons(TEST_PORT);

        if(bind(s, (struct sockaddr *) &ip_addr , sizeof(ip_addr) ) == -1 || listen(s, 50) == -1) {

        }

        for(j = 0; j < SEND_MSG_COUNT; j++) {
            if((client_s = accept(s , (struct sockaddr *) &client_addr, &addrlen)) != -1) {
                memset(msg,'\0',1024);
                if(read(client_s, msg, strlen(TEST_MSG)) != strlen(TEST_MSG) || strcmp(msg, TEST_MSG) != 0) {
                    TEST_ERROR();
                    break;
                }
            }

            if(shutdown(client_s, SHUT_RDWR) != 0) {
                TEST_ERROR();
            }
        }

        if(shutdown(s, SHUT_RDWR) != 0) {
            TEST_ERROR();
        }
    }
        
    return test_result;
}