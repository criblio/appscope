#define _GNU_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <string.h>
#include <unistd.h>

// The intention of the this test is to check handling NULL
// value in case of msg by the scope
// Calling `sendmmsg`/`recvmmsg` with empty values can be used
// for checking the support for the calls

int main() {
  int sockfd, ret;
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
      perror("socket");
      exit(EXIT_FAILURE);
  }
  ret = sendmmsg(sockfd, NULL, 0, 0);
  fprintf(stdout, "\nsendmmsg returned %d", ret);
  ret = recvmmsg(sockfd, NULL, 0, 0, NULL);
  fprintf(stdout, "\nrecvmmsg returned %d", ret);
  
  return 0;
}
