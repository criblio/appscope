/* 
 * tcpserver.c - A simple TCP echo server 
 * usage: tcpserver <port>
 *
 * gcc -g test/manual/tcpserver.c -lpthread -o tcpserver
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/poll.h>

#define BUFSIZE 4096
#define MAXFDS 500
#define CMDFILE "/tmp/cmdin"

int main(int argc, char **argv) {
  int parentfd; /* parent socket */
  int childfd; /* child socket */
  int portno; /* port to listen on */
  int clientlen; /* byte size of client's address */
  struct sockaddr_in serveraddr; /* server's addr */
  struct sockaddr_in clientaddr; /* client addr */
  struct hostent *hostp; /* client host info */
  char buf[BUFSIZE]; /* message buffer */
  char *hostaddrp; /* dotted decimal host addr string */
  int optval; /* flag value for setsockopt */
  int rc, i, j, fd, arr;
  int numfds;
  int timeout;
  struct pollfd fds[MAXFDS];

  /* 
   * check command line arguments 
   */
  if (argc != 2) {
      fprintf(stderr, "usage: %s <port>\n", argv[0]);
      exit(1);
  }
  portno = atoi(argv[1]);

  /* 
   * socket: create the parent socket 
   */
  parentfd = socket(AF_INET, SOCK_STREAM, 0);
  if (parentfd < 0) {
      perror("ERROR opening socket");
      exit(1);
  }

  /* setsockopt: Handy debugging trick that lets 
   * us rerun the server immediately after we kill it; 
   * otherwise we have to wait about 20 secs. 
   * Eliminates "ERROR on binding: Address already in use" error. 
   */
  optval = 1;
  setsockopt(parentfd, SOL_SOCKET, SO_REUSEADDR, 
             (const void *)&optval , sizeof(int));

  /*
   * build the server's Internet address
   */
  bzero((char *) &serveraddr, sizeof(serveraddr));

  /* this is an Internet address */
  serveraddr.sin_family = AF_INET;

  /* let the system figure out our IP address */
  serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);

  /* this is the port we will listen on */
  serveraddr.sin_port = htons((unsigned short)portno);

  /* 
   * bind: associate the parent socket with a port 
   */
  if (bind(parentfd, (struct sockaddr *) &serveraddr, 
           sizeof(serveraddr)) < 0) {
      perror("ERROR on binding");
      exit(1);
  }
  
  /* 
   * listen: make this socket ready to accept connection requests 
   */
  if (listen(parentfd, 15) < 0) { /* allow 15 requests to queue up */ 
      perror("ERROR on listen");
      exit(1);
  }

  // wait for a connection request then echo
  clientlen = sizeof(clientaddr);

  timeout = 10 * 1000;
  bzero(fds, sizeof(fds));
  fds[0].fd = parentfd;
  fds[0].events = POLLIN;
  fds[1].fd = 0;
  fds[1].events = POLLIN;
  numfds = 2;

  while (1) {
      rc = poll(fds, numfds, -1);

      // Error or timeout from poll;
      if (rc <= 0) continue;
  
      for (i = 0; i < numfds; ++i) {
          //printf("%s:%d fds[%d].fd = %d\n", __FUNCTION__, __LINE__, i, fds[i].fd);
          if (fds[i].revents == 0) {
              //printf("%s:%d No event\n", __FUNCTION__, __LINE__);
              continue;
          }

          if (fds[i].revents & POLLHUP) {
              printf("%s:%d Disconnect on fd %d\n", __FUNCTION__, __LINE__, fd);
              close(fds[1].fd);
              fds[i].fd = -1;
              fds[i].events = 0;
              continue;
          }
          
          if (fds[i].revents & POLLERR) {
              printf("%s:%d Error on fd %d\n", __FUNCTION__, __LINE__, fd);
              close(fds[i].fd);
              fds[i].fd = -1;
              fds[i].events = 0;
              continue;
          }

          if (fds[i].revents & POLLNVAL) {
              printf("%s:%d Invalid on fd %d\n", __FUNCTION__, __LINE__, fd);
              close(fds[i].fd);
              fds[i].fd = -1;
              fds[i].events = 0;
              continue;
          }

          if (fds[i].fd == parentfd) {
              childfd = accept(parentfd, (struct sockaddr *) &clientaddr, &clientlen);
              if (childfd < 0) {
                  perror("ERROR on accept");
                  continue;
              }

              if (numfds > MAXFDS) {
                  printf("%s:%d exceeded max FDs supported\n", __FUNCTION__, __LINE__);
                  continue;
              }

              // try to re-use an entry
              for (j=0; j < numfds; j++) {
                  if (fds[j].fd == -1) {
                      fds[j].fd = childfd;
                      fds[j].events = POLLIN;
                      arr = j;
                      break;
                  }
              }

              // if not, use a new entry
              if (j >= numfds) {
                  fds[numfds].fd = childfd;
                  fds[numfds].events = POLLIN;
                  arr = numfds;
                  numfds++;
              }
                            
              // who sent the message 
              hostp = gethostbyaddr((const char *)&clientaddr.sin_addr.s_addr, 
                                    sizeof(clientaddr.sin_addr.s_addr), AF_INET);
              if (hostp == NULL) {
                  //perror("ERROR on gethostbyaddr");
                  printf("server established connection on [%d].%d\n", arr, childfd);
                  continue;
              }

              hostaddrp = inet_ntoa(clientaddr.sin_addr);
              if (hostaddrp == NULL) {
                  //perror("ERROR on inet_ntoa\n");
                  printf("server established connection on [%d].%d with %s\n", 
                         arr, childfd, hostp->h_name);
                  continue;
              }

              printf("server established connection on [%d].%d with %s (%s:%d)\n", 
                     arr, childfd, hostp->h_name, hostaddrp, htons(clientaddr.sin_port));
              break;
          } else if (fds[i].fd == 0) {
              // command input from stdin
              char *cmd;
                  
              if (fgetc(stdin) == 'U') {
                  printf("%s:%d\n", __FUNCTION__, __LINE__);
                  if ((fd = open(CMDFILE, O_RDONLY)) < 0) {
                      perror("open");
                      continue;
                  }

                  if ((cmd = calloc(1, BUFSIZE)) == NULL) {
                      perror("calloc");
                      close(fd);
                      continue;
                  }

                  rc = read(fd, cmd, (size_t)BUFSIZE);
                  if (rc <= 0) {
                      perror("read");
                      free(cmd);
                      close(fd);
                      continue;
                  }
                  
                  for (j = 2; j < numfds; j++) {
                      if ((fds[j].fd != -1) && (fds[j].fd > 2)) {
                          printf("%s:%d fds[%d].fd=%d rc %d\n%s\n", __FUNCTION__, __LINE__,
                                 j, fds[j].fd, rc, cmd);
                          if (send(fds[j].fd, cmd, rc, 0) < 0) { // MSG_DONTWAIT
                              perror("send");
                          }
                      }
                  }
                  
                  close(fd);
                  free(cmd);
              }
          } else {
              do {
                  bzero(buf, BUFSIZE);
                  rc = recv(fds[i].fd, buf, (size_t)BUFSIZE, MSG_DONTWAIT);
                  if (rc < 0) {
                      break;
                  } else if (rc == 0) {
                      // EOF
                      close(fds[i].fd);
                      fds[i].fd = -1;
                      fds[i].events = 0;
                  }
                  // echo input to stdout
                  write(1, buf, rc);
              } while (1);
          }
      }
  }
}
