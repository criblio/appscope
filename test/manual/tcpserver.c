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

#define BUFSIZE 4096

/*
 * error - wrapper for perror
 */
void error(char *msg) {
  perror(msg);
  //exit(1);
}

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
  int n; /* message byte size */
  pthread_t tid;
  fd_set workfds, masterfds;
  struct timeval tv;
  int rc, i, nfds;
  
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
  if (parentfd < 0) 
      error("ERROR opening socket");

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
           sizeof(serveraddr)) < 0) 
      error("ERROR on binding");
  
  /* 
   * listen: make this socket ready to accept connection requests 
   */
  if (listen(parentfd, 15) < 0) /* allow 15 requests to queue up */ 
      error("ERROR on listen");

  FD_ZERO(&masterfds);
  FD_SET(parentfd, &masterfds);
  tv.tv_sec = 5;
  tv.tv_usec = 0;
  nfds = parentfd;
  
  // wait for a connection request then echo
  clientlen = sizeof(clientaddr);
  while (1) {
      memcpy(&workfds, &masterfds, sizeof(masterfds));
      rc = select(nfds + 1, &workfds, NULL, NULL, NULL);

      for (i = 0; i <= nfds; ++i) {
          if (FD_ISSET (i, &workfds)) {
              if (i == parentfd) {
                  childfd = accept(parentfd, (struct sockaddr *) &clientaddr, &clientlen);
                  if (childfd < 0) 
                      error("ERROR on accept");                  

                  FD_SET(childfd, &masterfds);
                  if (childfd > nfds)
                      nfds = childfd;

                  // who sent the message 
                  hostp = gethostbyaddr((const char *)&clientaddr.sin_addr.s_addr, 
                                        sizeof(clientaddr.sin_addr.s_addr), AF_INET);
                  if (hostp == NULL)
                      error("ERROR on gethostbyaddr");

                  hostaddrp = inet_ntoa(clientaddr.sin_addr);
                  if (hostaddrp == NULL)
                      error("ERROR on inet_ntoa\n");

                  printf("server established connection on %d with %s (%s)\n", 
                         childfd, hostp->h_name, hostaddrp);
                  break;
              } else {
                  do {
                      bzero(buf, BUFSIZE);
                      rc = read(i, buf, BUFSIZE);
                      if (rc < 0) 
                          error("ERROR reading from socket");
      
                      // echo input to stdout
                      n = write(1, buf, strlen(buf));
                      if (n < 0) 
                          error("ERROR writing to stdout");
                  } while (rc > 0);
                  FD_CLR(i, &masterfds);
                  nfds -= 1;
                  break;
              }
          }
      }
  }
}
