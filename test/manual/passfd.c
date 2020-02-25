/*
 * passfd.c - Test to see that Scope is handling the passing of fd access
 * rights between proceses correctly
 *
 * gcc -g test/manual/passfd.c -lpthread -o test/linux/passfd
 *
 * 1) passfd is a parent that sends access rights to a child process over a UNIX socket.
 *
 * 2) In order to send a file or socket descriptor, the parent opens
 *    a file and creates a TCP socket. In order to get a TCP socket descriptor,
 *    the parent creates a thread which implements the server-side (accept) socket.
 *
 * 3) The parent sends the access rights of the file &/or socket descriptor
 *    to the child over the UNIX socket.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/wait.h>

#define TESTFILE "/tmp/pass.txt"
#define EVENTFILE "/tmp/scope_events.log"
#define TESTPORT "9009"
#define TESTPORTNO 9009
#define HOST "localhost"

#define CMSG_OK(mhdr, cmsg) ((cmsg)->cmsg_len >= sizeof(struct cmsghdr) && \
                             (cmsg)->cmsg_len <= (unsigned long)        \
                             ((mhdr)->msg_controllen -                  \
                              ((char *)(cmsg) - (char *)(mhdr)->msg_control)))

char *pfile;

void
usage(char *prog) {
    fprintf(stderr,"usage: %s [-v] -f pipe/file\n", prog);
    exit(-1);
}

int
check_event(char **validation, int numval)
{
    int fd, rc, i;
    char *buf;
    struct stat sbuf;

    if ((fd = open(EVENTFILE, O_RDONLY)) == -1) {
        perror("open:check_event");
        return -1;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("fstat:check_event");
        return -1;
    }

    if ((buf = calloc(1, sbuf.st_size)) == NULL) {
        perror("calloc:check_event");
        return -1;
    }

    if ((read(fd, buf, sbuf.st_size)) <= 0) {
        perror("read:check_event");
        return -1;
    }

    // look for validation strings within the same event line
    if (numval > 0) {
        char *thisval;
        if ((thisval = strstr(buf, validation[0])) != NULL) {
            for (i = 1; i < numval; i++) {
                if ((thisval = strstr(thisval, validation[i])) == NULL) {
                    rc = -1;
                    break;
                }
            }
        } else {
            rc = -1;
        }
    }

    if (close(fd) == -1) {
        perror("close:check_event");
        return -1;
    }

    if (unlink(EVENTFILE) == -1) {
        perror("unlink:check_event");
        return -1;
    }

    if (buf) free(buf);
    return rc;
}

int
send_array_rights(int sd, int *sendfds, int numfds)
{
    int clen;
    struct msghdr msg;
    struct iovec iov[1];
    struct cmsghdr *cmptr;
    char control[CMSG_SPACE(sizeof(int) * numfds)];
    char tdata[] = "test";

    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int) * numfds);
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memcpy(CMSG_DATA(cmptr), sendfds, sizeof(int) * numfds);
    fprintf(stderr, "Parent:numfds %d controllen %ld cmsg_len %ld\n",
            numfds, msg.msg_controllen, cmptr->cmsg_len);

    msg.msg_name = NULL;
    msg.msg_namelen = 0;

    iov[0].iov_base = tdata;
    iov[0].iov_len = strlen(tdata);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    fprintf(stderr, "Parent:send_array_rights\n");
    if ((clen = sendmsg(sd, &msg, 0)) == -1) {
        perror("sendmsg:send_array_rights");
        return -1;
    }

    fprintf(stderr, "Parent:sent %d bytes\n", clen);
    return 0;
}

/*
 * This is not used. Left it here in case
 * we find that this changes or other UNIX
 * releases do support this.
 *
 * The Linux kernel does not allow multiple
 * cmsg headers sent in a single message.
 * Ref Linux source:
 * net/unix/af_unix.c:unix_stream_sendmsg()
 *
 */
int
send_multiple_rights(int sd, int *sendfds, int numfds)
{
    int clen;
    size_t csize;
    struct msghdr msg;
    struct iovec iov;
    struct cmsghdr *cmptr;
    //struct ucred *credp;
    //char control[CMSG_SPACE(sizeof(int) * numfds)];
    char *control;
    char tdata[] = "test";

    csize = CMSG_SPACE(sizeof(int)) + CMSG_SPACE(sizeof(int));
    if ((control = malloc(CMSG_LEN(csize))) == NULL) return -1;
    memset(control, 0, CMSG_LEN(csize));
    memset(&iov, 0, sizeof(iov));

    msg.msg_control = control;
    msg.msg_controllen = csize;

    cmptr = CMSG_FIRSTHDR(&msg);
    if (!cmptr) {
        fprintf(stderr, "Can't get a first header\n");
        free(control);
        return -1;
    }
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    *(int *) CMSG_DATA(cmptr) = sendfds[0];
    if (!CMSG_OK(&msg, cmptr)) {
        fprintf(stderr, "ERROR: self test1: invalid argument\n");
        free(control);
        return -1;
    }

    CMSG_NXTHDR(&msg, cmptr);
    if (!cmptr) {
        fprintf(stderr, "Can't get a second header\n");
        free(control);
        return -1;
    }
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    *(int *) CMSG_DATA(cmptr) = sendfds[1];
    if (!CMSG_OK(&msg, cmptr)) {
        fprintf(stderr, "ERROR: self test2: invalid argument\n");
        free(control);
        return -1;
    }

    msg.msg_name = NULL;
    msg.msg_namelen = 0;

    iov.iov_base = tdata;
    iov.iov_len = strlen(tdata);
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    fprintf(stderr, "cmsghdr %ld control 0x%p ctl len %ld msg1 0x%p msg2 0x%p\n",
            sizeof(struct cmsghdr), msg.msg_control, msg.msg_controllen,
            CMSG_FIRSTHDR(&msg), CMSG_NXTHDR(&msg, cmptr));

    fprintf(stderr, "Parent:send_multiple_rights\n");
    if ((clen = sendmsg(sd, &msg, 0)) == -1) {
        perror("sendmsg:send_multiple_rights");
        free(control);
        return -1;
    }

    fprintf(stderr, "Parent:sent %d bytes\n", clen);
    free(control);
    return 0;
}

int
send_access_rights(int sd, int sendfd)
{
    int clen;
    struct msghdr msg;
    struct iovec iov[1];
    struct cmsghdr *cmptr;
    char control[CMSG_SPACE(sizeof(int))];
    char tdata[] = "test";


    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    *((int *) CMSG_DATA(cmptr)) = sendfd;


    msg.msg_name = NULL;
    msg.msg_namelen = 0;

    iov[0].iov_base = tdata;
    iov[0].iov_len = strlen(tdata);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;


    fprintf(stderr, "Parent:send_access_rights\n");
    if ((clen = sendmsg(sd, &msg, 0)) == -1) {
        perror("sendmsg:send_access_rights");
        return -1;
    }

    fprintf(stderr, "Parent:sent %d bytes\n", clen);
    return 0;
}

int
send_ttl_rights(int sd, int sendfd)
{

    int clen;
    struct msghdr	msg;
    struct iovec	iov[1];
    struct cmsghdr *cmptr;
    char control[CMSG_SPACE(sizeof(int))];
    char tdata[] = "test";


    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = IPPROTO_IP;
    cmptr->cmsg_type = IP_TTL;
    *((int *) CMSG_DATA(cmptr)) = sendfd;


    msg.msg_name = NULL;
    msg.msg_namelen = 0;

    iov[0].iov_base = tdata;
    iov[0].iov_len = strlen(tdata);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;


    fprintf(stderr, "Parent:sending TTL\n");
    if ((clen = sendmsg(sd, &msg, 0)) == -1) {
        perror("sendmsg:send_access_rights");
        return -1;
    }

    fprintf(stderr, "Parent:sent %d bytes\n", clen);
    return 0;
}

int
get_send_sock(void)
{
    int sd;
    struct addrinfo	hints, *res, *ressave;
    char port[] = TESTPORT;
    char host[] = HOST;
    char test_data[] = "Start from the server, the source\n";

	bzero(&hints, sizeof(struct addrinfo));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

	if (getaddrinfo(host, port, &hints, &res) != 0) {
        perror("getaddrinfo:get_send_sock");
        return -1;
    }

    ressave = res;

	do {
		if ((sd = socket(res->ai_family, res->ai_socktype, res->ai_protocol)) == -1) {
            continue;
        }

		if (connect(sd, res->ai_addr, res->ai_addrlen) == 0)
			break;		/* success */

		close(sd);	/* ignore this one */
	} while ((res = res->ai_next) != NULL);

	if (res == NULL) {
        fprintf(stderr, "Connect error: get_send_sock\n");
        freeaddrinfo(ressave);
        return -1;
    }

	freeaddrinfo(ressave);

    if (send(sd, test_data, strlen(test_data), 0) == -1) {
        perror("send");
        return -1;
    }

	return(sd);
}

int
get_send_fd()
{
    int fd;
    char test_data[] = "Start from the server, the source\n";

    if ((fd = open(TESTFILE, O_CREAT|O_WRONLY|O_APPEND|O_CLOEXEC, 0666)) == -1) {
        perror("open:get_send_fd");
        return -1;
    }

    if (write(fd, test_data, strlen(test_data)) == -1) {
        perror("write:get_send_fd");
        return -1;
    }
    return fd;
}

void *
recv_thread(void *param)
{
    int sd, rsd, optval;
    unsigned short port = TESTPORTNO;
    struct sockaddr_in serveraddr;
    struct sockaddr_in clientaddr;
    socklen_t clientlen = sizeof(clientaddr);

    sd = socket(AF_INET, SOCK_STREAM, 0);
    if (sd < 0) {
        perror("ERROR:recv_thread:opening socket");
        exit(-1);
    }

    optval = 1;
    setsockopt(sd, SOL_SOCKET, SO_REUSEADDR,
               (const void *)&optval , sizeof(int));

    bzero((char *)&serveraddr, sizeof(serveraddr));

    serveraddr.sin_family = AF_INET;

    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);

    serveraddr.sin_port = htons(port);

    if (bind(sd, (struct sockaddr *)&serveraddr,
             sizeof(serveraddr)) < 0) {
        perror("ERROR:recv_thread:bind");
        exit(-1);
    }

    if (listen(sd, 5) < 0) { /* allow 5 requests to queue up */
        perror("ERROR:recv_thread:listen");
        exit(-1);
    }

    while (1) {
        int rc;
        char buf[64];

        rsd = accept(sd, (struct sockaddr *)&clientaddr, &clientlen);
        if (rsd < 0) {
            perror("ERROR:recv_thread:accept");
            continue;
        }

        if ((rc = recv(rsd, buf, 64, 0)) <= 0) {
            perror("ERROR:recv_thread:recv");
            continue;
        }

        // echo
        write(1, buf, rc);
        close(rsd);
    }

    return NULL;
}

int
unruly_kid()
{
  int clientsock;
  int *recvfd;
  int i, rc;
  int numfds;
  struct cmsghdr *cmptr;
  struct iovec iov[1];
  struct sockaddr_un clientaddr;
  struct msghdr	msg;
  struct stat sbuf;
  char control[CMSG_SPACE(sizeof(int))];
  char test_data[] = "Passing the buck\n";

  // socket: create the parent socket
  clientsock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (clientsock < 0) {
      perror("ERROR opening child socket");
      exit(-1);
  }

  bzero((char *)&clientaddr, sizeof(clientaddr));

  clientaddr.sun_family = AF_UNIX;

  // pfile is the socket path
  strncpy(clientaddr.sun_path, pfile, sizeof(clientaddr.sun_path)-1);

  if (connect(clientsock, (const struct sockaddr *)&clientaddr,
              sizeof(struct sockaddr_un)) == -1) {
      perror("connect child");
      exit(-1);
  }

  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  msg.msg_name = NULL;
  msg.msg_namelen = 0;

  iov[0].iov_base = NULL;
  iov[0].iov_len = 0;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  if ((rc = recvmsg(clientsock, &msg, 0)) == -1) {
      perror("recvmsg child");
      exit(-1);
  }

  for (cmptr = CMSG_FIRSTHDR(&msg); cmptr != NULL; cmptr = CMSG_NXTHDR(&msg, cmptr)) {
      if  (cmptr->cmsg_len >= CMSG_LEN(sizeof(int))) {

          if (cmptr->cmsg_level != SOL_SOCKET) {
              fprintf(stderr, "Child:control level != SOL_SOCKET\n");
              exit(-1);
          }

          if (cmptr->cmsg_type != SCM_RIGHTS) {
              fprintf(stderr, "Child:control type != SCM_RIGHTS\n");
              exit(-1);
          }

          recvfd = ((int *) CMSG_DATA(cmptr));
          fprintf(stdout, "Child:Received len %ld\n", cmptr->cmsg_len);
          numfds = (cmptr->cmsg_len - CMSG_ALIGN(sizeof(struct cmsghdr))) / sizeof(int);
          fprintf(stdout, "Child:Received %d fds %ld\n", numfds, sizeof(int));

          for (i = 0; i < numfds; i++) {
              // on the new fd
              if (fstat(recvfd[i], &sbuf) != -1) {
                  if ((sbuf.st_mode & S_IFMT) == S_IFSOCK) {
                      if (send(recvfd[i], test_data, strlen(test_data), 0) == -1) {
                          perror("send child");
                          continue;
                      }
                  } else {
                      if (write(recvfd[i], test_data, strlen(test_data)) == -1) {
                          perror("write child");
                          continue;
                      }
                  }
              } else {
                  perror("fstat child");
                  exit(-1);
              }

              close(recvfd[i]);
          }
      } else {
          fprintf(stderr, "Child:Did not receive access rights\n");
      }
  }
  exit(0);
}

int
main(int argc, char **argv) {
  pid_t child;
  int servsock, child_stat;
  int optval, opt;
  int ttype = 0;
  int verbose = 0;
  pthread_t testTID;
  struct sockaddr_un serveraddr;

  if (argc < 2) {
      usage(argv[0]);
  }

  while ((opt = getopt(argc, argv, "vhf:1234")) > 0) {
    switch (opt) {
      case 'v': verbose++; break;
      case 'f': pfile=strdup(optarg); break;
      case '1': ttype=1; break;
      case '2': ttype=2; break;
      case '3': ttype=3; break;
      case '4': ttype=4; break;
      case 'h': default: usage(argv[0]); break;
    }
  }

  // socket: create the parent socket
  servsock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (servsock < 0) {
      perror("ERROR opening socket");
      exit(1);
  }

  /* setsockopt: Handy debugging trick that lets
   * us rerun the server immediately after we kill it;
   * otherwise we have to wait about 20 secs.
   * Eliminates "ERROR on binding: Address already in use" error.
   */
  optval = 1;
  setsockopt(servsock, SOL_SOCKET, SO_REUSEADDR,
             (const void *)&optval , sizeof(int));

  bzero((char *)&serveraddr, sizeof(serveraddr));

  serveraddr.sun_family = AF_UNIX;

  // pfile is the socket path
  strncpy(serveraddr.sun_path, pfile, sizeof(serveraddr.sun_path)-1);
  if (unlink(pfile) == -1) {
      fprintf(stderr, "pfile: %s\n", pfile);
      perror("unlink");
  }

  if (bind(servsock, (struct sockaddr *) &serveraddr,
           sizeof(serveraddr)) < 0) {
      perror("ERROR on binding");
      exit(1);
  }

  if (listen(servsock, 15) < 0) { /* allow 15 requests to queue up */
      perror("ERROR on listen");
      exit(1);
  }

  if (chmod(pfile, 0777) == -1) {
      perror("chmod");
      exit(-1);
  }

  // start a thread to receive test messages
  if (pthread_create(&testTID, NULL, recv_thread, NULL) != 0) {
      perror("pthread_create");
      exit(-1);
  }

  struct timespec ts = {.tv_sec=0, .tv_nsec=010000000}; // 10 ms
   nanosleep(&ts, NULL);

  // start the child that receives the access rights
  if ((child = fork()) == 0) {
      //We are the child proc
      unruly_kid();
  }

  // loop on this until we are told to quit
  // do we want or need to loop?
  while(1) {
      int sd, sendfd, sendsock;
      struct sockaddr caddr;
      socklen_t clen = sizeof(struct sockaddr);
      int sendfds[2];

      // wait for the child to tell us it's ready
      fprintf(stderr, "Parent:accepting\n");
      sd = accept(servsock, (struct sockaddr *)&caddr, &clen);
      if (sd < 0) {
          perror("ERROR on accept");
          continue;
      }

      // get the send fd
      if ((sendfd = get_send_fd()) == -1) exit(-1);

      // get a socket to send
      if ((sendsock = get_send_sock()) == -1) exit(-1);

      // send fd access rights
      switch (ttype) {
      case 1:
          if (send_access_rights(sd, sendfd) == -1) exit(-1);
          break;
      case 2:
          if (send_access_rights(sd, sendsock) == -1) exit(-1);
          break;
      case 3:
          sendfds[0] = sendfd;
          sendfds[1] = sendsock;
          if (send_array_rights(sd, sendfds, 2) == -1) exit(-1);
          break;
      case 4:
          if (send_ttl_rights(sd, sendsock) == -1) exit(-1);
          break;
      default:
          fprintf(stderr, "Parent: no test type\n");
          exit(-1);
      }

      close(sd);
      close(sendfd);
      close(sendsock);
      break;
  }

  waitpid(child, &child_stat, 0);

  switch (ttype) {
      char *verify[2];
  case 1:
      verify[0] = "fs.op.open";
      verify[1] = "recvmsg";
      verify[2] = "Received_File_Descriptor";
      if (check_event(verify, 3) == -1) {
          fprintf(stderr, "Parent:ERROR:no event for write to a file\n");
          exit(-1);
      }
      break;
      case 2:
          verify[0] = "net.tx";
          verify[1] = "remotep";
          verify[2] = "9009";
          if (check_event(verify, 3) == -1) {
              fprintf(stderr, "Parent:ERROR:no event for send on socket\n");
              exit(-1);
          }
          break;
      case 3:
          verify[0] = "fs.op.open";
          verify[1] = "recvmsg";
          verify[2] = "Received_File_Descriptor";
          verify[3] = "net.tx";
          verify[4] = "remotep";
          verify[5] = "9009";
          if (check_event(verify, 6) == -1) {
              fprintf(stderr, "Parent:ERROR:no event for write an array of rights\n");
              exit(-1);
          }
          break;
      case 4:
          verify[0] = "net.tx";
          if (check_event(verify, 1) != -1) {
              fprintf(stderr, "Parent:ERROR:should not have an event for sending TTL\n");
              exit(-1);
          }
          break;
      default:
          fprintf(stderr, "Parent: no test type\n");
          exit(-1);
      }

  return WEXITSTATUS(child_stat);
}
