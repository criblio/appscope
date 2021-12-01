/*
 * unixpeer.c
 *
 * gcc -g -Wall test/manual/unixpeer.c -lpthread -o test/linux/unixpeer
 *
 * Create a UNIX socket between 2 procs; client & server
 * Send data over the socket
 * Test that events describe the connection
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define EVENTFILE "/tmp/scope_peer.log"
#define SEMNAME   "/sync"

char *pfile;
int verbose = 0;

void
usage(char *prog)
{
    fprintf(stderr, "usage: %s [-v] -f pipe/file\n", prog);
    exit(-1);
}

ino_t
get_node(char *events, char *start, char *tag)
{
    ino_t node = (ino_t)-1;
    char *thisval, *value, *endval;

    // look for node numbers within the same event line
    if (((thisval = strstr(events, start)) != NULL) && ((thisval = strstr(thisval, tag)) != NULL) && ((value = index(thisval, ':')) != NULL) && ((endval = index(value, ',')) != NULL)) {
        *endval = '\0';
        node = (ino_t)atol(value + 1);
        *endval = ',';
    }

    return node;
}

char *
get_events()
{
    int fd, rc;
    char *buf;
    struct stat sbuf;

    if ((fd = open(EVENTFILE, O_RDONLY | O_CREAT, 0666)) == -1) {
        perror("open:check_event");
        return NULL;
    }

    if (fstat(fd, &sbuf) == -1) {
        perror("fstat:check_event");
        return NULL;
    }

    if ((buf = calloc(1, sbuf.st_size)) == NULL) {
        perror("calloc:check_event");
        return NULL;
    }

    rc = read(fd, buf, sbuf.st_size);
    if (rc <= 0) {
        perror("read:check_event");
        return NULL;
    }

    if (close(fd) == -1) {
        perror("close:check_event");
        return NULL;
    }

    return buf;
}

int
server_child()
{
    int parentfd;
    int childfd;
    unsigned int clientlen;
    int optval;
    int rc;
    sem_t *sem;
    struct sockaddr_un serveraddr;
    struct sockaddr_un clientaddr;
    char buf[128];

    if ((sem = sem_open(SEMNAME, O_RDWR)) == SEM_FAILED) {
        perror("Server:sem_open");
        exit(-1);
    }

    // socket: create the parent socket
    parentfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (parentfd < 0) {
        perror("ERROR opening socket");
        exit(1);
    }

    optval = 1;
    setsockopt(parentfd, SOL_SOCKET, SO_REUSEADDR, (const void *)&optval, sizeof(int));

    bzero((char *)&serveraddr, sizeof(serveraddr));

    serveraddr.sun_family = AF_UNIX;

    // pfile is the socket path
    strncpy(serveraddr.sun_path, pfile, sizeof(serveraddr.sun_path) - 1);
    if (unlink(pfile) == -1) {
        perror("unlink");
    }

    if (bind(parentfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) < 0) {
        perror("ERROR on binding");
        exit(1);
    }

    if (listen(parentfd, 15) < 0) { /* allow 15 requests to queue up */
        perror("ERROR on listen");
        exit(1);
    }

    if (chmod(pfile, 0777) == -1) {
        perror("chmod");
        exit(-1);
    }

    // wait for a connection request then echo
    fprintf(stdout, "Parent accepting\n");
    clientlen = sizeof(clientaddr);

    // Tell the client ready to connect
    if (sem_post(sem) == -1) {
        perror("Server:sem_post1");
        exit(-1);
    }

    childfd = accept(parentfd, (struct sockaddr *)&clientaddr, &clientlen);
    if (childfd < 0) {
        perror("ERROR on accept");
        exit(-1);
    }

    // Tell the client we are ready to get data
    if (sem_post(sem) == -1) {
        perror("Server:sem_post2");
        exit(-1);
    }

    do {
        rc = read(childfd, buf, sizeof(buf));
        if (rc < 0)
            perror("read");
        if (rc > 0) {
            if (verbose > 0)
                fprintf(stdout, "Parent received %d bytes on %d\n", rc, childfd);
        }

        if (rc == 0) {
            close(childfd);
            fprintf(stdout, "Child Closed the connection\n");
            break;
        }
        // echo input to stdout
        if (verbose > 0)
            write(1, buf, rc);

        // Tell the client we got the data
        if (sem_post(sem) == -1) {
            perror("Server:sem_post3");
            exit(-1);
        }

    } while (1);

    sem_close(sem);
    exit(0);
}

int
client_child()
{
    int clientsock;
    sem_t *sem;
    struct sockaddr_un clientaddr;
    char test_data[] = "Passing the buck\n";

    if ((sem = sem_open(SEMNAME, O_RDWR)) == SEM_FAILED) {
        perror("Client:sem_open");
        exit(-1);
    }

    if (verbose > 0)
        fprintf(stdout, "Child starting\n");

    // socket: create the parent socket
    clientsock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (clientsock < 0) {
        perror("ERROR opening child socket");
        exit(-1);
    }

    bzero((char *)&clientaddr, sizeof(clientaddr));

    clientaddr.sun_family = AF_UNIX;

    // pfile is the socket path
    strncpy(clientaddr.sun_path, pfile, sizeof(clientaddr.sun_path) - 1);

    // Is server ready to connect
    if (sem_wait(sem) == -1) {
        perror("Client:sem_wait1");
        exit(-1);
    }

    if (connect(clientsock, (const struct sockaddr *)&clientaddr, sizeof(struct sockaddr_un)) == -1) {
        perror("connect child");
        exit(-1);
    }

    // Wait for the server to tell us it's ready
    if (sem_wait(sem) == -1) {
        perror("Client:sem_wait2");
        exit(-1);
    }

    if (verbose > 0)
        fprintf(stdout, "Child sending: %s\n", test_data);

    if (write(clientsock, test_data, strlen(test_data)) == -1) {
        perror("write");
        exit(-1);
    }

    // Let the server report on this end before we close
    if (sem_wait(sem) == -1) {
        perror("Client:sem_wait3");
        exit(-1);
    }

    fprintf(stdout, "Child closing and exiting\n");
    close(clientsock);
    sem_close(sem);
    exit(0);
}

int
main(int argc, char **argv)
{
    int rc, opt, client_stat, server_stat;
    pid_t client, server;
    ino_t tx_lnode, tx_rnode, rx_lnode, rx_rnode;
    char *events;
    sem_t *sem;

    if (argc < 2) {
        usage(argv[0]);
    }

    while ((opt = getopt(argc, argv, "vhf:")) > 0) {
        switch (opt) {
            case 'v':
                verbose++;
                break;
            case 'f':
                pfile = strdup(optarg);
                break;
            case 'h':
            default:
                usage(argv[0]);
                break;
        }
    }

    if ((sem = sem_open(SEMNAME, O_CREAT | O_RDWR, 0666, 0)) == SEM_FAILED) {
        perror("Parent:sem_open");
        exit(-1);
    }

    if (sem_init(sem, 1, 0) == -1) {
        perror("Parent:sem_init");
        exit(-1);
    }

    // Note: we use procs so that events are flushed reliably on proc exit
    // start the server child
    if ((server = fork()) == 0) {
        // We are the child proc
        server_child();
    }

    // start the client child
    if ((client = fork()) == 0) {
        // We are the child proc
        client_child();
    }

    waitpid(client, &client_stat, 0);
    waitpid(server, &server_stat, 0);

    rc = 0;
    if ((events = get_events()) == NULL)
        exit(-1);

    if ((tx_lnode = get_node(events, "\"net.tx\"", "\"localn\"")) == -1)
        rc = -1;
    if ((tx_rnode = get_node(events, "\"net.tx\"", "\"remoten\"")) == -1)
        rc = -1;
    if ((rx_lnode = get_node(events, "\"net.rx\"", "\"localn\"")) == -1)
        rc = -1;
    if ((rx_rnode = get_node(events, "\"net.rx\"", "\"remoten\"")) == -1)
        rc = -1;

    if ((rc == 0) && (tx_lnode == rx_rnode) && (tx_rnode == rx_lnode)) {
        fprintf(stdout, "Nodes match tx_%ld:%ld rx_%ld:%ld\n", tx_lnode, tx_rnode, rx_lnode, rx_rnode);
    } else {
        rc = -1;
        fprintf(stderr, "ERROR: nodes do not match tx_%ld:%ld rx_%ld:%ld\n", tx_lnode, tx_rnode, rx_lnode, rx_rnode);
    }

    if (events)
        free(events);

    if (unlink(EVENTFILE) == -1) {
        perror("unlink:check_event");
    }

    sem_close(sem);

    if ((rc == 0) && (WEXITSTATUS(client_stat) == 0) && (WEXITSTATUS(server_stat) == 0)) {
        exit(0);
    } else {
        exit(-1);
    }
}
