// gcc -g test/manual/dns_response_parse.c -lresolv -o ~/scope/utils/dp

#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <resolv.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// example arg for payload file: "/tmp/77698_127.0.0.53:53:0.in"
// quotes are needed on the command line

struct response {
    HEADER hdr;
    u_char buf[NS_PACKETSZ]; /* defined in arpa/nameser.h */
};

int
main(int argc, char **argv)
{
    int fd, rc, i, nmsg;
    ns_rr rr;
    ns_msg handle;
    struct response resp;
    char buf[1024];

    if (argc < 2) {
        printf("Usage: %s <path-to-a-DNS-payload-file>\n", argv[0]);
        exit(1);
    }

    printf("Using %s\n", argv[1]);
    if ((fd = open(argv[1], O_RDONLY)) < 0) {
        perror("open");
        exit(1);
    }

    if ((rc = read(fd, &resp, sizeof(struct response))) < 0) {
        perror("read");
        exit(1);
    }

    close(fd);

    // init ns lib
    ns_initparse((char *)&resp, rc, &handle);

    nmsg = ns_msg_count(handle, ns_s_an);

    // error?
    if ((rc = ns_msg_getflag(handle, ns_f_rcode)) != ns_r_noerror) {
        fprintf(stderr, "response error: %d\n", rc);
        exit(1);
    }

    // authoritative response?
    if (!ns_msg_getflag(handle, ns_f_aa)) {
        fprintf(stderr, "not an authoratative answer\n");
    }

    for (i = 0; i < nmsg; i++) {
        char dispbuf[4096];
        ns_parserr(&handle, ns_s_an, i, &rr);
        ns_sprintrr(&handle, &rr, NULL, NULL, dispbuf, sizeof(dispbuf));
        printf("%s\n", dispbuf);
        inet_ntop(AF_INET, (struct sockaddr_in *)rr.rdata, dispbuf, sizeof(dispbuf));
        printf("resolved addr is %s\n", dispbuf);
    }

    printf("Success\n");
}
