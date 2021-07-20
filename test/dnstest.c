#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#include "dns.h"
#include "test.h"

/*
  This is just reference. Packet details for a correct
  DNS Query and a response from the name server.

DNS Query
13:26:17.927799 IP 127.0.0.1.46129 > 127.0.0.53.53: 57007+ [1au] A? www.google.com. (76)
	0x0000:  4500 0068 b8cf 4000 4011 837f 7f00 0001  E..h..@.@.......
	0x0010:  7f00 0035 b431 0035 0054 fe9b deaf 0100  ...5.1.5.T......
                                           (ID)
                                                QR=0 Opcode=0 (query)
                                                AA=0 TC=0 RD=1 
	0x0020:  0001 0000 0000 0001 0377 7777 0667 6f6f  .........www.goo
             qdcount=1
                  ancount=0
                       nscount=0
                            arcount=1 
                                 03 = octet len of 'www'
                                           06 len of 'google'
	0x0030:  676c 6503 636f 6d00 0001 0001 0000 0000  gle.com.........
                    03 len of 'com'  
                              00 string is terminated
                                 0001 is qtype of query
                                      0001 is qclass of INT (internet)  
	0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
	0x0050:  0000 0000 0000 0000 0000 0000 0000 0000  ................
	0x0060:  0000 0000 0000 0000                      ........
Response from the name server
13:26:17.945488 IP 127.0.0.53.53 > 127.0.0.1.46129: 57007 1/0/0 A 172.217.6.4 (48)
	0x0000:  4500 004c cb52 4000 4011 7118 7f00 0035  E..L.R@.@.q....5
	0x0010:  7f00 0001 0035 b431 0038 fe7f deaf 8180  .....5.1.8......
	0x0020:  0001 0001 0000 0000 0377 7777 0667 6f6f  .........www.goo
	0x0030:  676c 6503 636f 6d00 0001 0001 c00c 0001  gle.com.........
	0x0040:  0001 0000 0005 0004 acd9 0604            ............
 */

#define DNS_ID 0xdeaf
#define DOMAIN "www.google.com"
#define QNAME "\003www\006google\003com"
#define MAXLINES 10000
#define HB3QUERY 01
#define HB4QUERY 00
#define QTYPE_AAAA 0x001c
#define OPCODE_INVERSE_QUERY 0x08

typedef struct header_t {
  unsigned short id;
  unsigned char  hb3,hb4;
  unsigned short qdcount,ancount,nscount,arcount;
} header;

typedef struct query_t {
    header qhead;
    unsigned char name[64];
} query;

static void
truncFile()
{
   FILE *file = fopen("/tmp/dnstest.log", "w");
   fclose(file);
}

static int
createSocket(struct sockaddr_in *sa)
{
    int fd;

    if (!sa) return -1;

    fd = socket(AF_INET, SOCK_DGRAM, 0);

    bzero((char *)sa, sizeof(struct sockaddr_in));
    sa->sin_family = AF_INET;
    sa->sin_addr.s_addr = htonl(DNS_SERVICE);
    sa->sin_port = htons((unsigned short)DNS_PORT);

    return fd;
}

static int
CreateQuery(query *pkt)
{
    question *question;
    
    bzero((char *)&pkt->qhead, sizeof(struct dns_header));
    pkt->qhead.id = (unsigned short)htons(DNS_ID);
    pkt->qhead.ancount = htons(0);
    pkt->qhead.nscount = htons(0);
    pkt->qhead.arcount = htons(1);
    pkt->qhead.qdcount = htons(1);
    pkt->qhead.hb4 = HB4QUERY;
    pkt->qhead.hb3 = HB3QUERY;
    strncpy((char *)pkt->name, QNAME, sizeof(pkt->name));

    question = (struct question_t *)(pkt->name + strlen((char *)pkt->name) + 1);
    question->qtype = htons(QTYPE_QUERY);
    question->qclass = htons(QCLASS_IN);

    return 0;
}

static int
sendToNS(int fd, char *packet, size_t len, struct sockaddr_in *dst, int recv)
{
    struct msghdr msg;
    struct iovec iov[1]; 

    iov[0].iov_base = packet;
    iov[0].iov_len = len;

    msg.msg_control = NULL;
    msg.msg_controllen = 0;
    msg.msg_flags = 0;
    msg.msg_name = dst;
    msg.msg_namelen = sizeof(struct sockaddr_in);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
  
    if (sendmsg(fd, &msg, 0) <= 0) return -1;

    if (recv) {
        return recvmsg(fd, &msg, 0);
    } else {
        return 0;
    }
}

static int
checkMetric(char *domain)
{
    int rc = -1, loopcnt = 0;
    char *line = NULL;
    size_t len = 0;
    struct stat sbuf = {0};

    FILE *fp = fopen("/tmp/dnstest.log", "r");
    if (fp == NULL) {
        printf("%s:%d can't open metric file\n", __FUNCTION__, __LINE__);
        return -1;
    }

    while (sbuf.st_size == 0) {
        sleep(1);
        fstat(fileno(fp), &sbuf);
    }

    while ((getline(&line, &len, fp) != -1) && (loopcnt < MAXLINES)) {
        if (strstr(line, "net.dns") != NULL) {
            char *key, *value, *end;
                if ((key = strstr(line, "domain:")) != NULL) {
                if ((value = index(key, ':')) == NULL) break;
                value += 1;
                if ((end = index(value, ',')) == NULL) break;
                *end = '\0';
                //printf("%s:%d %s\n", __FUNCTION__, __LINE__, value);
                if (!domain || (strncmp(value, domain, strlen(value)) == 0)) rc = 0;
                break;
            }
        }

        /* 
         * In the case where we don't expect data getline will
         * loop forever reading lines because it is creating metrics.
         * If we don't see a net.dns line in the first MAXLINES it's an
         * error.
         */
        loopcnt++;
        if (line) free(line);
        line = NULL;
    }

    if (line) free(line);
    fclose(fp);
    return rc;
}

// Can we send a DNS query and get a response from the local NS
static void
dnsReply(void** state)
{
    struct sockaddr_in sa;
    query pkt;

    int fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    assert_int_equal(0, CreateQuery(&pkt));
    assert_int_not_equal(-1, sendToNS(fd, (char *)&pkt, sizeof(pkt), &sa, 0));
    truncFile();
    close(fd);
}

// Can we send a DNS query and extract the name
static void
dnsName(void** state)
{
    struct sockaddr_in sa;
    query pkt;
    question *question;

    int fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    // Do an A query
    assert_int_equal(0, CreateQuery(&pkt));
    assert_int_not_equal(-1, sendToNS(fd, (char *)&pkt, sizeof(pkt), &sa, 0));
    assert_int_equal(0, checkMetric(DOMAIN));
    truncFile();
    close(fd);

    // Do a AAAA query
    fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    question = (struct question_t *)(pkt.name + strlen((char *)pkt.name) + 1);
    question->qtype = htons(QTYPE_AAAA);
    assert_int_not_equal(-1, sendToNS(fd, (char *)&pkt, sizeof(pkt), &sa, 0));
    assert_int_equal(0, checkMetric(DOMAIN));
    truncFile();
    close(fd);
}

static void
getDNSNameBadRequest(void** state)
{
    // This test was created after a crash of firefox was captured with
    // this specific packet data.  The signature was stdout or stderr
    // output of "stack smashing detected".  In response, we created this
    // test and added three kinds of checks to getDNSName that collectively
    // ensure that we won't try to create dns output from packets like this.
    // The checks are that: label length is 63 or less, we don't read past
    // the end of the packet, and that the characters that make up the domain
    // are legal.  Before these checks this packet would output something like:
    // net.dns:1|c|#proc:dnstest,pid:62427,host:ubuntu18_4,domain:<98>à^ZI<88>Ì®^Ý.!^M«=ÖDWd\^H<91>zÒ^};<96>,`Îq<96>^HÀ) o[÷ÿ^?,duration:0,unit:request

    struct sockaddr_in sa;
    int fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    uint8_t pkt[] = {
        0x17, 0x03, 0x03, 0x00, 0x29, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x09, 0x98, 0xe0, 0x1a,
        0x49, 0x88, 0xcc, 0xae, 0x5e, 0xdd, 0x57, 0x21,
        0x0d, 0xab, 0x3d, 0xd6, 0x44, 0x57, 0x64, 0x5c,
        0x08, 0x91, 0x7a, 0xd2, 0x5e, 0x7d, 0x3b, 0x96,
        0x2c, 0x60, 0xce, 0x71, 0x96, 0x08, 0xc0, 0x29,
    };
    assert_int_not_equal(-1, sendToNS(fd, (char*)&pkt, 46, &sa, 0));
    assert_int_equal(-1, checkMetric(NULL));
    truncFile();
    close(fd);
}

static void
getDNSNameMaxDomainLen(void** state)
{
    struct sockaddr_in sa;
    int fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    uint8_t pkt[] = {
        0x17, 0x03, 0x03, 0x00, 0x29, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0f, 0x61, 0x61, 0x61,
        0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61, 0x61,
        0x61, 0x61, 0x61, 0x61, 0x0f, 0x62, 0x62, 0x62,
        0x62, 0x62, 0x62, 0x62, 0x62, 0x62, 0x62, 0x62,
        0x62, 0x62, 0x62, 0x62, 0x0f, 0x63, 0x63, 0x63,
        0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
        0x63, 0x63, 0x63, 0x63, 0x0f, 0x64, 0x64, 0x64,
        0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64,
        0x64, 0x64, 0x64, 0x64, 0x0f, 0x65, 0x65, 0x65,
        0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
        0x65, 0x65, 0x65, 0x65, 0x0f, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x0f, 0x67, 0x67, 0x67,
        0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67, 0x67,
        0x67, 0x67, 0x67, 0x67, 0x0f, 0x68, 0x68, 0x68,
        0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68, 0x68,
        0x68, 0x68, 0x68, 0x68, 0x0f, 0x69, 0x69, 0x69,
        0x69, 0x69, 0x69, 0x69, 0x69, 0x69, 0x69, 0x69,
        0x69, 0x69, 0x69, 0x69, 0x0f, 0x6a, 0x6a, 0x6a,
        0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a, 0x6a,
        0x6a, 0x6a, 0x6a, 0x6a, 0x0f, 0x6b, 0x6b, 0x6b,
        0x6b, 0x6b, 0x6b, 0x6b, 0x6b, 0x6b, 0x6b, 0x6b,
        0x6b, 0x6b, 0x6b, 0x6b, 0x0f, 0x6c, 0x6c, 0x6c,
        0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c, 0x6c,
        0x6c, 0x6c, 0x6c, 0x6c, 0x0f, 0x6d, 0x6d, 0x6d,
        0x6d, 0x6d, 0x6d, 0x6d, 0x6d, 0x6d, 0x6d, 0x6d,
        0x6d, 0x6d, 0x6d, 0x6d, 0x0f, 0x6e, 0x6e, 0x6e,
        0x6e, 0x6e, 0x6e, 0x6e, 0x6e, 0x6e, 0x6e, 0x6e,
        0x6e, 0x6e, 0x6e, 0x6e, 0x0f, 0x6f, 0x6f, 0x6f,
        0x6f, 0x6f, 0x6f, 0x6f, 0x6f, 0x6f, 0x6f, 0x6f,
        0x6f, 0x6f, 0x6f, 0x6f, 0x0f, 0x70, 0x70, 0x70,
        0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70,
        0x70, 0x70, 0x70, 0x70, 0x00
    };
    assert_int_not_equal(-1, sendToNS(fd, (char*)&pkt, sizeof(pkt), &sa, 0));
    assert_int_not_equal(-1, checkMetric(
       "aaaaaaaaaaaaaaa.bbbbbbbbbbbbbbb.ccccccccccccccc.ddddddddddddddd."
       "eeeeeeeeeeeeeee.fffffffffffffff.ggggggggggggggg.hhhhhhhhhhhhhhh."
       "iiiiiiiiiiiiiii.jjjjjjjjjjjjjjj.kkkkkkkkkkkkkkk.lllllllllllllll."
       "mmmmmmmmmmmmmmm.nnnnnnnnnnnnnnn.ooooooooooooooo.ppppppppppppppp"));
    truncFile();
    close(fd);
}
// Do we not get a domain from a DNS packet that is not a query
static void
dnsNoName(void** state)
{
    struct sockaddr_in sa;
    query pkt;

    int fd = createSocket(&sa);
    assert_int_not_equal(-1, fd);
    assert_int_equal(0, CreateQuery(&pkt));

    // modify the packet such that it's not a query
    // Define an inverse query
    pkt.qhead.hb3 = OPCODE_INVERSE_QUERY;
    assert_int_not_equal(-1, sendToNS(fd, (char *)&pkt, sizeof(pkt), &sa, 0));
    assert_int_equal(-1, checkMetric(DOMAIN));
    truncFile();
    close(fd);
}

// Can we extract the domain resulting from gethostbyname
static void
dnsGethostbyname(void** state)
{
    assert_non_null(gethostbyname(DOMAIN));
    assert_int_equal(0, checkMetric(DOMAIN));
    truncFile();
}

// Can we extract the domain resulting from gethostbyname2
static void
dnsGethostbyname2(void** state)
{
    assert_non_null(gethostbyname2(DOMAIN, AF_INET));
    assert_int_equal(0, checkMetric(DOMAIN));
    truncFile();
}

// Can we extract the domain resulting from getaddrinfo
static void
dnsGetaddrinfo(void** state)
{
#if defined(__aarch64__)
    // TODO: restore after we re-enable hooking getaddrinfo() on ARM
    skip();
#endif

    struct addrinfo hints = {0};
    struct addrinfo *addrs = NULL;

    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    
    assert_int_equal(0, getaddrinfo(DOMAIN, "8080", &hints, &addrs));
    assert_int_equal(0, checkMetric(DOMAIN));
    freeaddrinfo(addrs);
    truncFile();
}

int
main(int argc, char* argv[])
{
    printf("running %s\n", argv[0]);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(dnsReply),
        cmocka_unit_test(dnsName),
        cmocka_unit_test(getDNSNameBadRequest),
        cmocka_unit_test(getDNSNameMaxDomainLen),
        cmocka_unit_test(dnsNoName),
        cmocka_unit_test(dnsGethostbyname),
        cmocka_unit_test(dnsGethostbyname2),
        cmocka_unit_test(dnsGetaddrinfo),
    };
    return cmocka_run_group_tests(tests, groupSetup, groupTeardown);
}
