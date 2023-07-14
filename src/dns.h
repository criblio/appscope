#ifndef _DNS_H_
#define _DNS_H_

#include <arpa/nameser.h>
#include <resolv.h>
#include <arpa/inet.h>

#define DNS_SERVICE 0x7f000035 // 127.0.0.53
#define DNS_PORT 53
#define DNS_MAXLABEL 63  // Maximum size limit of DNS label
#define QTYPE_QUERY 0x01
#define QCLASS_IN 0x01

// DNS full header definition per RFC 1035
/*
      0  1  2  3  4  5  6  7  8  9  0  1  2  3  4  5
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                      ID                       |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |QR|   Opcode  |AA|TC|RD|RA|   Z    |   RCODE   |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    QDCOUNT                    |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    ANCOUNT                    |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    NSCOUNT                    |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    ARCOUNT                    |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

Question Field
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    QNAME                      |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    QTYPE                      |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |                    QCLASS                     |
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

 */

struct dns_header
{
    unsigned short id; // identification number
    unsigned char qr :1;     // query/response flag
    unsigned char opcode :4; // purpose of message
    unsigned char aa :1;     // authoritative answer
    unsigned char tc :1;     // truncated message
    unsigned char rd :1;     // recursion desired

    unsigned char ra :1;     // recursion available
    unsigned char z :1;      // its z! reserved
    unsigned char rcode :4;  // response code
    unsigned char cd :1;     // checking disabled
    unsigned char ad :1;     // authenticated data
 
    unsigned short qdcount;  // number of question entries
    unsigned short ancount;  // number of answer entries
    unsigned short nscount;  // number of authority entries
    unsigned short arcount;  // number of resource entries
};

typedef enum {
    DNS_OPCODE_QUERY = 0,
    DNS_OPCODE_IQUERY = 1,
    DNS_OPCODE_STATUS = 2,
} dns_op_code_t;

typedef enum {
    DNS_QR_QUERY = 0,
    DNS_QR_RESP  = 1,
} dns_qr_t;


// DNS query definition
typedef struct question_t
{
    unsigned short qtype;
    unsigned short qclass;
} question;

// A DNS Query
struct query
{
    unsigned char *name;
    struct question *question;
};

typedef struct dns_query_t {
    struct dns_header qhead;
    unsigned char name[];
} dns_query;

#ifdef __linux__
struct response {
    HEADER hdr;
    u_char buf[NS_PACKETSZ];      /* defined in arpa/nameser.h */
};
#endif // __linux__
#endif // _DNS_H_
