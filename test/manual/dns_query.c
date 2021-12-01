// gcc -g test/manual/dns_query.c -lresolv -o ~/scope/utils/dnsq

#include <arpa/nameser.h>
#include <netinet/in.h>
#include <resolv.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int
main(int argc, char *argv[])
{
    u_char nsbuf[4096];
    char dispbuf[4096];
    ns_msg msg;
    ns_rr rr;
    int i, j, l;

    if (argc < 2) {
        printf("Usage: %s <domain>[...]\n", argv[0]);
        exit(1);
    }

    for (i = 1; i < argc; i++) {
        // IPv4
        l = res_query(argv[i], ns_c_in, ns_t_a, nsbuf, sizeof(nsbuf));
        if (l < 0) {
            perror(argv[i]);
        } else {
            /* MX answer info */
            ns_initparse(nsbuf, l, &msg);
            printf("%s :\n", argv[i]);
            l = ns_msg_count(msg, ns_s_an);
            for (j = 0; j < l; j++) {
                ns_parserr(&msg, ns_s_an, j, &rr);
                ns_sprintrr(&msg, &rr, NULL, NULL, dispbuf, sizeof(dispbuf));
                printf("%s\n", dispbuf);
            }
        }

        // IPv6
        l = res_query(argv[i], ns_c_in, ns_t_aaaa, nsbuf, sizeof(nsbuf));
        if (l < 0) {
            perror(argv[i]);
        } else {
            /* MX answer info */
            ns_initparse(nsbuf, l, &msg);
            printf("%s :\n", argv[i]);
            l = ns_msg_count(msg, ns_s_an);
            for (j = 0; j < l; j++) {
                ns_parserr(&msg, ns_s_an, j, &rr);
                ns_sprintrr(&msg, &rr, NULL, NULL, dispbuf, sizeof(dispbuf));
                printf("%s\n", dispbuf);
            }
        }
    }

    exit(0);
}
