/*
 * Testing string search algorithms
 *
 * gcc test/manual/stringsearch.c -Wall -g -o ss
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <syslog.h>
#include <unistd.h>

#define ASIZE 256
//#define OUTPUT(n) printf("Found a match at index %d\n", n);
#define OUTPUT(n)
#define TRUE     1
#define FALSE    0
#define MAX_PROC 4096
#define BINSIZE  (1024 * 10)

typedef int bool;

char haystack[] = "We are in self-imposed stay at home. It is  quarantine";
char needle[] = "stay at home";

char req[] =
    "GET /analytics?ver=3&visitor_id=27707599&visitor_id_sign=fc9fc24376a273c02f9495a7a8c5aa7692434d4d1d33664536f7c241fb80682adfcf1844f36aca5766543ef51ded29cc0dfcbe48&pi_opt_in=&campaign_id=38807&account_id=764193&title=Top%20News%20%7C%20Positive%20Encouraging%20K-LOVE&url=https%3A%2F%2Fww2.klove.com%2Fnews%2F&referrer=http%3A%2F%2Fwww.klove.com%2F HTTP/1.1\r\nHost: pi.pardot.com\r\nUser-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0\r\nAccept: */*\r\nAccept-Language: en-US,en;q=0.5\r\nAccept-Encoding: gzip, deflate, br\r\nConnection: keep-alive\r\nReferer: https://ww2.klove.com/";

char resp[] =
    "HTTP/1.0 200 OK\r\nDate: Wed, 08 Apr 2020 14:56:09 GMT\r\nSet-Cookie: pardot=2uggt7qdo2jjtk82rkdd7ubr78; path=/\r\nExpires: Thu, 19 Nov 1981 08:52:00 GMT\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nX-Pardot-Rsp: 16/55/51\r\nP3p: CP=\"NOI DSP COR CURa ADMa DEVa TAIa OUR BUS IND UNI COM NAV INT\", policyref=\"/w3c/p3p.xml\", CP=\"NOI DSP COR CURa ADMa DEVa TAIa OUR BUS IND UNI COM NAV INT\", policyref=\"/w3c/p3p.xml\"\r\nSet-Cookie: visitor_id763193=27707599; expires=Sat, 06-Apr-2030 14:56:09 GMT; Max-Age=315360000; path=/; SameSite=None; domain=.pardot.com; secure\r\nSet-Cookie: visitor_id763193-hash=6ed3e7b79411fdc3b63f04fa9482d83761dbbfca1eca7513b13f04cb50764f25efba96d6374a6cb6422bf9a4e80b8b5a9b1aebdb; expires=Sat, 06-Apr-2030 14:56:09 GMT; Max-Age=315360000; path=/; SameSite=None; domain=.pardot.com; secure\r\nSet-Cookie: lpv763193=aHR0cHM6Ly93dzIua2xvdmUuY29tL25ld3Mv; expires=Wed, 08-Apr-2020 15:26:09 GMT; Max-Age=1800; path=/; SameSite=None; secure\r\nVary: Accept-Encoding,User-Agent\r\nContent-Encoding: gzip\r\nContent-Length: 558\r\nContent-Type: text/javascript; charset=utf-8\r\nX-Pardot-Route: 13c7a24cfc43e49b0467af9964bf67ec\r\nServer: PardotServer\r\nX-Pardot-LB: a5df88223e39cf9fcb783877fed82f24\r\nConnection: keep-alive";

typedef struct {
    bool tsc_invariant;
    bool tsc_rdtscp;
    uint64_t freq;
} platform_time_t;

platform_time_t g_time;

static inline uint64_t
getTime(void)
{
    unsigned low, high;

    if (g_time.tsc_rdtscp == TRUE) {
        asm volatile("rdtscp" : "=a"(low), "=d"(high));
    } else {
        asm volatile("rdtsc" : "=a"(low), "=d"(high));
    }
    return ((uint64_t)low) | (((uint64_t)high) << 32);
}

// Return the time delta from start to now in nanoseconds
static inline uint64_t
getDuration(uint64_t start)
{
    // before the constructor runs, g_time.freq is zero.
    // Avoid div by zero during this time.
    if (!g_time.freq)
        return 0ULL;

    /*
     * The clock frequency is in Mhz.
     * In order to get NS resolution we
     * multiply the difference by 1000.
     *
     * If the counter rolls over we adjust
     * by using the max value of the counter.
     * A roll over is rare. But, we should handle it.
     */
    uint64_t now = getTime();
    if (start < now) {
        return ((now - start) * 1000) / g_time.freq;
    } else {
        return (((ULONG_MAX - start) + now) * 1000) / g_time.freq;
    }
}

static int
initTSC(platform_time_t *cfg)
{
    int fd;
    char *entry, *last;
    const char delim[] = ":";
    const char path[] = "/proc/cpuinfo";
    const char freqStr[] = "cpu MHz";
    char *buf;

    if ((fd = open(path, O_RDONLY)) == -1) {
        perror("open");
        return -1;
    }

    /*
     * Anecdotal evidence that there is a max size to proc entrires.
     * In any case this should be big enough.
     */
    if ((buf = malloc(MAX_PROC)) == NULL) {
        perror("malloc");
        close(fd);
        return -1;
    }

    if (read(fd, buf, MAX_PROC) == -1) {
        perror("read");
        close(fd);
        free(buf);
        return -1;
    }

    if (strstr(buf, "rdtscp") == NULL) {
        cfg->tsc_rdtscp = FALSE;
    } else {
        cfg->tsc_rdtscp = TRUE;
    }

    if (strstr(buf, "tsc_reliable") == NULL) {
        cfg->tsc_invariant = FALSE;
    } else {
        cfg->tsc_invariant = TRUE;
    }

    entry = strtok_r(buf, delim, &last);
    while (1) {
        if ((entry = strtok_r(NULL, delim, &last)) == NULL) {
            cfg->freq = (uint64_t)-1;
            break;
        }

        if (strcasestr((const char *)entry, freqStr) != NULL) {
            // The next token should be what we want
            if ((entry = strtok_r(NULL, delim, &last)) != NULL) {
                if ((cfg->freq = (uint64_t)strtoll(entry, NULL, 0)) == (long long)0) {
                    cfg->freq = (uint64_t)-1;
                }
                break;
            }
        }
    }

    close(fd);
    free(buf);
    if (cfg->freq == (uint64_t)-1) {
        printf("ERROR: no clk freq\n");
        return -1;
    }

    return 0;
}

static int
memsearch(const char *hay, int haysize, const char *needle, int needlesize)
{
    int haypos, needlepos;

    haysize -= needlesize;
    for (haypos = 0; haypos <= haysize; haypos++) {
        for (needlepos = 0; needlepos < needlesize; needlepos++) {
            if (hay[haypos + needlepos] != needle[needlepos]) {
                // Next character in haystack.
                break;
            }
        }
        if (needlepos == needlesize) {
            return haypos;
        }
    }
    return -1;
}

void
preBmBc(unsigned char *x, int m, int bmBc[])
{
    int i;

    for (i = 0; i < ASIZE; ++i)
        bmBc[i] = m;
    for (i = 0; i < m - 1; ++i)
        bmBc[x[i]] = m - i - 1;
}

void
HORSPOOL(char *x, int m, char *y, int n)
{
    int j, bmBc[ASIZE];
    unsigned char c;

    /* Preprocessing */
    preBmBc((unsigned char *)x, m, bmBc);

    /* Searching */
    j = 0;
    while (j <= n - m) {
        c = y[j + m - 1];
        if (x[m - 1] == c && memcmp(x, y + j, m - 1) == 0)
            OUTPUT(j);
        j += bmBc[c];
    }
}

void
preComp(unsigned char *needle, int nlen, int bmBc[])
{
    int i;

    for (i = 0; i < ASIZE; ++i)
        bmBc[i] = nlen;
    for (i = 0; i < nlen - 1; ++i)
        bmBc[needle[i]] = nlen - i - 1;
}

void
strsrch(char *needle, int nlen, char *haystack, int hlen, int *bmBc)
{
    int j;
    unsigned char c;

    /* Preprocessing; passed in as bmBc */

    /* Searching */
    j = 0;
    while (j <= hlen - nlen) {
        c = haystack[j + nlen - 1];
        if (needle[nlen - 1] == c && memcmp(needle, haystack + j, nlen - 1) == 0) {
            // printf("%s: Found a match at index %d\n", __FUNCTION__, j);
        }

        j += bmBc[c];
    }
}

int
main(int argc, char **argv)
{
    uint64_t start, dur;

    if (argc < 2) {
        printf("Which test?\n");
        exit(1);
    }

    printf("Starting string search test\n");

    if (initTSC(&g_time) == -1) {
        printf("ERROR: TSC\n");
        exit(1);
    }

    if (memcmp(argv[1], "H", 2) == 0) {
        start = getTime();
        HORSPOOL(needle, strlen(needle), haystack, strlen(haystack));
        // dur = getDuration(start);
        // printf("Horspool @ %ld ns\n", dur);

        HORSPOOL("HTTP/1", strlen("HTTP/1"), req, strlen(req));
        HORSPOOL("HTTP/1", strlen("HTTP/1"), resp, strlen(resp));

        HORSPOOL("HTTP/1", strlen("HTTP/1"), resp, strlen("HTTP/1"));
        HORSPOOL("HTTP/1", strlen("HTTP/1"), req, strlen("HTTP/1"));
        dur = getDuration(start);
        printf("Horspool @ %ld ns\n", dur);
    } else if (memcmp(argv[1], "M", 2) == 0) {
        // if (memsearch(haystack, strlen(haystack), needle, strlen(needle)) != -1) printf("T1 Pass\n");
        // if (memsearch(req, strlen(req), "HTTP/1", strlen("HTTP/1")) != -1) printf("T2 Pass\n");
        // if (memsearch(resp, strlen(resp), "HTTP/1", strlen("HTTP/1")) != -1) printf("T3 Pass\n");

        // if (memsearch(resp, strlen("HTTP/1"), "HTTP/1", strlen("HTTP/1")) != -1) printf("T4 Pass\n");
        // if (memsearch(req, strlen("HTTP/1"), "HTTP/1", strlen("HTTP/1")) == -1) printf("T5 fail Expected\n");

        start = getTime();
        memsearch(haystack, strlen(haystack), needle, strlen(needle));
        // dur = getDuration(start);
        // printf("Brute Force @ %ld ns\n", dur);

        memsearch(req, strlen(req), "HTTP/1", strlen("HTTP/1"));
        memsearch(resp, strlen(resp), "HTTP/1", strlen("HTTP/1"));

        memsearch(resp, strlen("HTTP/1"), "HTTP/1", strlen("HTTP/1"));
        memsearch(req, strlen("HTTP/1"), "HTTP/1", strlen("HTTP/1"));

        dur = getDuration(start);
        printf("Brute Force @ %ld ns\n", dur);
    } else if (memcmp(argv[1], "S", 2) == 0) {
        int needle_home[ASIZE];
        int needle_http[ASIZE];

        preComp((unsigned char *)needle, strlen(needle), needle_home);
        preComp((unsigned char *)"HTTP/1", strlen("HTTP/1"), needle_http);

        start = getTime();
        strsrch(needle, strlen(needle), haystack, strlen(haystack), needle_home);
        strsrch("HTTP/1", strlen("HTTP/1"), req, strlen(req), needle_http);
        strsrch("HTTP/1", strlen("HTTP/1"), resp, strlen(resp), needle_http);

        strsrch("HTTP/1", strlen("HTTP/1"), resp, strlen("HTTP/1"), needle_http);
        strsrch("HTTP/1", strlen("HTTP/1"), req, strlen("HTTP/1"), needle_http);
        dur = getDuration(start);
        printf("Strsrch @ %ld ns\n", dur);
    } else if (memcmp(argv[1], "B", 2) == 0) {
        int i;
        int needle_http[ASIZE];
        int *haystack;

        if ((haystack = malloc(BINSIZE)) == NULL) {
            perror("malloc");
            exit(1);
        }

        for (i = 0; i < (BINSIZE / sizeof(int)); i++) {
            haystack[i] = i;
        }

        preComp((unsigned char *)"HTTP/1", strlen("HTTP/1"), needle_http);

        start = getTime();
        strsrch("HTTP/1", strlen("HTTP/1"), (char *)haystack, BINSIZE, needle_http);
        dur = getDuration(start);
        printf("Strsrch @ %ld ns\n", dur);

        start = getTime();
        HORSPOOL("HTTP/1", strlen("HTTP/1"), (char *)haystack, BINSIZE);
        dur = getDuration(start);
        printf("Horspool @ %ld ns\n", dur);

        start = getTime();
        memsearch((char *)haystack, BINSIZE, "HTTP/1", strlen("HTTP/1"));
        dur = getDuration(start);
        printf("Brute Force @ %ld ns\n", dur);

        free(haystack);
    } else {
        printf("Wrong test!!\n");
    }

    exit(0);
}
