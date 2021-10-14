/*
 * Read QIF and encode it using HPACK.  Use for benchmarking.
 *
 * Based on ls-qpack's interop-encode.
 *
 * How it works: read in QIF into a list of header set objects and encode
 * the list a number of times.
 *
 * QIF Format:
 * https://github.com/quicwg/base-drafts/wiki/QPACK-Offline-Interop
 */

#define _GNU_SOURCE /* for memmem */
#include <assert.h>

#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__)
#include <sys/endian.h>
#define bswap_16 bswap16
#define bswap_32 bswap32
#define bswap_64 bswap64
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define bswap_16 OSSwapInt16
#define bswap_32 OSSwapInt32
#define bswap_64 OSSwapInt64
#elif defined(WIN32)
#error Not supported on Windows
#else
#include <byteswap.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/mman.h>

#include "lshpack.h"

static int s_verbose;

#define TABLE_SIZE 4096

static void
usage (const char *name)
{
    fprintf(stderr,
"Usage: %s [options] [-i input] [-o output]\n"
"\n"
"Options:\n"
"   -i FILE     Input file.\n"
"   -o FILE     Output file.  If not spepcified or set to `-', the output\n"
"                 is written to stdout.\n"
"   -K          Discard output: encoded output is discarded.\n"
"   -H          Do not use the history heuristic.\n"
"   -n NUMBER   Number of times to iterate over the header set list.\n"
"   -t NUMBER   Dynamic table size.  Defaults to %u.\n"
"   -v          Verbose: print various messages to stderr.\n"
"\n"
"   -h          Print this help screen and exit\n"
    , name, TABLE_SIZE);
}


struct header
{
    const char    *name;
    const char    *val;
    size_t         name_len;
    size_t         val_len;
};


#define MAX_HEADERS 32

struct header_set
{
    STAILQ_ENTRY(header_set)    next;
    unsigned                    n_headers;
    struct header               headers[MAX_HEADERS];
};


static inline void
lsxpack_header_set_ptr(lsxpack_header_t *hdr,
                       const char *name, size_t name_len,
                       const char *val, size_t val_len)
{
    static char buf[65536];
    memcpy(buf, name, name_len);
    memcpy(&buf[name_len], val, val_len);
    lsxpack_header_set_offset2(hdr, buf, 0, name_len, name_len, val_len);
}


int
main (int argc, char **argv)
{
    FILE *out = stdout;
    int opt, qif_fd = -1;
    int discard = 0, use_history = 1;
    unsigned n_iters = 1, n, i;
    unsigned dyn_table_size     = TABLE_SIZE;
    STAILQ_HEAD(, header_set) header_sets = STAILQ_HEAD_INITIALIZER(header_sets);
    struct stat st;
    const unsigned char *qif = NULL, *p, *tab, *nl, *nlnl;
    unsigned char *s;
    struct header *header;
    struct header_set *hset;
    struct lshpack_enc encoder;
    unsigned char buf[0x2000];

    while (-1 != (opt = getopt(argc, argv, "Hi:o:Kn:t:vh")))
    {
        switch (opt)
        {
        case 'n':
            n_iters = atoi(optarg);
            break;
        case 'i':
            qif_fd = open(optarg, O_RDONLY);
            if (qif_fd < 0)
            {
                perror("open");
                exit(EXIT_FAILURE);
            }
            if (0 != fstat(qif_fd, &st))
            {
                perror("fstat");
                exit(EXIT_FAILURE);
            }
            qif = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, qif_fd, 0);
            if (!qif)
            {
                perror("mmap");
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            if (0 != strcmp(optarg, "-"))
            {
                out = fopen(optarg, "wb");
                if (!out)
                {
                    fprintf(stderr, "cannot open `%s' for writing: %s\n",
                                                optarg, strerror(errno));
                    exit(EXIT_FAILURE);
                }
            }
            break;
        case 'K':
            discard = 1;
            break;
        case 'H':
            use_history = 0;
            break;
        case 't':
            dyn_table_size = atoi(optarg);
            break;
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        case 'v':
            ++s_verbose;
            break;
        default:
            exit(EXIT_FAILURE);
        }
    }

    if (!qif)
    {
        fprintf(stderr, "Please specify input QIF file using -i flag\n");
        exit(EXIT_FAILURE);
    }

    const unsigned char *const begin = qif;
    const unsigned char *const end = begin + st.st_size;
    while (qif + 2 < end)
    {
        nlnl = memmem(qif, end - qif, "\n\n", 2);
        if (!nlnl)
            nlnl = end;
        hset = calloc(1, sizeof(*hset));
        if (!hset)
        {
            perror("malloc");
            exit(EXIT_FAILURE);
        }
        STAILQ_INSERT_TAIL(&header_sets, hset, next);
        p = qif;
        while (p < nlnl)
        {
            if (hset->n_headers >= MAX_HEADERS)
            {
                fprintf(stderr, "max headers > 32, off: %u",
                                            (unsigned) (p - begin));
                exit(EXIT_FAILURE);
            }
            tab = memmem(p, nlnl - p, "\t", 1);
            if (!tab)
            {
                fprintf(stderr, "tab not found, off: %u",
                                            (unsigned) (p - begin));
                exit(EXIT_FAILURE);
            }
            nl = memmem(tab + 1, nlnl - tab - 1, "\n", 1);
            if (!nl)
                nl = nlnl;
            hset->headers[ hset->n_headers ] = (struct header) {
                .name = (const char *) p,
                .val = (const char *) tab + 1,
                .name_len =  tab - p,
                .val_len = nl - tab - 1,
            };
            ++hset->n_headers;
            p = nl + 1;
        }
        qif = nlnl + 2;
    }

    lsxpack_header_t hdr;
    for (n = 0; n < n_iters; ++n)
    {
        if (0 != lshpack_enc_init(&encoder))
        {
            perror("lshpack_enc_init");
            exit(EXIT_FAILURE);
        }
        (void) lshpack_enc_use_hist(&encoder, use_history);

        STAILQ_FOREACH(hset, &header_sets, next)
        {
            for (i = 0; i < hset->n_headers; ++i)
            {
                header = &hset->headers[i];
                lsxpack_header_set_ptr(&hdr, header->name, header->name_len,
                                       header->val, header->val_len);
                s = lshpack_enc_encode(&encoder, buf, buf + sizeof(buf), &hdr);
                if (s <= buf)
                {
                    fprintf(stderr, "cannot encode\n");
                    exit(EXIT_FAILURE);
                }
                if (!discard)
                    (void) fwrite(buf, 1, s - buf, out);
            }
        }

        lshpack_enc_set_max_capacity(&encoder, dyn_table_size);
        lshpack_enc_cleanup(&encoder);
    }

    munmap((void *) begin, st.st_size);
    if (qif_fd >= 0)
        close(qif_fd);
    exit(EXIT_SUCCESS);
}
