/* find-xxh -- find XXH seed common for ls-hpack and ls-qpack
 *
 * To speed up decoding/encoding process in a proxy, we need to use the
 * same name and nameval XXH hashes in ls-hpack and ls-qpack libraries.
 * This program finds an XXH seed common to both and shifts and offsets
 * for construction of name and nameval XXH tables in both libraries.
 */

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include XXH_HEADER_NAME

#define NAME_VAL(a, b) sizeof(a) - 1, sizeof(b) - 1, (a), (b)

struct table_elem
{
    unsigned          name_len;
    unsigned          val_len;
    const char       *name;
    const char       *val;
};


static const struct table_elem hpack_table[] =
{
    { NAME_VAL(":authority",                    "") },
    { NAME_VAL(":method",                       "GET") },
    { NAME_VAL(":method",                       "POST") },
    { NAME_VAL(":path",                         "/") },
    { NAME_VAL(":path",                         "/index.html") },
    { NAME_VAL(":scheme",                       "http") },
    { NAME_VAL(":scheme",                       "https") },
    { NAME_VAL(":status",                       "200") },
    { NAME_VAL(":status",                       "204") },
    { NAME_VAL(":status",                       "206") },
    { NAME_VAL(":status",                       "304") },
    { NAME_VAL(":status",                       "400") },
    { NAME_VAL(":status",                       "404") },
    { NAME_VAL(":status",                       "500") },
    { NAME_VAL("accept-charset",                "") },
    { NAME_VAL("accept-encoding",               "gzip, deflate") },
    { NAME_VAL("accept-language",               "") },
    { NAME_VAL("accept-ranges",                 "") },
    { NAME_VAL("accept",                        "") },
    { NAME_VAL("access-control-allow-origin",   "") },
    { NAME_VAL("age",                           "") },
    { NAME_VAL("allow",                         "") },
    { NAME_VAL("authorization",                 "") },
    { NAME_VAL("cache-control",                 "") },
    { NAME_VAL("content-disposition",           "") },
    { NAME_VAL("content-encoding",              "") },
    { NAME_VAL("content-language",              "") },
    { NAME_VAL("content-length",                "") },
    { NAME_VAL("content-location",              "") },
    { NAME_VAL("content-range",                 "") },
    { NAME_VAL("content-type",                  "") },
    { NAME_VAL("cookie",                        "") },
    { NAME_VAL("date",                          "") },
    { NAME_VAL("etag",                          "") },
    { NAME_VAL("expect",                        "") },
    { NAME_VAL("expires",                       "") },
    { NAME_VAL("from",                          "") },
    { NAME_VAL("host",                          "") },
    { NAME_VAL("if-match",                      "") },
    { NAME_VAL("if-modified-since",             "") },
    { NAME_VAL("if-none-match",                 "") },
    { NAME_VAL("if-range",                      "") },
    { NAME_VAL("if-unmodified-since",           "") },
    { NAME_VAL("last-modified",                 "") },
    { NAME_VAL("link",                          "") },
    { NAME_VAL("location",                      "") },
    { NAME_VAL("max-forwards",                  "") },
    { NAME_VAL("proxy-authenticate",            "") },
    { NAME_VAL("proxy-authorization",           "") },
    { NAME_VAL("range",                         "") },
    { NAME_VAL("referer",                       "") },
    { NAME_VAL("refresh",                       "") },
    { NAME_VAL("retry-after",                   "") },
    { NAME_VAL("server",                        "") },
    { NAME_VAL("set-cookie",                    "") },
    { NAME_VAL("strict-transport-security",     "") },
    { NAME_VAL("transfer-encoding",             "") },
    { NAME_VAL("user-agent",                    "") },
    { NAME_VAL("vary",                          "") },
    { NAME_VAL("via",                           "") },
    { NAME_VAL("www-authenticate",              "") }
};
#define HPACK_STATIC_TABLE_SIZE (sizeof(hpack_table) / sizeof(hpack_table[0]))

static const unsigned hpack_name_indexes[] = {
    0, 1, 3, 5, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
};
#define HPACK_NAME_SIZE (sizeof(hpack_name_indexes) / sizeof(hpack_name_indexes[0]))

/* [draft-ietf-quic-qpack-03] Appendix A */
static const struct table_elem qpack_table[] =
{
    { NAME_VAL(":authority", "") },
    { NAME_VAL(":path", "/") },
    { NAME_VAL("age", "0") },
    { NAME_VAL("content-disposition", "") },
    { NAME_VAL("content-length", "0") },
    { NAME_VAL("cookie", "") },
    { NAME_VAL("date", "") },
    { NAME_VAL("etag", "") },
    { NAME_VAL("if-modified-since", "") },
    { NAME_VAL("if-none-match", "") },
    { NAME_VAL("last-modified", "") },
    { NAME_VAL("link", "") },
    { NAME_VAL("location", "") },
    { NAME_VAL("referer", "") },
    { NAME_VAL("set-cookie", "") },
    { NAME_VAL(":method", "CONNECT") },
    { NAME_VAL(":method", "DELETE") },
    { NAME_VAL(":method", "GET") },
    { NAME_VAL(":method", "HEAD") },
    { NAME_VAL(":method", "OPTIONS") },
    { NAME_VAL(":method", "POST") },
    { NAME_VAL(":method", "PUT") },
    { NAME_VAL(":scheme", "http") },
    { NAME_VAL(":scheme", "https") },
    { NAME_VAL(":status", "103") },
    { NAME_VAL(":status", "200") },
    { NAME_VAL(":status", "304") },
    { NAME_VAL(":status", "404") },
    { NAME_VAL(":status", "503") },
    { NAME_VAL("accept", "*/*") },
    { NAME_VAL("accept", "application/dns-message") },
    { NAME_VAL("accept-encoding", "gzip, deflate, br") },
    { NAME_VAL("accept-ranges", "bytes") },
    { NAME_VAL("access-control-allow-headers", "cache-control") },
    { NAME_VAL("access-control-allow-headers", "content-type") },
    { NAME_VAL("access-control-allow-origin", "*") },
    { NAME_VAL("cache-control", "max-age=0") },
    { NAME_VAL("cache-control", "max-age=2592000") },
    { NAME_VAL("cache-control", "max-age=604800") },
    { NAME_VAL("cache-control", "no-cache") },
    { NAME_VAL("cache-control", "no-store") },
    { NAME_VAL("cache-control", "public, max-age=31536000") },
    { NAME_VAL("content-encoding", "br") },
    { NAME_VAL("content-encoding", "gzip") },
    { NAME_VAL("content-type", "application/dns-message") },
    { NAME_VAL("content-type", "application/javascript") },
    { NAME_VAL("content-type", "application/json") },
    { NAME_VAL("content-type", "application/x-www-form-urlencoded") },
    { NAME_VAL("content-type", "image/gif") },
    { NAME_VAL("content-type", "image/jpeg") },
    { NAME_VAL("content-type", "image/png") },
    { NAME_VAL("content-type", "text/css") },
    { NAME_VAL("content-type", "text/html; charset=utf-8") },
    { NAME_VAL("content-type", "text/plain") },
    { NAME_VAL("content-type", "text/plain;charset=utf-8") },
    { NAME_VAL("range", "bytes=0-") },
    { NAME_VAL("strict-transport-security", "max-age=31536000") },
    { NAME_VAL("strict-transport-security", "max-age=31536000; includesubdomains") },
    { NAME_VAL("strict-transport-security", "max-age=31536000; includesubdomains; preload") },
    { NAME_VAL("vary", "accept-encoding") },
    { NAME_VAL("vary", "origin") },
    { NAME_VAL("x-content-type-options", "nosniff") },
    { NAME_VAL("x-xss-protection", "1; mode=block") },
    { NAME_VAL(":status", "100") },
    { NAME_VAL(":status", "204") },
    { NAME_VAL(":status", "206") },
    { NAME_VAL(":status", "302") },
    { NAME_VAL(":status", "400") },
    { NAME_VAL(":status", "403") },
    { NAME_VAL(":status", "421") },
    { NAME_VAL(":status", "425") },
    { NAME_VAL(":status", "500") },
    { NAME_VAL("accept-language", "") },
    { NAME_VAL("access-control-allow-credentials", "FALSE") },
    { NAME_VAL("access-control-allow-credentials", "TRUE") },
    { NAME_VAL("access-control-allow-headers", "*") },
    { NAME_VAL("access-control-allow-methods", "get") },
    { NAME_VAL("access-control-allow-methods", "get, post, options") },
    { NAME_VAL("access-control-allow-methods", "options") },
    { NAME_VAL("access-control-expose-headers", "content-length") },
    { NAME_VAL("access-control-request-headers", "content-type") },
    { NAME_VAL("access-control-request-method", "get") },
    { NAME_VAL("access-control-request-method", "post") },
    { NAME_VAL("alt-svc", "clear") },
    { NAME_VAL("authorization", "") },
    { NAME_VAL("content-security-policy", "script-src 'none'; object-src 'none'; base-uri 'none'") },
    { NAME_VAL("early-data", "1") },
    { NAME_VAL("expect-ct", "") },
    { NAME_VAL("forwarded", "") },
    { NAME_VAL("if-range", "") },
    { NAME_VAL("origin", "") },
    { NAME_VAL("purpose", "prefetch") },
    { NAME_VAL("server", "") },
    { NAME_VAL("timing-allow-origin", "*") },
    { NAME_VAL("upgrade-insecure-requests", "1") },
    { NAME_VAL("user-agent", "") },
    { NAME_VAL("x-forwarded-for", "") },
    { NAME_VAL("x-frame-options", "deny") },
    { NAME_VAL("x-frame-options", "sameorigin") },
};
#define QPACK_STATIC_TABLE_SIZE (sizeof(qpack_table) / sizeof(qpack_table[0]))

/* This is calculated at runtime */
static unsigned qpack_name_indexes[QPACK_STATIC_TABLE_SIZE];
static unsigned n_qpack_name_indexes;

/* QPACK static table has an interesting property that headers names are
 * not all placed in contiguous sequences.  For example, :status appears
 * in two places in the table.
 */
static void
calculate_qpack_name_indexes (void)
{
    unsigned i, n;

    qpack_name_indexes[0] = 0;
    n_qpack_name_indexes = 1;
    for (n = 1; n < QPACK_STATIC_TABLE_SIZE; ++n)
    {
        for (i = 0; i < n
                && !(qpack_table[i].name_len == qpack_table[n].name_len
                && 0 == memcmp(qpack_table[i].name,
                            qpack_table[n].name, qpack_table[n].name_len)); ++i)
            ;
        if (i == n)
            qpack_name_indexes[n_qpack_name_indexes++] = n;
    }
}


/* Need to strike some balance here.  A small start width will result in
 * potentially long search time and -- more importantly -- in a hpack_table that
 * is very small, which may require lookup code to perform more comparisons.
 * On the other hand, a large width will result in a hpack_table that may be
 * slower to use.
 */
static unsigned MIN_WIDTH = 9;
static unsigned MAX_WIDTH = 9;

static unsigned MIN_SHIFT = 0;
static unsigned MAX_SHIFT = 31;

/* Return true if acceptable shift and width were found, false otherwise */
static int
find_shift_and_width (const uint32_t *hashes, const unsigned n_hashes,
                                        unsigned *shift_p, unsigned *width_p)
{
    unsigned shift, width, hash, i, j;

    for (width = MIN_WIDTH; width <= MAX_WIDTH; ++width)
    {
        for (shift = MIN_SHIFT; shift <= MAX_SHIFT
                                            && shift < 32 - width; ++shift)
        {
            for (i = 1; i < n_hashes; ++i)
            {
                hash = hashes[i] & (((1u << width) - 1) << shift);
                for (j = 0; j < i; ++j)
                    if ((hashes[j] & (((1u << width) - 1) << shift)) == hash)
                        goto check;
            }
  check:    if (i >= n_hashes)
            {
                *shift_p = shift;
                *width_p = width;
                return 1;
            }
        }
    }

    return 0;
}


int
main (int argc, char **argv)
{
    uint32_t hpack_name_hashes[HPACK_NAME_SIZE];
    uint32_t hpack_nameval_hashes[HPACK_STATIC_TABLE_SIZE];
    uint32_t qpack_nameval_hashes[QPACK_STATIC_TABLE_SIZE];
    uint32_t qpack_name_hashes[QPACK_STATIC_TABLE_SIZE];
    uint32_t seed, init_seed = 0;
    unsigned n, idx;
    unsigned hpack_nameval_shift, hpack_name_width, hpack_nameval_width,
                hpack_name_shift;
    unsigned qpack_nameval_shift, qpack_name_width, qpack_nameval_width,
                qpack_name_shift;
    int opt, dont_stop = 0, print_tables = 0;

    while (-1 != (opt = getopt(argc, argv, "i:w:W:s:S:phN")))
    {
        switch (opt)
        {
        case 'i':
            init_seed = atoi(optarg);
            break;
        case 'w':
            MIN_WIDTH = atoi(optarg);
            break;
        case 'W':
            MAX_WIDTH = atoi(optarg);
            break;
        case 's':
            MIN_SHIFT = atoi(optarg);
            break;
        case 'S':
            MAX_SHIFT = atoi(optarg);
            break;
        case 'p':
            print_tables = 1;
            break;
        case 'N':
            dont_stop = 1;
            break;
        case 'h': printf(
"Usage: %s [options]\n"
"\n"
"   -i seed     Initial seed (defaults to 0)\n"
"   -w width    Minimum width (defaults to %u)\n"
"   -W width    Maximum width (defaults to %u)\n"
"   -s shift    Minimum shift (defaults to %u)\n"
"   -S shift    Maximum shift (defaults to %u)\n"
"   -N          Don't stop after finding a match, keep searching\n"
"   -p          Print resulting HPACK and QPACK tables\n"
"   -h          Print this help screen and exit\n"
            , argv[0], MIN_WIDTH, MAX_WIDTH, MIN_SHIFT, MAX_SHIFT);
            return 0;
        }
    }

    seed = init_seed;
    calculate_qpack_name_indexes();

  again:
    for (n = 0; n < HPACK_NAME_SIZE; ++n)
    {
        idx = hpack_name_indexes[n];
        hpack_name_hashes[n] = XXH32(hpack_table[idx].name,
                                        hpack_table[idx].name_len, seed);
    }
    if (!find_shift_and_width(hpack_name_hashes, HPACK_NAME_SIZE,
                                    &hpack_name_shift, &hpack_name_width))
        goto incr_seed;

    for (n = 0; n < HPACK_STATIC_TABLE_SIZE; ++n)
    {
        hpack_nameval_hashes[n] = XXH32(hpack_table[n].name,
                                            hpack_table[n].name_len, seed);
        hpack_nameval_hashes[n] = XXH32(hpack_table[n].val,
                            hpack_table[n].val_len, hpack_nameval_hashes[n]);
    }
    if (!find_shift_and_width(hpack_nameval_hashes, HPACK_STATIC_TABLE_SIZE,
                                &hpack_nameval_shift, &hpack_nameval_width))
        goto incr_seed;

    for (n = 0; n < n_qpack_name_indexes; ++n)
    {
        idx = qpack_name_indexes[n];
        qpack_name_hashes[n] = XXH32(qpack_table[idx].name,
                                            qpack_table[idx].name_len, seed);
    }
    if (!find_shift_and_width(qpack_name_hashes, n_qpack_name_indexes,
                                    &qpack_name_shift, &qpack_name_width))
        goto incr_seed;

    for (n = 0; n < QPACK_STATIC_TABLE_SIZE; ++n)
    {
        qpack_nameval_hashes[n] = XXH32(qpack_table[n].name,
                                                qpack_table[n].name_len, seed);
        qpack_nameval_hashes[n] = XXH32(qpack_table[n].val,
                            qpack_table[n].val_len, qpack_nameval_hashes[n]);
    }
    if (!find_shift_and_width(qpack_nameval_hashes, QPACK_STATIC_TABLE_SIZE,
                                &qpack_nameval_shift, &qpack_nameval_width))
        goto incr_seed;

    printf("unique set: seed %u\n"
           "  hpack:\n"
           "    name shift: %u; width: %u\n"
           "    nameval shift: %u; width: %u\n"
           "  qpack:\n"
           "    name shift: %u; width: %u\n"
           "    nameval shift: %u; width: %u\n"
           , seed
           , hpack_name_shift, hpack_name_width
           , hpack_nameval_shift, hpack_nameval_width
           , qpack_name_shift, qpack_name_width
           , qpack_nameval_shift, qpack_nameval_width
           );

    if (print_tables)
    {
        printf("#define XXH_SEED %"PRIu32"\n", seed);

        printf("#define XXH_HPACK_NAME_WIDTH %"PRIu32"\n", hpack_name_width);
        printf("#define XXH_HPACK_NAME_SHIFT %"PRIu32"\n", hpack_name_shift);
        printf("static const unsigned char hpack_name2id[ 1 << XXH_HPACK_NAME_WIDTH ] =\n{\n");
        for (n = 0; n < HPACK_NAME_SIZE; ++n)
            printf("[%u] = %u, ", (hpack_name_hashes[n] >> hpack_name_shift)
                    & ((1 << hpack_name_width) - 1), hpack_name_indexes[n] + 1);
        printf("\n};\n");

        printf("#define XXH_HPACK_NAMEVAL_WIDTH %"PRIu32"\n", hpack_nameval_width);
        printf("#define XXH_HPACK_NAMEVAL_SHIFT %"PRIu32"\n", hpack_nameval_shift);
        printf("static const unsigned char hpack_nameval2id[ 1 << XXH_HPACK_NAMEVAL_WIDTH ] =\n{\n");
        for (n = 0; n < HPACK_STATIC_TABLE_SIZE; ++n)
            printf("[%u] = %u, ", (hpack_nameval_hashes[n] >> hpack_nameval_shift)
                    & ((1 << hpack_nameval_width) - 1), n + 1);
        printf("\n};\n");

        printf("#define XXH_QPACK_NAME_WIDTH %"PRIu32"\n", qpack_name_width);
        printf("#define XXH_QPACK_NAME_SHIFT %"PRIu32"\n", qpack_name_shift);
        printf("static const unsigned char qpack_name2id[ 1 << XXH_QPACK_NAME_WIDTH ] =\n{\n");
        for (n = 0; n < n_qpack_name_indexes; ++n)
            printf("[%u] = %u, ", (qpack_name_hashes[n] >> qpack_name_shift)
                    & ((1 << qpack_name_width) - 1), qpack_name_indexes[n] + 1);
        printf("\n};\n");

        printf("#define XXH_QPACK_NAMEVAL_WIDTH %"PRIu32"\n", qpack_nameval_width);
        printf("#define XXH_QPACK_NAMEVAL_SHIFT %"PRIu32"\n", qpack_nameval_shift);
        printf("static const unsigned char qpack_nameval2id[ 1 << XXH_QPACK_NAMEVAL_WIDTH ] =\n{\n");
        for (n = 0; n < QPACK_STATIC_TABLE_SIZE; ++n)
            printf("[%u] = %u, ", (qpack_nameval_hashes[n] >> qpack_nameval_shift)
                    & ((1 << qpack_nameval_width) - 1), n + 1);
        printf("\n};\n");
    }

    if (dont_stop)
    {
        fflush(stdout);
        goto incr_seed;
    }

#if 0   /* TODO */
    for (i = 0; i < TABLE_SIZE; ++i)
#if NAMEVAL_SEARCH
        printf("[%u] = %u,\n", (hashes[i] >> min_shift) & ((1 << min_width) - 1), nameval_indexes[i] + 1);
#else
        printf("[%u] = %u,\n", (hashes[i] >> min_shift) & ((1 << min_width) - 1), name_indexes[i] + 1);
#endif
#endif

    return 0;

  incr_seed:
    ++seed;
    if ((seed - init_seed) % 100000 == 0)
        fprintf(stderr, "....  seed: %u\n", seed);
    goto again;
}
