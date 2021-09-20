/* Decode Huffman string -- use for benchmarking
 *
 * Usage: huff-decode $file $count $mode
 *
 * $mode is either "fast" or "slow"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LS_HPACK_USE_LARGE_TABLES
#define LS_HPACK_USE_LARGE_TABLES 1
#endif

#if LS_HPACK_USE_LARGE_TABLES
int
lshpack_dec_huff_decode_full (const unsigned char *src, int src_len,
                                    unsigned char *dst, int dst_len);
#endif

int
lshpack_dec_huff_decode (const unsigned char *src, int src_len,
                                    unsigned char *dst, int dst_len);

int
main (int argc, char **argv)
{
    size_t in_sz;
    int count, i, rv;
    FILE *in;
    int (*decode)(const unsigned char *, int, unsigned char *, int);
    unsigned char in_buf[0x1000];
    unsigned char out_buf[0x4000];

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s $file $count $mode\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (strcasecmp(argv[3], "slow") == 0)
#if LS_HPACK_USE_LARGE_TABLES
        decode = lshpack_dec_huff_decode_full;
#else
        decode = lshpack_dec_huff_decode;
#endif
    else if (strcasecmp(argv[3], "fast") == 0)
#if LS_HPACK_USE_LARGE_TABLES
        decode = lshpack_dec_huff_decode;
#else
    {
        fprintf(stderr, "support for fast decoder is compiled out\n");
        exit(EXIT_FAILURE);
    }
#endif
    else
    {
        fprintf(stderr, "Mode `%s' is invalid.  Specify either `slow' or "
                                                        "`fast'.\n", argv[3]);
        exit(EXIT_FAILURE);
    }

    in = fopen(argv[1], "rb");
    if (!in)
    {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    in_sz = fread(in_buf, 1, sizeof(in_buf), in);
    if (in_sz == 0 || in_sz == sizeof(in_buf))
    {
        fprintf(stderr, "input file is either too short or too long\n");
        exit(EXIT_FAILURE);
    }
    (void) fclose(in);

    count = atoi(argv[2]);
    if (!count)
        count = 1;

    /* TODO: validate against slow if fast is selected */

    rv = decode(in_buf, in_sz, out_buf, sizeof(out_buf));
    if (rv < 0)
    {
        fprintf(stderr, "decode-%s returned %d\n", argv[3], rv);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < count; ++i)
    {
        rv = decode(in_buf, in_sz, out_buf, sizeof(out_buf));
        (void) rv;
    }

    exit(EXIT_SUCCESS);
}
