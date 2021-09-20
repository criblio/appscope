/* huff-encode -- Huffman-encode string */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
lshpack_enc_huff_encode (const unsigned char *src,
    const unsigned char *const src_end, unsigned char *const dst, int dst_len);

int
main (int argc, char **argv)
{
    unsigned char buf[0x1000];
    size_t len;
    int sz;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s string > output\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    len = strlen(argv[1]);
    sz = lshpack_enc_huff_encode((unsigned char *) argv[1],
                            (unsigned char *) argv[1] + len, buf, sizeof(buf));
    fwrite(buf, 1, sz, stdout);

    exit(EXIT_SUCCESS);
}
