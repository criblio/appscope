#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lshpack.h"
#include "lshpack-test.h"

struct int_test
{
    int             it_lineno;
    unsigned        it_prefix_bits;
    unsigned char   it_encoded[20];
    size_t          it_enc_sz;
    uint32_t        it_decoded;
    int             it_dec_retval;
};

static const struct int_test tests[] =
{

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
        .it_encoded     = { 0x7F, 0x02, },
        .it_enc_sz      = 2,
        .it_decoded     = 0x81,
        .it_dec_retval  = 0,
    },

    /* RFC 7541, Appendinx C.1.1 */
    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 5,
        .it_encoded     = { 0b1010, 0x02, },
        .it_enc_sz      = 1,
        .it_decoded     = 10,
        .it_dec_retval  = 0,
    },

    /* RFC 7541, Appendinx C.1.2 */
    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 5,
        .it_encoded     = { 0b11111, 0b10011010, 0b00001010, },
        .it_enc_sz      = 3,
        .it_decoded     = 1337,
        .it_dec_retval  = 0,
    },

    /* RFC 7541, Appendinx C.1.3 */
    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 8,
        .it_encoded     = { 0b101010, },
        .it_enc_sz      = 1,
        .it_decoded     = 42,
        .it_dec_retval  = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
        .it_encoded     = { 0b01111111, 0b10000001, 0b10000010, 0b00000011, },
        .it_enc_sz      = 4,
                       /*     01234560123456 */
        .it_decoded     = 0b1100000100000001    + 0b1111111,
        .it_dec_retval  = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
        .it_encoded     = { 0b01111111, 0b10000001, 0b10000010, 0b10000011,
                            0b00000011, },
        .it_enc_sz      = 5,
                       /*     012345601234560123456 */
        .it_decoded     = 0b11000001100000100000001    + 0b1111111,
        .it_dec_retval  = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
        .it_encoded     = { 0b01111111, 0b10000000, 0b11111111, 0b11111111,
                            0b11111111, 0b00001111, },
        .it_enc_sz      = 6,
        .it_decoded     = UINT32_MAX,
        .it_dec_retval  = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
            /* Same as above, but with extra bit that overflows it */
                                      /* ----v---- */
        .it_encoded     = { 0b01111111, 0b10010000, 0b11111111, 0b11111111,
                            0b11111111, 0b00001111, },
        .it_enc_sz      = 6,
        .it_dec_retval  = -2,
    },

    {   .it_lineno      = __LINE__,
        .it_prefix_bits = 7,
        .it_encoded     = { 0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, },
        .it_enc_sz      = 11,
        .it_dec_retval  = -2,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0xFE, },
        .it_enc_sz      = 1,
        .it_prefix_bits = 4,
        .it_dec_retval  = 0,
        .it_decoded     = 0xE,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0x3F, 0x64, },
        .it_enc_sz      = 2,
        .it_prefix_bits = 6,
        .it_dec_retval  = 0,
        .it_decoded     = 0x3F + 0x64,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0xFF, },
        .it_enc_sz      = 1,
        .it_prefix_bits = 4,
        .it_dec_retval  = -1,   /* Ran out of buffer */
        .it_decoded     = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0x87, 0xF0, 0x80, 0x7F, },
        .it_enc_sz      = 4,
        .it_prefix_bits = 3,
        .it_dec_retval  = 0,
        .it_decoded     = 0b111111100000001110111,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0x87, 0x80, 0x80, 0x80, 0x01, },
        .it_enc_sz      = 5,
        .it_prefix_bits = 3,
        .it_dec_retval  = 0,
        .it_decoded     = 0b1000000000000000000111,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0x87, 0x80, 0x80, 0x80, 0x01, },
        .it_enc_sz      = 4,
        .it_prefix_bits = 3,
        .it_dec_retval  = -1,   /* Ran out of buffer */
        .it_decoded     = 0,
    },

    {   .it_lineno      = __LINE__,
        .it_encoded     = { 0x87, 0x80, 0x80, 0x80, 0x80, 0x01, },
        .it_enc_sz      = 6,
        .it_prefix_bits = 3,
        .it_dec_retval  = 0,
        .it_decoded     = 0b10000000000000000000000000111,
    },

};

int
main (void)
{
    const struct int_test *test;
    const unsigned char *src;
    unsigned char *dst;
    unsigned char buf[ sizeof(((struct int_test *) NULL)->it_encoded) ];
    uint32_t val;
    size_t sz;
    int rv;

    /* Test the decoder */
    for (test = tests; test < tests + sizeof(tests) / sizeof(tests[0]); ++test)
    {
        for (sz = 0; sz < test->it_enc_sz - 1; ++sz)
        {
            src = test->it_encoded;
            rv = lshpack_dec_dec_int(&src, src + sz, test->it_prefix_bits, &val);
            if (test->it_dec_retval == -2)
                assert(-1 == rv || -2 == rv);
            else
                assert(-1 == rv);
        }
        src = test->it_encoded;
        rv = lshpack_dec_dec_int(&src, src + test->it_enc_sz,
                                                    test->it_prefix_bits, &val);
        assert(rv == test->it_dec_retval);
        if (0 == rv)
            assert(val == test->it_decoded);
    }

    /* Test the encoder */
    for (test = tests; test < tests + sizeof(tests) / sizeof(tests[0]); ++test)
    {
        if (test->it_dec_retval != 0)
            continue;
        for (sz = 1; sz < test->it_enc_sz; ++sz)
        {
            dst = lshpack_enc_enc_int(buf, buf + sz, test->it_decoded,
                                                        test->it_prefix_bits);
            assert(dst == buf);     /* Not enough room */
        }
        for (; sz <= sizeof(buf); ++sz)
        {
            buf[0] = '\0';
            dst = lshpack_enc_enc_int(buf, buf + sz, test->it_decoded,
                                                        test->it_prefix_bits);
            assert(dst - buf == (intptr_t) test->it_enc_sz);
            assert((test->it_encoded[0] & ((1 << test->it_prefix_bits) - 1))
                                                                    == buf[0]);
            if (test->it_enc_sz > 1)
                assert(0 == memcmp(buf + 1, test->it_encoded + 1,
                                                        test->it_enc_sz - 1));
        }
    }

    return 0;
}
