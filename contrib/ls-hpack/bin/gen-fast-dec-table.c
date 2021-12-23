/* Generate table for fast Huffman decoding */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


struct el
{
    uint16_t    code;
    unsigned    bits;
    uint8_t     out;
};

static const struct el els[] =
{
    {        0x0,     5,  48, },
    {        0x1,     5,  49, },
    {        0x2,     5,  50, },
    {        0x3,     5,  97, },
    {        0x4,     5,  99, },
    {        0x5,     5, 101, },
    {        0x6,     5, 105, },
    {        0x7,     5, 111, },
    {        0x8,     5, 115, },
    {        0x9,     5, 116, },
    {       0x14,     6,  32, },
    {       0x15,     6,  37, },
    {       0x16,     6,  45, },
    {       0x17,     6,  46, },
    {       0x18,     6,  47, },
    {       0x19,     6,  51, },
    {       0x1a,     6,  52, },
    {       0x1b,     6,  53, },
    {       0x1c,     6,  54, },
    {       0x1d,     6,  55, },
    {       0x1e,     6,  56, },
    {       0x1f,     6,  57, },
    {       0x20,     6,  61, },
    {       0x21,     6,  65, },
    {       0x22,     6,  95, },
    {       0x23,     6,  98, },
    {       0x24,     6, 100, },
    {       0x25,     6, 102, },
    {       0x26,     6, 103, },
    {       0x27,     6, 104, },
    {       0x28,     6, 108, },
    {       0x29,     6, 109, },
    {       0x2a,     6, 110, },
    {       0x2b,     6, 112, },
    {       0x2c,     6, 114, },
    {       0x2d,     6, 117, },
    {       0x5c,     7,  58, },
    {       0x5d,     7,  66, },
    {       0x5e,     7,  67, },
    {       0x5f,     7,  68, },
    {       0x60,     7,  69, },
    {       0x61,     7,  70, },
    {       0x62,     7,  71, },
    {       0x63,     7,  72, },
    {       0x64,     7,  73, },
    {       0x65,     7,  74, },
    {       0x66,     7,  75, },
    {       0x67,     7,  76, },
    {       0x68,     7,  77, },
    {       0x69,     7,  78, },
    {       0x6a,     7,  79, },
    {       0x6b,     7,  80, },
    {       0x6c,     7,  81, },
    {       0x6d,     7,  82, },
    {       0x6e,     7,  83, },
    {       0x6f,     7,  84, },
    {       0x70,     7,  85, },
    {       0x71,     7,  86, },
    {       0x72,     7,  87, },
    {       0x73,     7,  89, },
    {       0x74,     7, 106, },
    {       0x75,     7, 107, },
    {       0x76,     7, 113, },
    {       0x77,     7, 118, },
    {       0x78,     7, 119, },
    {       0x79,     7, 120, },
    {       0x7a,     7, 121, },
    {       0x7b,     7, 122, },
    {       0xf8,     8,  38, },
    {       0xf9,     8,  42, },
    {       0xfa,     8,  44, },
    {       0xfb,     8,  59, },
    {       0xfc,     8,  88, },
    {       0xfd,     8,  90, },
    {      0x3f8,    10,  33, },
    {      0x3f9,    10,  34, },
    {      0x3fa,    10,  40, },
    {      0x3fb,    10,  41, },
    {      0x3fc,    10,  63, },
    {      0x7fa,    11,  39, },
    {      0x7fb,    11,  43, },
    {      0x7fc,    11, 124, },
    {      0xffa,    12,  35, },
    {      0xffb,    12,  62, },
    {     0x1ff8,    13,   0, },
    {     0x1ff9,    13,  36, },
    {     0x1ffa,    13,  64, },
    {     0x1ffb,    13,  91, },
    {     0x1ffc,    13,  93, },
    {     0x1ffd,    13, 126, },
    {     0x3ffc,    14,  94, },
    {     0x3ffd,    14, 125, },
    {     0x7ffc,    15,  60, },
    {     0x7ffd,    15,  96, },
    {     0x7ffe,    15, 123, },
};


static void
generate_entry (uint16_t idx)
{
    unsigned int bits_left, n_outs;
    const struct el *el;
    uint8_t outs[3];

    bits_left = 16;
    n_outs = 0;
    do
    {
        for (el = els; el < els + sizeof(els)
                        / sizeof(els[0]) && el->bits <= bits_left; ++el)
            if (el->code == (uint32_t) ((idx >> (bits_left - el->bits)) & ((1 << el->bits) - 1)))
                break;
        if (el >= els + sizeof(els) / sizeof(els[0]) || el->bits > bits_left)
            break;
        outs[n_outs++] = el->out;
        bits_left -= el->bits;
    }
    while (bits_left >= 5 /* shortest code */);

    printf("/* %"PRIu16" */ ", idx);
    if (n_outs)
    {
        printf("{(%u<<2)|%u,{", 16 - bits_left, n_outs);
        switch (n_outs)
        {
        case 3:
            printf("%u,%u,%u", outs[0], outs[1], outs[2]);
            break;
        case 2:
            printf("%u,%u,0", outs[0], outs[1]);
            break;
        case 1:
            printf("%u,0,0", outs[0]);
            break;
        default:    exit(EXIT_FAILURE);
        }
        printf("}}");
    }
    else
        printf("{0,0,0,0,}");
    printf(",\n");
}


int
main (void)
{
    unsigned idx;

    printf("static const struct hdec { uint8_t lens; uint8_t out[3]; } "
                                                            "hdecs[] =\n{\n");
    for (idx = 0; idx <= UINT16_MAX; ++idx)
        generate_entry(idx);
    printf("};\n");

    return 0;
}
