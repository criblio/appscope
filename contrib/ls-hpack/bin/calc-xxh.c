/* calc-xxh: calculate XXH32 hashes for name and value strings */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include XXH_HEADER_NAME

int
main (int argc, char **argv)
{
    uint32_t seed, name_hash, nameval_hash;
    size_t name_len, value_len;
    const char *name, *value;
    XXH32_state_t hash_state;
    int mode;

    if (argc != 5)
    {
        fprintf(stderr,
"Usage: %s mode seed name value\n"
"\n"
"   mode is one of:\n"
"       0   Single seed, update\n"
"       1   Use name hash as seed for calculating nameval hash\n"
"\n"
"   seed is the initial seed\n"
"\n", argv[0]);
        return 0;
    }

    mode = atoi(argv[1]);
    if (!(mode == 0 || mode == 1))
    {
        fprintf(stderr, "mode `%s' is invalid\n", argv[1]);
        return 1;
    }

    seed = atoi(argv[2]);
    name = argv[3];
    value = argv[4];
    name_len = strlen(name);
    value_len = strlen(value);

    switch (mode)
    {
    case 0:
        XXH32_reset(&hash_state, seed);
        XXH32_update(&hash_state, name, name_len);
        name_hash = XXH32_digest(&hash_state);
        XXH32_update(&hash_state, value, value_len);
        nameval_hash = XXH32_digest(&hash_state);
        break;
    default:
        name_hash = XXH32(name, name_len, seed);
        nameval_hash = XXH32(value, value_len, name_hash);
        break;
    }

    printf(
"In mode %d, name: `%s', value: `%s' hash to:\n"
"name hash 0x%08"PRIX32"\n"
"nameval hash 0x%08"PRIX32"\n"
        , mode, name, value, name_hash, nameval_hash);

    return 0;
}
