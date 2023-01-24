#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.h"

unsigned long g_libscopesz;
unsigned long g_scopedynsz;

int
groupSetup(void** state)
{
    return 0;
}

int
groupTeardown(void** state)
{
    return 0;
}

int
writeFile(const char* path, const char* text)
{
    FILE* f = fopen(path, "w");
    if (!f)
        fail_msg("Couldn't open file");

    if (!fwrite(text, strlen(text), 1, f))
        fail_msg("Couldn't write file");

    if (fclose(f))
        fail_msg("Couldn't close file");

    return 0;
}

int
deleteFile(const char* path)
{
    return unlink(path);
}

long
fileEndPosition(const char* path)
{
    FILE* f;
    if ((f = fopen(path, "r"))) {
        fseek(f, 0, SEEK_END);
        long pos = ftell(f);
        fclose(f);
        return pos;
    }
    return -1;
}
