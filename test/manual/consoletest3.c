// gcc -g test/manual/consoletest3.c -o consoletest3

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>
#include <wchar.h>

// This test was created when we moved from instrumenting lots of individual string
// functions to funchooking glibc's __write.  It tries to guarantee coverage and 
// guard against "double capture" too.


void 
vprintf_func(char *format, ...) 
{
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

void
vfprintf_func(FILE *stream, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stream, format, args);
    va_end(args);
}

void
vdprintf_func(int fd, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vdprintf(fd, format, args);
    va_end(args);
}

void
vsprintf_func(char *buf, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsprintf(buf, format, args);
    va_end(args);
}

void
vsnprintf_func(char *buf, size_t size, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(buf, size, format, args);
    va_end(args);
}

#define LOG_FILE_NAME "/tmp/consoletestfile.log"

int
main(void)
{
    // just a reminder...  the stdout stream is *line buffered* by default.


    // These should appear in console events
    int i = 0;
    printf("%d:printf\n", i++);                 // to stdout
    fprintf(stdout, "%d:fprintf\n", i++);
    vprintf_func("%d:vprintf\n", i++);          // to stdout
    vfprintf_func(stdout, "%d:vfprintf\n", i++);
    dprintf(STDOUT_FILENO, "%d:dprintf\n", i++);
    vdprintf_func(STDOUT_FILENO, "%d:vdprintf\n", i++);


    // Not to stdout. These should *not* appear in console events
    char buf[256];
    sprintf(buf, "%d:sprintf\n", i);
    snprintf(buf, sizeof(buf), "%d:snprintf\n", i);
    vsprintf_func(buf, "%d:vsprintf\n", i);
    vsnprintf_func(buf, sizeof(buf), "%d:vsnprintf\n", i);


    // These should appear in file events
    int fd = open(LOG_FILE_NAME, O_CREAT|O_CLOEXEC|O_WRONLY|O_APPEND);
    pwrite64(fd, "pwrite64\n", strlen("pwrite64\n"), 0); // pwrite64
    { 
        struct iovec iov[2] = {{"p", 1}, {"writev\n", strlen("writev\n")}};
        pwritev(fd, iov, 2, 0);
    }
    {
        struct iovec iov[2] = {{"p", 1}, {"writev64\n", strlen("writev64\n")}};
        pwritev64(fd, iov, 2, 0);
    }
    {
        struct iovec iov[2] = {{"p", 1}, {"writev2\n", strlen("writev2\n")}};
        pwritev2(fd, iov, 2, 0, 0);
    }
    {
        struct iovec iov[2] = {{"p", 1}, {"writev64v2\n", strlen("writev64v2\n")}};
        pwritev64v2(fd, iov, 2, 0, 0);
    }
    pwrite(fd, "pwrite\n", strlen("pwrite\n"), 0);
    close(fd);
    unlink(LOG_FILE_NAME);

/*
    __overflow("
*/

    // These should appear in console events
    fwrite_unlocked("fwrite_unlocked\n", 1, strlen("fwrite_unlocked\n"), stdout);
    write(STDOUT_FILENO, "write\n", strlen("write\n"));
    {
        struct iovec iov[2] = {{"wri", 3}, {"tev\n", 4}};
        writev(STDOUT_FILENO, iov, 2);
    }
    fwrite("fwrite\n", 1, strlen("fwrite\n"), stdout);
    puts("puts");
    putchar('J');
    fputs("fputs\n", stdout);
    fputs_unlocked("fputs_unlocked\n", stdout);

    fputc('Q', stdout);
    fputc_unlocked('Z', stdout);
    fputc_unlocked('\n', stdout);


    // "... you must not mix byte and wide oriented operations on the same FILE
    // stream.  Failure to observe this rule is not a reportable error; it
    // simply results in undefined behavior."

    // These should appear in file events
    FILE * f = fopen(LOG_FILE_NAME, "a");
    fwide(f, 1);
    putwc(L'M', f);
    fputwc(L'N', f);
    fputwc(L'\n', f);
    fclose(f);
    unlink(LOG_FILE_NAME);

    return 0;
}
