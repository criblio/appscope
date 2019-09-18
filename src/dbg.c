#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dbg.h"


#define MAX_INSTANCES_PER_LINE 2
#define DEFAULT_NUM_LINES 8

typedef struct {
    time_t time;
    int err; // errno
    char* str;
} occurrence_t;

typedef struct {
    const char* key;
    unsigned long long count;
    occurrence_t instance[MAX_INSTANCES_PER_LINE];
} line_t;

struct _dbg_t {
    line_t** lines;
    unsigned max_lines;
};

dbg_t* g_dbg = NULL;


void
dbgInit()
{
    dbgDestroy();

    g_dbg = calloc(1, sizeof(dbg_t));
    if (!g_dbg) return;
    g_dbg->max_lines = DEFAULT_NUM_LINES;

}

static void
updateLine(line_t* line, char* str)
{
    if (!line) return;

    // This keeps overwriting the latest one.
    int i = (line->count < MAX_INSTANCES_PER_LINE )
            ? line->count
            : MAX_INSTANCES_PER_LINE - 1;

    line->instance[i].time = time(NULL);
    line->instance[i].err = errno;
    if (line->instance[i].str) free(line->instance[i].str);
    line->instance[i].str = str;
    line->count++;
}

static line_t*
createLine(const char* key, char* str)
{
    if (!key) return NULL;
    line_t* line = calloc(1, sizeof(line_t));
    if (!line) return NULL;

    line->key = key;
    updateLine(line, str);

    return line;
}

static void
destroyLine(line_t** line)
{
    if (!line || !*line) return;
    int i;
    for (i=0; i < MAX_INSTANCES_PER_LINE; i++) {
        if ((*line)->instance[i].str) free((*line)->instance[i].str);
    }
    free(*line);
    *line = NULL;
}

void
dbgDestroy()
{
     if (!g_dbg) return;

     if (g_dbg->lines) {
        int i = 0;
        while (g_dbg->lines[i]) destroyLine(&g_dbg->lines[i++]);
        free(g_dbg->lines);
    }

    free(g_dbg);
    g_dbg = NULL;
}

// Accessors
unsigned long long
dbgCountAllLines()
{
    if (!g_dbg || !g_dbg->lines) return 0ULL;

    unsigned long long i;
    for (i=0; i<g_dbg->max_lines && g_dbg->lines[i]; i++);
    return i;
}

unsigned long long
dbgCountMatchingLines(const char* str)
{
    if (!g_dbg || !g_dbg->lines || !str) return 0ULL;
    unsigned long long result = 0ULL;
    unsigned long long i;
    for (i=0; i<g_dbg->max_lines && g_dbg->lines[i]; i++) {
        if (strstr(g_dbg->lines[i]->key, str)) {
            result++;
        }
    }
    return result;
}

static void
dbgOutputHeaderLine(FILE* f)
{
    if (!f) return;

    time_t t;
    time(&t);

    char buf[128] = {0};
    strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
    fprintf(f, "Scope Version: %s   Dump From: %s\n", SCOPE_VER, buf);
}

void
dbgDumpAll(FILE* f)
{
    if (!f) return;

    dbgOutputHeaderLine(f);

    if (!g_dbg || !g_dbg->lines) return;

    int i;
    for (i=0; i<g_dbg->max_lines && g_dbg->lines[i]; i++) {
        line_t* l = g_dbg->lines[i];
        occurrence_t* o = &l->instance[0];

        struct tm t;
        if (!gmtime_r(&o->time, &t)) continue;
        char t_str[64];
        if (!strftime(t_str, sizeof(t_str), "%s", &t)) continue;
        fprintf(f, "%llu: %s %s %d(%s) %s\n",
                l->count, l->key,
                t_str, o->err, strerror(o->err), o->str);

        o = &l->instance[1];
        if (!o->time) continue;
        if (!gmtime_r(&o->time, &t)) continue;
        if (!strftime(t_str, sizeof(t_str), "%s", &t)) continue;
        fprintf(f, "    %s %d(%s) %s\n",
                t_str, o->err, strerror(o->err), o->str);
    }
}

// Setters
void
dbgAddLine(const char* key, const char* fmt, ...)
{
    if (!g_dbg) return;

    if (!g_dbg->lines) {
        g_dbg->lines = calloc(1, sizeof(line_t*) * g_dbg->max_lines);
        if (!g_dbg->lines) return;
    }

    // Create the string
    char* str = NULL;
    if (fmt) {
        va_list argptr;
        va_start(argptr, fmt);
        if (vasprintf(&str, fmt, argptr) == -1) {
            return;
        }
    }

    // See if the line is already there
    line_t* line = 0;
    int i = 0;
    while (g_dbg->lines[i]) {
        if (!strcmp(key, g_dbg->lines[i]->key)) {
            line = g_dbg->lines[i];
            break;
        }
        i++;
    }

    // The line is already there, just update it
    if (line) {
        updateLine(line, str);
        return;
    }

    // The line (key) is not there.  Find space, or create it.
    for (i=0; i<g_dbg->max_lines && g_dbg->lines[i]; i++);

    // If we're out of space, try to get more space
    if (i >= g_dbg->max_lines-1) {     // null delimiter is always required
        int tmp_max_lines = g_dbg->max_lines * 2;  // double each time
        line_t** temp = realloc(g_dbg->lines, sizeof(line_t*) * tmp_max_lines);
        if (!temp) {
            if (str) free(str);
            return;
        }
        // Yeah!  We have more space!  init it, and set our state to remember it
        memset(&temp[g_dbg->max_lines], 0, sizeof(line_t*) * (tmp_max_lines - g_dbg->max_lines));
        g_dbg->lines = temp;
        g_dbg->max_lines = tmp_max_lines;
    }

    // Create the new line
    g_dbg->lines[i] = createLine(key, str);
}
