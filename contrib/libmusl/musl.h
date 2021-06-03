#ifndef __LIBMUSL_H__
#define __LIBMUSL_H__

/*
 * From musl/src/internal/stdio_impl.h
 * This is specifc to libmusl.
 * It is fragile.
 *
 * We modify the write function pointer in the
 * stdout & stderr structures to cause writes
 * to use our function; __stdio_write().
 * In order to do this we need to know the definition
 * of the internal FILE struct for the stdio functions.
 * If the structure changes this has the potential
 * to break.
 * The following pointer is hooked:
 * size_t (*write)(FILE *, const unsigned char *, size_t);
 */
struct MUSL_IO_FILE {
	unsigned flags;
	unsigned char *rpos, *rend;
	int (*close)(FILE *);
	unsigned char *wend, *wpos;
	unsigned char *mustbezero_1;
	unsigned char *wbase;
	size_t (*read)(FILE *, unsigned char *, size_t);
	size_t (*write)(FILE *, const unsigned char *, size_t);
	off_t (*seek)(FILE *, off_t, int);
	unsigned char *buf;
	size_t buf_size;
	FILE *prev, *next;
	int fd;
	int pipe_pid;
	long lockcount;
	int mode;
	volatile int lock;
	int lbf;
	void *cookie;
	off_t off;
	char *getln_buf;
	void *mustbezero_2;
	unsigned char *shend;
	off_t shlim, shcnt;
	FILE *prev_locked, *next_locked;
	struct __locale_struct *locale;
};

#endif // __LIBMUSL_H__
