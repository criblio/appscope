# Internal Libc Notes

With release 1.1 AppScope has added the use of an internal libc. The rationale and design approach are documented in a blog. Refer to the blog for detail. This document is intended to simply delineate categories of functions used with the internal libc. Some functions need to sue the internal libc and some functions need to use the native libc.

The term native libc is meant to imply the libc used with the current Linux distribution. For example; in Ubuntu the native libc is glibc, in Alpine the native libc is musl libc. The term internal libc is meant to describe the instance of libc code contained in libscope.so.

In this document we are focused on categories of functions provided by a libc. We define categories as those utilized from the internal libc and those that use the native libc. A starting point might be to define that all libc functions should utilize the internal libc except for those explicitly listed. Therefore, an exception list has been created. Function categories in the following exception list must utilize the native libc. All other function categories must use the internal libc.

# Exception list
Functions in the following categories must utilize the native libc:
threads
signals
loader functions (e.g. dlopen, dlsym)
environment variables (e.g. getenv, setenv)
getopts (command line arguments)
fork
execve
exit

