#ifndef __LIBVER_H__
#define __LIBVER_H__

const char * libverNormalizedVersion(const char *);

// TODO there is a probably better place to put this function
typedef enum {
    MKDIR_STATUS_CREATED = 0,          // path was created
    MKDIR_STATUS_EXISTS = 1,           // path already points to existing directory
    MKDIR_STATUS_NOT_ABSOLUTE_DIR = 2, // path does not point to a directory
    MKDIR_STATUS_OTHER_ISSUE = 3,      // other error
} mkdir_status_t;

mkdir_status_t libverMkdirNested(const char *);

#endif // __LIBVER_H__
