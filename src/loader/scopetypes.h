#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

#include <stdbool.h>

/***********************************************************************
 * Consider updating src/scopetypes.h if you make changes to this file *
 ***********************************************************************/

#define ARRAY_SIZE(a) (sizeof(a)/sizeof((a)[0]))
#define C_STRLEN(a)  (sizeof(a) - 1)

#define FALSE 0
#define TRUE 1

#define SCOPE_FILTER_USR_PATH ("/usr/lib/appscope/scope_filter")
#define SCOPE_FILTER_TMP_PATH ("/tmp/appscope/scope_filter")
#define SCOPE_USR_PATH "/usr/lib/appscope/"
#define SCOPE_TMP_PATH "/tmp/appscope/"

#define DYN_CONFIG_CLI_DIR "/dev/shm"
#define DYN_CONFIG_CLI_PREFIX "scope_dconf"

#define SM_NAME "scope_anon"

#define SCOPE_PID_ENV "SCOPE_PID"

typedef enum {
    SERVICE_STATUS_SUCCESS = 0,         // service operation was success
    SERVICE_STATUS_ERROR_OTHER = 1,     // service was not installed
    SERVICE_STATUS_NOT_INSTALLED = 2    // service operation was failed
} service_status_t;

typedef enum {
    CFG_LOG_TRACE,
    CFG_LOG_DEBUG,
    CFG_LOG_INFO,
    CFG_LOG_WARN,
    CFG_LOG_ERROR,
    CFG_LOG_NONE
} cfg_log_level_t;

typedef struct {
    unsigned long cmdAttachAddr;
    bool scoped;
} export_sm_t;

#endif // __SCOPETYPES_H__
