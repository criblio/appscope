#ifndef __SCOPETYPES_H__
#define __SCOPETYPES_H__

typedef enum {CFG_EXPANDED_STATSD, CFG_NEWLINE_DELIMITED} cfg_out_format_t;
typedef enum {CFG_UDP, CFG_UNIX, CFG_FILE, CFG_SYSLOG, CFG_SHM} cfg_transport_t;
typedef enum {CFG_OUT, CFG_LOG, CFG_WHICH_MAX} which_transport_t;
typedef enum {CFG_LOG_DEBUG,
              CFG_LOG_INFO,
              CFG_LOG_WARN,
              CFG_LOG_ERROR,
              CFG_LOG_NONE} cfg_log_level_t;

#endif // __SCOPETYPES_H__

