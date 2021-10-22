#ifndef __WRAP_H__
#define __WRAP_H__

#define EXPORT __attribute__((visibility("default")))
#define EXPORTOFF  __attribute__((visibility("hidden")))
#define EXPORTON __attribute__((visibility("default")))
#define EXPORTWEAK __attribute__((weak))

#define DYN_CONFIG_PREFIX "scope"
#define MAXTRIES 10
#define CONN_LOG_INTERVAL 10

typedef struct nss_list_t {
    uint64_t id;
    PRIOMethods *ssl_methods;
    PRIOMethods *ssl_int_methods;
} nss_list;

typedef struct thread_timing_t {
    unsigned interval;                   // in seconds
    time_t startTime; 
    bool once;
    pthread_t periodicTID;
    const struct sigaction *act;
} thread_timing;

typedef struct {
    uint64_t initial;
    uint64_t duration;
} elapsed_t;

extern int close$NOCANCEL(int);
extern int guarded_close_np(int, void *);

// struct to hold the next 6 numeric (int/ptr etc) variadic arguments
// use LOAD_FUNC_ARGS_VALIST to populate this structure
struct FuncArgs{
    uint64_t arg[6]; // pick the first 6 args
};

#define LOAD_FUNC_ARGS_VALIST(a, lastNamedArg)  \
    do{                                         \
        va_list __args;                         \
        va_start(__args, lastNamedArg);         \
        a.arg[0] = va_arg(__args, uint64_t);    \
        a.arg[1] = va_arg(__args, uint64_t);    \
        a.arg[2] = va_arg(__args, uint64_t);    \
        a.arg[3] = va_arg(__args, uint64_t);    \
        a.arg[4] = va_arg(__args, uint64_t);    \
        a.arg[5] = va_arg(__args, uint64_t);    \
        va_end(__args);                         \
    }while(0)

#endif // __WRAP_H__
