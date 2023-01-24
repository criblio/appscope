#ifndef __SIGNALHANDLER_H__
#define __SIGNALHANDLER_H__

#include <signal.h>

/*
 *  Manage signal handler for backtrace
 *  IMPORTANT NOTE:
 *  The API used in this module must be the signal safety
 *  https://man7.org/linux/man-pages/man7/signal-safety.7.html
 */

void scopeSignalHandlerBacktrace(int, siginfo_t *, void *);

#endif // __SIGNALHANDLER_H__
