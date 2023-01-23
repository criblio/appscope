#ifndef __SIGNALHANDLER_H__
#define __SIGNALHANDLER_H__

#include <signal.h>

/*
 *  Manage signal handler for backtrace
 */

void scopeSignalHandlerBacktrace(int , siginfo_t *, void *);

#endif // __SIGNALHANDLER_H__
