#ifndef __SNAPSHOT_H__
#define __SNAPSHOT_H__

#include <signal.h>
#include "scopetypes.h"
/*
 *  Manage snapshot
 *  IMPORTANT NOTE:
 *  The API used in this module must be the signal safety
 *  https://man7.org/linux/man-pages/man7/signal-safety.7.html
 */

/*
 * Signal handler for snapshot (SIGSEGV, SIGBUS, SIGILL and SIGFPE)
 */

void snapshotSignalHandler(int, siginfo_t *, void *);
void snapshotSetCoredump(bool);
void snapshotSetBacktrace(bool);
bool snapshotIsEnabled(void);

/*
 * API to handle application signal handler
 */

bool snapshotBackupAppSignalHandler(int, const struct sigaction*);
bool snapshotRetrieveAppSignalHandler(int, struct sigaction * const);

#endif // __SNAPSHOT_H__
