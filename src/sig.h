#ifndef __SIG_H__
#define __SIG_H__

#include <signal.h>
#include "scopetypes.h"

bool sigIsAppcopeActionActive(void);
bool sigIsAppActionInstalled(void);
void sigCallAppAction(int, siginfo_t *, void *);
bool sigIsSigFromAppscopeTimer(const siginfo_t *);
void sigSaveAppAction(const struct sigaction *);
bool sigHandlerRegister(int, void(*handler)(int, siginfo_t *, void *));
bool sigTimerStart(int, unsigned);
bool sigTimerStop(void);

#endif // __SIG_H__
