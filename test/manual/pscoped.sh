#!/usr/bin/bash
#
# pscoped - ps showing only scoped processes
#
# Handy with watch; i.e. `watch -n.5 sudo scoped.sh`
#
# Warning: the `grep /proc/*/maps` won't include results for processes you
# don't have access to so if you run this as a non-root user, you may miss some
# results. Consider running this under `sudu`.
#

LIBS=$(grep libscope /proc/*/maps 2>/dev/null | grep -v 'ldscope')

PIDS=$(sed 's/^\/proc\/\([0-9]*\).*/\1/' <<< $LIBS)

if [ -n "$PIDS" ]; then
    echo "AppScope is loaded into the following processes."
    echo
    ps -f --pid $PIDS
else 
    echo "No scoped processes found."
fi

