#! /bin/bash

# LD_BIND_NOW requires that all symbols be able to be resolved.
#
# If this test fails, libscope.so has an external dependency on something
# the dynamic linker couldn't resolve.  We don't want to ship anything
# like this because some environments set LD_BIND_NOW.
export LD_BIND_NOW=1


export SCOPE_EVENT_METRIC=true
export SCOPE_EVENT_HTTP=true
export SCOPE_EVENT_LOG=true
export SCOPE_EVENT_CONSOLE=true
export SCOPE_EVENT_DEST=file:///tmp/scope_events.log
export LD_PRELOAD=./lib/linux/$(uname -m)/libscope.so

declare -i ERR=0

echo "================================="
echo "      Undefined Symbol Test      "
echo "================================="

/bin/echo "All symbols can be resolved."
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}
