#! /bin/bash

export SCOPE_CRIBL_ENABLE=false
export SCOPE_EVENT_METRIC=true
export SCOPE_EVENT_DEST=file:///tmp/scope_events.log
export LD_PRELOAD=./lib/linux/$(uname -m)/libscope.so

declare -i ERR=0

echo "================================="
echo "      Access Rights Test         "
echo "================================="

./test/linux/passfd -f /tmp/pass.pipe -1
ERR+=$?

./test/linux/passfd -f /tmp/pass.pipe -2
ERR+=$?

./test/linux/passfd -f /tmp/pass.pipe -3
ERR+=$?

./test/linux/passfd -f /tmp/pass.pipe -4
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}
