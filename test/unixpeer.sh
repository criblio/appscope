#! /bin/bash

export SCOPE_EVENT_METRIC=true
export SCOPE_EVENT_DEST=file:///tmp/scope_peer.log
export LD_PRELOAD=./lib/linux/$(uname -m)/libscope.so

declare -i ERR=0

echo "================================="
echo "      UNIX Socket Peer Test      "
echo "================================="

./test/linux/unixpeer -v -f /tmp/pass.pipe
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}
