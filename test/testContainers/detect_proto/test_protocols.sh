#! /bin/bash

declare -i ERR=0
preload=`env | grep LD_PRELOAD`

echo "==============================================="
echo "             Testing Redis                     "
echo "==============================================="

redis-cli SET detect hello
unset LD_PRELOAD

grep remote_protocol /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep '"protocol":"Redis"' /opt/test-runner/logs/events.log > /dev/null
ERR+=$?


if [ $ERR -eq "0" ]; then
    echo "*************** Redis Success ***************"
else
    echo "*************** Redis Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

if [ "x86_64" = "$(uname -m)" ]; then # x86_64 only

echo "==============================================="
echo "             Testing Mongo                     "
echo "==============================================="

export $preload
mongo /opt/test-runner/bin/mongo.js
unset LD_PRELOAD

grep remote_protocol /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep '"protocol":"Mongo"' /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** Mongo Success ***************"
else
    echo "*************** Mongo Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

fi # x86_64 only

rm /opt/test-runner/logs/events.log
exit ${ERR}
