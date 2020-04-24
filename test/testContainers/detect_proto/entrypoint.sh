#!/bin/sh
mkdir -p /var/log/gogen
mkdir /var/log/beats

if [ "$1" = "start" ]; then
#    sleep 5 && sh /sbin/loaddata.sh &
    redis-server --protected-mode no &

    /opt/test-runner/bin/test_protocols.sh
#    tail -f /dev/null
fi

