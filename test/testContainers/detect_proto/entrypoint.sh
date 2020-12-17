#!/bin/sh
mkdir -p /var/log/gogen
mkdir /var/log/beats

if [ "$1" = "start" ]; then
#    sleep 5 && sh /sbin/loaddata.sh &
    redis-server --protected-mode no &
    #    service mongod start
    mongod --dbpath /var/lib/mongo --logpath /var/log/mongodb/mongod.log --fork

    /opt/test-runner/bin/test_protocols.sh
#    tail -f /dev/null
fi

