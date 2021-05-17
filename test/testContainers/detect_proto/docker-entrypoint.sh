#!/bin/bash
set -e

if [ "$1" = "test" ]; then
    mkdir -p /var/log/gogen /var/log/beats

    #sleep 5 && sh /sbin/loaddata.sh &

    redis-server --protected-mode no &

    #service mongod start
    mongod --dbpath /var/lib/mongo --logpath /var/log/mongodb/mongod.log --fork

    exec /opt/test-runner/bin/test_protocols.sh
fi

exec "$@"
