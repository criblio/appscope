#!/bin/bash
set -e
cd `dirname $0`/testContainers

# working="cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls"
enabled="cribl detect_proto"
docker-compose pull -q $enabled
docker-compose build --parallel $enabled
for s in $enabled; do
    docker-compose up $s
done

docker-compose ps
if docker-compose ps |tail -n+3|grep -v 'Exit 0'; then
    2>&1 echo Integration tests failed
    exit 1
fi

if [ $GITHUB_REF = 'refs/heads/master' ]; then
    docker-compose push
fi
