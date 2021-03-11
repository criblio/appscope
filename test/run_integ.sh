#!/bin/bash
set -e
# working="cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls"
enabled="cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls go_6 go_7 go_8 go_9 go_10 go_11 go_12 go_13 go_14 go_15 go_16"
cd `dirname $0`/testContainers
docker-compose build $enabled
for s in $enabled; do
    docker-compose up $s
done

if docker-compose ps |tail -n+3|grep -v 'Exit 0'; then
    2>&1 echo some containers failed
    exit 1
fi
