#!/bin/bash
set -e
# working="cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls"
enabled="cribl detect_proto elastic gogen kafka nginx splunk tls"
cd `dirname $0`/testContainers
docker-compose build $enabled
for s in $enabled; do
    docker-compose up --exit-code-from $s $s
done
