#!/bin/bash
set -e
cd `dirname $0`/testContainers
docker-compose build cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls
for s in cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls; do
    docker-compose up --exit-code-from $s $s
done
