#!/bin/bash
set -e
cd `dirname $0`/testContainers

#TODO streamline & test java images (right now github actions runs out of space)
# poor man's solution for lack of space: pull images for one batch, run one batch, remove images, repeat...
enabled="cribl detect_proto elastic gogen interposed_func kafka nginx splunk tls go_2 go_3 go_4 go_5 go_6 go_7 go_8 go_9 go_10 go_11 go_12 go_13 go_14 go_15 go_16"
docker-compose pull -q $enabled
docker-compose build --parallel $enabled
# TODO run paralelly or in batches?
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
