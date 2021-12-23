#!/bin/sh

# Start Cribl and Splunk demo in a single container
docker run \
    -d \
    --rm \
    --name cribl-demo \
    --hostname cribl-demo \
    --publish 8000:8000 \
    --publish 8088:8088 \
    --publish 8089:8089 \
    --publish 9000:9000 \
    --publish 10080:10080 \
    -e DONT_TIMEOUT=1 \
    cribl/cribl-demo:latest
