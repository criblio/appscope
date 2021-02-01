#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ $GITHUB_REF == "refs/heads/master" ]; then
    echo $SCOPECI_TOKEN | docker login -u scopeci --password-stdin
    cd ${DIR}
    docker build -t cribl/scope-demo -f demo/Dockerfile .
    docker push cribl/scope-demo
fi