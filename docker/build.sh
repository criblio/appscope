#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VERSION=$(cat ${DIR}/../cli/VERSION)

echo $SCOPECI_TOKEN | docker login -u scopeci --password-stdin
cd ${DIR}
docker build -t cribl/scope:${VERSION} -f base/Dockerfile ..
docker push cribl/scope:${VERSION}
docker build -t cribl/scope-demo:${VERSION} --build-arg VERSION=${VERSION} -f demo/Dockerfile ..
docker push cribl/scope-demo:${VERSION}

if [[ $GITHUB_REF == "refs/tags/v"* && $GITHUB_REF != "refs/tags/"*"-rc"* ]]; then
    docker tag cribl/scope:${VERSION} cribl/scope:latest
    docker tag cribl/scope-demo:${VERSION} cribl/scope-demo:latest
    docker push cribl/scope:latest
    docker push cribl/scope-demo:latest
fi
