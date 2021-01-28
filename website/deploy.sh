#!/bin/bash -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

npm ci
npx gatsby build

BUCKET=io.appscope.staging

if [ $GITHUB_REF == "refs/heads/master" ]; then
    BUCKET=io.appscope
fi

aws s3 cp ${DIR}/public s3://$BUCKET --recursive
