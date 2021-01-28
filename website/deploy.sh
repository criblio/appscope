#!/bin/bash -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}

npm ci
npx gatsby build

BUCKET=io.appscope.staging
DISTRIBUTION_ID=E2O0IS8RABQ4AT

if [ $GITHUB_REF == "refs/heads/master" ]; then
    BUCKET=io.appscope
    DISTRIBUTION_ID=E3CI6UPKUT68NJ
fi

aws s3 cp ${DIR}/public s3://${BUCKET} --recursive
aws cloudfront create-invalidation --distribution-id=${DISTRIBUTION_ID} --paths '/*'
