#!/bin/bash -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}

# manage the version of node used
echo "versions originally provided by the environment..."
npm --version
node --version
nvm --version || true
echo "installing nvm"
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
source ~/.profile
nvm install 14.18.1
nvm use 14.18.1
echo "versions we're going to use..."
npm --version
node --version
nvm --version

npm ci
npx gatsby build

BUCKET=io.appscope.staging
DISTRIBUTION_ID=E2O0IS8RABQ4AT

if [[ $GITHUB_REF == refs/tags/web* ]]; then
    BUCKET=io.appscope
    DISTRIBUTION_ID=E3CI6UPKUT68NJ
fi

aws s3 rm s3://${BUCKET} --recursive
aws s3 cp ${DIR}/public s3://${BUCKET} --recursive
aws cloudfront create-invalidation --distribution-id=${DISTRIBUTION_ID} --paths '/*'
