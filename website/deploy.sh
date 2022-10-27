#!/bin/bash -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}


### Build Gatsby Static Docs ###
# nvm is provided by the runner, but needs to be setup for this shell env
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
nvm --version

echo "npm and node versions originally provided..."
npm --version
node --version

# use nvm to control the version of node used
nvm install 14.18.1
nvm use 14.18.1

echo "npm and node versions we're going to use..."
npm --version
node --version

# the algolia plugin will create/update/delete indices
export GATSBY_ALGOLIA_APP_ID=$STAGING_GATSBY_ALGOLIA_APP_ID
export GATSBY_ALGOLIA_SEARCH_KEY=$STAGING_GATSBY_ALGOLIA_SEARCH_KEY
export GATSBY_ALGOLIA_WRITE_KEY=$STAGING_GATSBY_ALGOLIA_WRITE_KEY
if [[ $GITHUB_REF == refs/tags/web* ]]; then
    export GATSBY_ALGOLIA_APP_ID=$PROD_GATSBY_ALGOLIA_APP_ID
    export GATSBY_ALGOLIA_SEARCH_KEY=$PROD_GATSBY_ALGOLIA_SEARCH_KEY
    export GATSBY_ALGOLIA_WRITE_KEY=$PROD_GATSBY_ALGOLIA_WRITE_KEY
fi

# build static docs
npm ci
npx gatsby build


### Push Docs to AWS S3 ###
BUCKET=io.appscope.staging
DISTRIBUTION_ID=E2O0IS8RABQ4AT
if [[ $GITHUB_REF == refs/tags/web* ]]; then
    BUCKET=io.appscope
    DISTRIBUTION_ID=E3CI6UPKUT68NJ
fi

aws s3 rm s3://${BUCKET} --recursive
aws s3 cp ${DIR}/public s3://${BUCKET} --recursive
aws cloudfront create-invalidation --distribution-id=${DISTRIBUTION_ID} --paths '/*'
