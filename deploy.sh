#!/bin/bash -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ $GITHUB_REF == "refs/tags/v*" ]]; then
    VERSION=$(echo ${GITHUB_REF} | sed -e "s*^refs/tags/v**")
    LATEST=$VERSION
else if [ $GITHUB_REF == "refs/heads/master" ]; then
    VERSION=next
else if [[ $GITHUB_REF == "refs/heads/*" ]]; then
    VERSION="branch/$(echo ${GITHUB_REF} | sed -e 's*^refs/heads/v**')"
else
    VERSION=whoops
fi fi fi

TMPDIR=$(mktemp -d)
mkdir -p ${TMPDIR}/scope
cp ${DIR}/cli/build/{scope,ldscope,libscope.so,scope.yml,scope_protocol.yml} ${TMPDIR}/scope/
cd ${TMPDIR} && tar cfz scope.tgz scope
cd scope && md5sum scope > scope.md5 && cd -
md5sum scope.tgz > scope.tgz.md5
if [ -n "${LATEST}" ]; then
    echo $LATEST > ${TMPDIR}/latest
    aws cp ${TMPDIR}/latest s3://io.cribl.cdn/dl/scope/
fi
aws s3 cp scope/scope s3://io.cribl.cdn/dl/scope/$VERSION/
aws s3 cp scope.tgz s3://io.cribl.cdn/dl/scope/$VERSION/
aws s3 cp scope/scope.md5 s3://io.cribl.cdn/dl/scope/$VERSION/
aws s3 cp scope.tgz.md5 s3://io.cribl.cdn/dl/scope/$VERSION/

aws cloudfront create-invalidation --distribution-id ${CF_DISTRIBUTION_ID} --paths '/dl/scope/'"$VERSION"'/*'