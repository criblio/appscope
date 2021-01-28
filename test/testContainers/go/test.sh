#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd /go/net && openssl genrsa -out server.key 2048 && \
                openssl ecparam -genkey -name secp384r1 -out server.key && \
                openssl req -new -x509 -sha256 -key server.key -out server.crt \
    -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"


source  ~/.gvm/scripts/gvm

# for ver in $GO_UNSUPPORTED_VERSIONS; do
#     gvm install go$ver -B
#     gvm use go$ver
#     echo "Building test commands"
#     source ${DIR}/build_go.sh
#     cd ${DIR}
#     echo "Running tests..."
#     bash ${DIR}/unsupported/test_go.sh
#     if [ $? -gt 0 ]; then
#         # exit $?
#         echo "FAILED"
#     fi
#     cd ${DIR}
# done

for ver in $GO_SUPPORTED_VERSIONS; do
    gvm install go$ver -B
    gvm use go$ver
    echo "Building test commands"
    source ${DIR}/build_go.sh
    cd ${DIR}
    echo "Running tests..."
    bash ${DIR}/test_go.sh
    if [ $? -gt 0 ]; then
        # exit $?
        echo "FAILED"
    fi
    cd ${DIR}
done