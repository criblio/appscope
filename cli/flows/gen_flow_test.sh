#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ARCH=$(uname -m)
GOBIN=$(pwd)/.gobin/$ARCH

BINDIR=../bin/linux/$ARCH
SCOPE=${BINDIR}/scope

GO_BINDATA=${GOBIN}/go-bindata

$SCOPE run -p -- curl --http1.1 -so /dev/null https://cribl.io/
cwd=$(pwd); cd $($SCOPE hist -d) && $GO_BINDATA -pkg flows -o ${DIR}/flow_file_test.go -fs . payloads; cd -
