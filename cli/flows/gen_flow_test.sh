#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GOBIN=$(pwd)/.gobin/x86_64

BINDIR=../bin/linux/x86_64
SCOPE=${BINDIR}/scope

GO_BINDATA=${GOBIN}/go-bindata

$SCOPE run -p -- curl --http1.1 -so /dev/null https://cribl.io/
cwd=$(pwd); cd $($SCOPE hist -d) && $GO_BINDATA -pkg flows -o ${DIR}/flow_file_test.go -fs . payloads; cd -
